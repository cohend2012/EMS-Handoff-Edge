"""
llm.py — one client for every language-model provider (Groq, Ollama, OpenRouter).

Purpose
    Send a prompt, get back text that should be JSON, and fail fast and clearly
    when the provider is down, slow, or returns nothing.

What changed from the intern version (handoff.py)
    * Explicit per-request timeout. The OpenAI SDK default is 600 s.
    * The SDK's own retries are off (max_retries=0). This module retries
      instead, so there are never 3 x 3 stacked attempts.
    * JSON mode (response_format=json_object) on Groq and Ollama, so the model
      cannot wrap the answer in prose.
    * Ollama: thinking is switched off in the request, not only by the
      "/no_think" hint in the prompt. A reasoning trace that eats the token
      budget used to produce empty replies.
    * The model name comes from Settings. The menu and the code can no longer
      disagree (the Jetson copy said qwen3:8b but ran llama3.2:1b).

Inputs
    Settings (provider, keys, model names, timeout). An OpenAI-compatible client
    can be injected for tests.

Outputs
    ``LLMClient.complete_json(prompt)`` returns cleaned text.
    ``LLMClient.check()`` returns (ok, message) for the provider status light.

Threads
    Called from the report worker thread. Holds no shared state.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from typing import Any, Callable, Dict, Optional, Tuple

from .config import Settings

logger = logging.getLogger(__name__)

BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
JSON_MODE_PROVIDERS = {"groq", "ollama"}

# Groq retires models over time. When the configured model is gone, the first
# of these that the account can use is picked instead (and the app says so).
GROQ_LLM_PREFERENCES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]
_NOT_CHAT = ("whisper", "guard", "tts", "playai", "orpheus", "embed", "distil", "compound", "allam")
_RESOLVED: dict = {}   # (provider, configured model) -> (model actually used, note)

# Ollama's model server (llama-server) is killed and restarted by systemd when the Jetson
# runs out of memory; the batch run of 2026-10-24 lost four reports to it. While it comes
# back, the API refuses connections or answers 500 with one of these. The client waits
# these many seconds between retries, 90 s in all, before giving up.
RESTART_WAITS = (2, 3, 5, 10, 15, 20, 35)
_RESTART_SIGNS = re.compile(r"unexpectedly stopped|unexpected eof|connection reset|connection refused|"
                            r"server (?:is )?loading|model is loading|runner.*(?:stopped|exited)", re.I)


def pick_model(available, wanted: str, preferences=GROQ_LLM_PREFERENCES) -> Tuple[Optional[str], str]:
    """Choose the model to use from what the provider offers. Returns (model, note-for-the-user)."""
    available = set(available)
    if wanted in available:
        return wanted, ""
    for p in preferences:
        if p in available:
            return p, f"'{wanted}' is no longer offered; using '{p}'"
    chat = sorted(m for m in available if not any(x in m.lower() for x in _NOT_CHAT))
    big = [m for m in chat if any(t in m.lower() for t in ("120b", "70b", "maverick", "32b"))]
    choice = (big or chat or [None])[0]
    if choice:
        return choice, f"'{wanted}' is no longer offered; using '{choice}'"
    return None, f"no chat model is available (wanted '{wanted}')"


# Ollama models in order of preference, largest first. When the configured model
# cannot fit in free memory, the first of these that is pulled and fits is used.
# Models under ~1.7B parameters are not on the list: in a test llama3.2:1b echoed the
# schema's placeholder text as findings. A wrong handoff is worse than a clear error.
# Neither is "qwen3:4b": that tag is the thinking-only variant. It reasons in plain
# text before the JSON even with thinking switched off, and at ~18 tok/s on the
# Orin Nano that overran the 180 s budget every time. "qwen3:4b-instruct" is the
# same size, answers directly, and is the model to pull on an 8 GB Jetson.
# Both can still be chosen on purpose with OLLAMA_MODEL.
OLLAMA_PREFERENCES = ["qwen3:8b", "qwen3:4b-instruct", "llama3.2:3b", "qwen3:1.7b"]


def ollama_need_gb(size_gb: float) -> float:
    """Memory a model needs to run: weights, the context cache, and the CUDA runtime.

    Measured on the 8 GB Orin Nano: qwen3:4b (2.5 GB on disk) was killed with
    3.4 GB free and ran with 3.7 GB, so a flat factor is too optimistic for small
    models. Weights x 1.25 plus 0.8 GB for the context cache and runtime fits
    what was seen (4b -> 3.9 GB, 8b -> 7.3 GB).
    """
    return round(size_gb * 1.25 + 0.8, 2)


def pick_ollama_model(models: Dict[str, float], wanted: str,
                      free_gb: Optional[float], pinned: bool = False) -> Tuple[Optional[str], str]:
    """Choose the Ollama model to use from {name: size GB}. Returns (model, note-for-the-user).

    The wanted model is kept when it is pulled and fits in free memory (or memory
    is unknown). Otherwise the largest preferred model that is pulled and fits is
    used, and the note says why. A model that does not fit gets killed while
    loading, which Ollama reports as HTTP 500 "signal: killed".

    ``pinned`` (OLLAMA_MODEL was set): the wanted model is the only acceptable one.
    It is returned even when it does not fit, with a warning, and None when it is
    not pulled. Nothing is ever substituted for a pinned model: a batch that must
    run on one model, or a person who chose one, gets that model or an error.
    """
    want = wanted if ":" in wanted else f"{wanted}:latest"
    if want not in models:
        return None, (f"'{wanted}' is not pulled (OLLAMA_MODEL pins it): run 'ollama pull {wanted}', "
                      "or unset OLLAMA_MODEL to let the app choose a pulled model") if pinned else ""
    if free_gb is None or ollama_need_gb(models[want]) <= free_gb:
        return wanted, ""
    need = ollama_need_gb(models[want])
    if pinned:
        return wanted, (f"'{wanted}' needs ~{need:.1f} GB but only {free_gb:.1f} GB is free; it is pinned by "
                        "OLLAMA_MODEL so nothing else will be used. It may be killed while loading. "
                        "Close other apps, or add swap (see README).")
    for name in OLLAMA_PREFERENCES:
        if name in models and name != want and ollama_need_gb(models[name]) <= free_gb:
            return name, (f"'{wanted}' needs ~{need:.1f} GB but only {free_gb:.1f} GB is free; "
                          f"using '{name}'")
    pulled = [f"{n} ~{ollama_need_gb(models[n]):.1f} GB" for n in OLLAMA_PREFERENCES if n in models and n != want]
    what = f"nothing pulled fits ({', '.join(pulled)})" if pulled else \
        "no smaller model is pulled (ollama pull qwen3:4b-instruct, or qwen3:1.7b for a small machine)"
    return wanted, (f"'{wanted}' needs ~{need:.1f} GB but only {free_gb:.1f} GB is free; {what}. "
                    "It may be killed while loading. Close other apps, or add swap (see README).")


def same_model(a: Optional[str], b: Optional[str]) -> bool:
    """'qwen3:8b' == 'qwen3:8b'; a bare name means its ':latest' tag."""
    if not a or not b:
        return False
    norm = lambda m: m if ":" in m else f"{m}:latest"  # noqa: E731
    return norm(a) == norm(b)


def _usage_dict(resp, took_s: float) -> dict:
    usage = getattr(resp, "usage", None)
    out = getattr(usage, "completion_tokens", None)
    if not out:
        return {}
    d = {"tokens_in": getattr(usage, "prompt_tokens", 0) or 0, "tokens_out": out}
    if took_s > 0:
        d["tok_s"] = round(out / took_s, 1)
    return d


def _usage_note(resp, took_s: float) -> str:
    """': 2100 prompt + 850 output tokens, 11 tok/s' when the provider reports usage; else ''.

    The output rate says where a slow local model spends its time: a low rate
    means generation (or a model running on the CPU), a normal rate with a long
    total means the model was still loading.
    """
    usage = getattr(resp, "usage", None)
    out = getattr(usage, "completion_tokens", None)
    if not out:
        return ""
    rate = f", {out / took_s:.0f} tok/s" if took_s > 0 else ""
    return f": {getattr(usage, 'prompt_tokens', 0) or 0} prompt + {out} output tokens{rate}"


class LLMError(RuntimeError):
    """A provider failure, worded for the person using the app."""


def available_memory_gb() -> Optional[float]:
    """Free RAM in GB from /proc/meminfo (Linux), else None. On a Jetson the GPU shares this RAM."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return None


def strip_to_json(text: Optional[str]) -> str:
    """Cut LLM output down to the JSON it contains.

    Removes a reasoning-model ``<think>...</think>`` trace and Markdown fences,
    then keeps the outermost {...} or [...] block if prose surrounds it.
    """
    if not text:
        return ""
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    text = text.replace("```json", "").replace("```", "").strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        return text
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    return text[start:end + 1] if end > start else text[start:]


class LLMClient:
    def __init__(self, settings: Settings, client: Any = None, attempts: int = 3):
        self.settings = settings
        self.provider = settings.provider
        self.model = settings.llm_model()
        self.attempts = max(1, attempts)
        self.last_usage: dict = {}     # tokens_in / tokens_out / tok_s of the last reply, when reported
        self.last_recovery: dict = {}  # set when the last reply came after waiting for Ollama to restart
        self.on_wait: Optional[Callable[[str], None]] = None   # progress text while waiting for a restart
        self._client = client if client is not None else self._make_client()

    # -- construction -----------------------------------------------------
    def _make_client(self):
        from openai import OpenAI

        s = self.settings
        if self.provider == "ollama":
            base_url, key = s.ollama_url, "ollama"
        elif self.provider == "groq":
            base_url, key = BASE_URLS["groq"], s.groq_api_key
        else:
            base_url, key = BASE_URLS["openrouter"], s.openrouter_api_key
        if not key:
            raise LLMError(f"No API key set for {self.provider}. Add it to the .env file.")
        return OpenAI(base_url=base_url, api_key=key,
                      timeout=s.llm_timeout_s, max_retries=0)

    # -- model selection --------------------------------------------------
    def list_models(self, timeout: float = 10) -> set:
        client = self._client.with_options(timeout=timeout) if hasattr(self._client, "with_options") else self._client
        return {m.id for m in client.models.list().data}

    def resolve_model(self) -> str:
        """Pick the model actually used; returns a note for the user when it differs.

        Groq: if the configured model was retired, the best available one (cached).
        Ollama: if the configured model does not fit in free memory, the largest
        pulled model that does (checked per report, because free memory changes).
        """
        if self.provider == "ollama":
            try:
                models = self._ollama_models()
            except Exception as e:  # noqa: BLE001 - Ollama down: the call itself will say so
                logger.debug("Ollama model list unavailable (%s); using %s", e, self.model)
                return ""
            pinned = self.settings.ollama_model_pinned
            model, note = pick_ollama_model(models, self.settings.ollama_model, self._ollama_free_gb(), pinned)
            if model is None and pinned:
                raise LLMError(f"Ollama: {note}.")
            if model:
                self.model = model
            if note:
                logger.warning("Ollama: %s", note)
            return note
        if self.provider != "groq":
            return ""
        key = (self.provider, self.settings.llm_model())
        if key not in _RESOLVED:
            try:
                model, note = pick_model(self.list_models(), self.model)
            except Exception as e:  # noqa: BLE001 - cannot list: keep the configured model
                logger.debug("Model list unavailable (%s); using %s", e, self.model)
                return ""
            if model is None:
                return note
            _RESOLVED[key] = (model, note)
            if note:
                logger.warning("Groq: %s. Set GROQ_LLM_MODEL in .env to choose another.", note)
        self.model, note = _RESOLVED[key]
        return note

    def preload(self, keep_alive: str = "30m") -> str:
        """Ollama: load the model into memory now, so the first report does not wait for it.

        Ollama unloads a model 5 minutes after its last use by default; ``keep_alive``
        asks it to keep this one for longer. Returns the model-choice note, if any.
        Other providers have nothing to preload.
        """
        if self.provider != "ollama":
            return ""
        note = self.resolve_model()
        root = self.settings.ollama_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        body = json.dumps({"model": self.model, "keep_alive": keep_alive}).encode()
        req = urllib.request.Request(f"{root}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.settings.llm_timeout_s) as r:
            r.read()
        logger.info("Ollama %s loaded in %.1fs (kept for %s)", self.model, time.perf_counter() - t0, keep_alive)
        return note

    # -- calls ------------------------------------------------------------
    def complete_json(self, prompt: str, temperature: float = 0.0,
                      max_tokens: int = 4096) -> str:
        """Return the model's JSON text, retrying empty replies and dropped connections."""
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        self.resolve_model()
        kwargs = dict(model=self.model, temperature=temperature, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if self.provider in JSON_MODE_PROVIDERS:
            kwargs["response_format"] = {"type": "json_object"}
        if self.provider == "ollama":
            kwargs["extra_body"] = {"think": False}
        if self.provider == "groq" and "gpt-oss" in self.model:
            # gpt-oss reasons before answering; low effort is faster and leaves room for the JSON
            kwargs["reasoning_effort"] = "low"
            kwargs["max_tokens"] = max(max_tokens, 8192)

        last_problem = "no reply"
        self.last_recovery = {}
        restart = {"waits": list(RESTART_WAITS), "waited": 0.0, "retries": 0}
        attempt = 0
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                took = time.perf_counter() - started
                replied_with = getattr(resp, "model", None)
                if (self.provider == "ollama" and self.settings.ollama_model_pinned
                        and isinstance(replied_with, str) and not same_model(replied_with, self.model)):
                    raise LLMError(f"Ollama answered with '{replied_with}' but OLLAMA_MODEL pins "
                                   f"'{self.model}'. The report was discarded.")
                text = strip_to_json(resp.choices[0].message.content)
                self.last_usage = _usage_dict(resp, took)
                logger.info("LLM %s/%s replied in %.1fs (attempt %d)%s", self.provider,
                            self.model, took, attempt, _usage_note(resp, took))
                if text.strip():
                    return text
                last_problem = "the model returned an empty reply"
            except APITimeoutError:
                # A timeout is not transient: repeating the same request wastes another 180 s.
                hint = (" The model may be too big for this machine or running on the CPU; "
                        "check 'ollama ps' while a report runs." if self.provider == "ollama" else "")
                raise LLMError(f"{self.provider} gave no reply within "
                               f"{self.settings.llm_timeout_s:.0f} s.{hint}")
            except APIConnectionError:
                if self.provider == "ollama":
                    self._wait_for_restart(restart, "cannot reach Ollama")
                    continue
                last_problem = self._connection_hint()
            except APIStatusError as e:
                if (self.provider == "ollama" and (getattr(e, "status_code", 0) or 0) >= 500
                        and _RESTART_SIGNS.search(str(e))):
                    self._wait_for_restart(restart, "the model server stopped")
                    continue
                # A model without JSON mode answers 400: ask again without it (once).
                if getattr(e, "status_code", None) == 400 and "reasoning_effort" in kwargs \
                        and "reasoning" in str(e).lower():
                    logger.warning("%s rejected reasoning_effort; retrying without it", self.model)
                    kwargs.pop("reasoning_effort")
                    continue
                if (getattr(e, "status_code", None) == 400 and "response_format" in kwargs
                        and re.search(r"json|response_format", str(e), re.I)):
                    logger.warning("%s does not support JSON mode; retrying without it", self.model)
                    kwargs.pop("response_format")
                    continue
                # Other 4xx are not worth retrying: bad key, unknown model, bad request.
                raise LLMError(self._status_hint(e)) from e
            logger.warning("LLM attempt %d/%d failed: %s", attempt, self.attempts, last_problem)
            if attempt >= self.attempts:
                break
            time.sleep(2 ** (attempt - 1))
        raise LLMError(f"{self.provider} failed after {self.attempts} tries: {last_problem}")

    def _wait_for_restart(self, restart: dict, why: str) -> None:
        """Sleep for the next wait in the restart schedule, telling the app; raise when it is used up."""
        if not restart["waits"]:
            raise LLMError(f"Ollama's model server did not come back within {sum(RESTART_WAITS)} s ({why}). "
                           "Check 'systemctl status ollama' and free memory; the Jetson may be out of memory.")
        wait = restart["waits"].pop(0)
        restart["waited"] += wait
        restart["retries"] += 1
        self.last_recovery = {"model_server": "restarted", "reason": why,
                              "waited_s": restart["waited"], "retries": restart["retries"]}
        msg = f"Model server restarting ({why}); retrying in {wait} s, {restart['waited']:.0f} s waited"
        logger.warning(msg)
        if self.on_wait:
            self.on_wait(msg)
        time.sleep(wait)

    # -- health check -----------------------------------------------------
    def check(self) -> Tuple[bool, str]:
        """Is the provider reachable and does it have the configured model?"""
        try:
            if self.provider == "ollama":
                return self._check_ollama()
            ids = self.list_models()
            if self.provider == "groq":
                model, note = pick_model(ids, self.settings.llm_model())
                if model is None:
                    return False, f"groq is reachable but {note}."
                _RESOLVED[(self.provider, self.settings.llm_model())] = (model, note)
                self.model = model
                return True, f"groq ready ({model})" + (f"; {note}" if note else "")
            if self.model not in ids:
                return False, f"{self.provider} is reachable but has no model '{self.model}'."
            return True, f"{self.provider} ready ({self.model})"
        except LLMError as e:
            return False, str(e)
        except Exception as e:  # noqa: BLE001 - any failure means "not ready"
            name = type(e).__name__
            if "Auth" in name or "Permission" in name:
                return False, f"{self.provider} rejected the API key."
            if "Connection" in name or "Timeout" in name:
                return False, self._connection_hint()
            return False, f"{self.provider} check failed: {name}: {e}"

    def _ollama_root(self) -> str:
        root = self.settings.ollama_url.rstrip("/")
        return root[:-3] if root.endswith("/v1") else root

    def _ollama_models(self) -> Dict[str, float]:
        """{model name: size in GB} for every model Ollama has pulled."""
        with urllib.request.urlopen(f"{self._ollama_root()}/api/tags", timeout=5) as r:
            return {m.get("name", ""): m.get("size", 0) / 1e9 for m in json.load(r).get("models", [])}

    def _ollama_free_gb(self) -> Optional[float]:
        """Memory a model can be loaded into: what is free now plus what models Ollama
        already holds, since loading another model evicts them. After preload, free
        memory alone would say the model that is already running does not fit."""
        free = available_memory_gb()
        if free is None:
            return None
        try:
            with urllib.request.urlopen(f"{self._ollama_root()}/api/ps", timeout=5) as r:
                loaded = sum(m.get("size", 0) for m in json.load(r).get("models", [])) / 1e9
        except Exception as e:  # noqa: BLE001 - optional refinement; free memory alone still works
            logger.debug("Ollama /api/ps unavailable (%s)", e)
            loaded = 0.0
        return round(free + loaded, 2)

    def ollama_inventory(self) -> dict:
        """What the Settings screen needs: every pulled model with the memory it needs, and
        what is free. {"models": [{"name", "size_gb", "need_gb", "fits"}], "free_gb": float|None}."""
        models = self._ollama_models()
        free_gb = self._ollama_free_gb()
        out = []
        for name, size in sorted(models.items(), key=lambda kv: -kv[1]):
            need = ollama_need_gb(size)
            out.append({"name": name, "size_gb": round(size, 2), "need_gb": need,
                        "fits": free_gb is None or need <= free_gb})
        return {"models": out, "free_gb": free_gb}

    def ollama_model_status(self, name: str) -> Tuple[bool, str]:
        """Can ``name`` be chosen as the Ollama model right now? (ok, why).

        Refuses a model that is not pulled or does not fit in free memory, with the
        reason, so a person never saves a choice that would be substituted or killed
        while loading. Ollama being down is a reason too."""
        try:
            inv = self.ollama_inventory()
        except Exception as e:  # noqa: BLE001 - urllib errors, bad JSON: all mean "cannot ask Ollama"
            return False, f"Cannot ask Ollama which models are pulled: {self._connection_hint()} ({e})"
        want = name if ":" in name else f"{name}:latest"
        entry = next((m for m in inv["models"] if m["name"] == want), None)
        if entry is None:
            return False, f"'{name}' is not pulled. Run: ollama pull {name}"
        if not entry["fits"]:
            return False, (f"'{name}' needs ~{entry['need_gb']:.1f} GB but only {inv['free_gb']:.1f} GB is free; "
                           "it would be killed while loading. Close other apps, or choose a smaller model.")
        free = f", {inv['free_gb']:.1f} GB free" if inv["free_gb"] is not None else ""
        return True, f"'{name}' is pulled and fits (needs ~{entry['need_gb']:.1f} GB{free})"

    def _check_ollama(self) -> Tuple[bool, str]:
        models = self._ollama_models()
        wanted, pinned = self.settings.ollama_model, self.settings.ollama_model_pinned
        want = wanted if ":" in wanted else f"{wanted}:latest"
        if want not in models:
            how = " (pinned by OLLAMA_MODEL)" if pinned else ""
            return False, f"Ollama is running but '{wanted}'{how} is not pulled. Run: ollama pull {wanted}"
        free_gb = self._ollama_free_gb()
        model, note = pick_ollama_model(models, wanted, free_gb, pinned)
        self.model = model or self.model
        msg = f"ollama ready ({self.model}{', pinned' if pinned else ''})"
        if free_gb is not None:
            msg += f"; needs ~{ollama_need_gb(models.get(self.model, 0)):.1f} GB, {free_gb:.1f} GB free"
        return True, msg + (f"; {note}" if note else "")

    # -- wording ------------------------------------------------------------
    def _connection_hint(self) -> str:
        if self.provider == "ollama":
            return f"cannot reach Ollama at {self.settings.ollama_url}. Is 'ollama serve' running?"
        return f"cannot reach {self.provider}. Check the internet connection."

    def _status_hint(self, e) -> str:
        code = getattr(e, "status_code", "?")
        if code in (401, 403):
            return f"{self.provider} rejected the API key (HTTP {code}). Create a new key and update .env."
        if code == 404:
            return (f"{self.provider} does not know the model '{self.model}' (HTTP 404). "
                    "Set GROQ_LLM_MODEL in .env to a model your account lists.")
        if code == 429:
            # Keep the provider's own words: Groq says which limit (per minute or per day)
            # and when to try again, and a batch runner needs that to back off or stop.
            detail = re.sub(r"\s+", " ", str(e))[:300]
            return f"{self.provider} rate limit reached (HTTP 429). Wait a minute and retry. {detail}"
        if self.provider == "ollama" and re.search(r"killed|terminated|out of memory", str(e), re.I):
            free = available_memory_gb()
            return (f"Ollama ran out of memory loading '{self.model}'"
                    + (f" ({free:.1f} GB free)" if free is not None else "")
                    + ". Close other apps, or pull a smaller model (ollama pull qwen3:4b-instruct): "
                      "the app switches to it by itself.")
        msg = re.sub(r"\s+", " ", str(e))[:200]
        return f"{self.provider} error HTTP {code}: {msg}"
