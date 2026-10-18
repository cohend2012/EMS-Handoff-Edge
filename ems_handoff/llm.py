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
from typing import Any, Optional, Tuple

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


class LLMError(RuntimeError):
    """A provider failure, worded for the person using the app."""


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
        """Groq only: if the configured model was retired, switch to the best available one (cached)."""
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

        last_problem = "no reply"
        for attempt in range(1, self.attempts + 1):
            started = time.perf_counter()
            try:
                resp = self._client.chat.completions.create(**kwargs)
                text = strip_to_json(resp.choices[0].message.content)
                logger.info("LLM %s/%s replied in %.1fs (attempt %d)", self.provider,
                            self.model, time.perf_counter() - started, attempt)
                if text.strip():
                    return text
                last_problem = "the model returned an empty reply"
            except APITimeoutError:
                last_problem = f"no reply within {self.settings.llm_timeout_s:.0f} s"
            except APIConnectionError:
                last_problem = self._connection_hint()
            except APIStatusError as e:
                # A model without JSON mode answers 400: ask again without it (once).
                if (getattr(e, "status_code", None) == 400 and "response_format" in kwargs
                        and re.search(r"json|response_format", str(e), re.I)):
                    logger.warning("%s does not support JSON mode; retrying without it", self.model)
                    kwargs.pop("response_format")
                    continue
                # Other 4xx are not worth retrying: bad key, unknown model, bad request.
                raise LLMError(self._status_hint(e)) from e
            logger.warning("LLM attempt %d/%d failed: %s", attempt, self.attempts, last_problem)
            if attempt < self.attempts:
                time.sleep(2 ** (attempt - 1))
        raise LLMError(f"{self.provider} failed after {self.attempts} tries: {last_problem}")

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

    def _check_ollama(self) -> Tuple[bool, str]:
        root = self.settings.ollama_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[:-3]
        with urllib.request.urlopen(f"{root}/api/tags", timeout=5) as r:
            names = {m.get("name", "") for m in json.load(r).get("models", [])}
        want = self.model if ":" in self.model else f"{self.model}:latest"
        if want not in names:
            return False, f"Ollama is running but '{self.model}' is not pulled. Run: ollama pull {self.model}"
        return True, f"ollama ready ({self.model})"

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
            return f"{self.provider} rate limit reached (HTTP 429). Wait a minute and retry."
        msg = re.sub(r"\s+", " ", str(e))[:200]
        return f"{self.provider} error HTTP {code}: {msg}"
