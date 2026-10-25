"""
config.py — every tunable setting for HANDOFF.AI, in one place.

Purpose
    Replace the values that used to be hardcoded across the code (mic device 24,
    model names, the Ollama URL, the stop phrase) with one ``Settings`` object.

Inputs
    Environment variables, filled by ``load_env_files()`` from two files:
      1. ~/.config/handoff-ai/secrets.env   API keys, set once for every copy of
         the app (written by scripts/set_groq_key.sh). HANDOFF_SECRETS overrides the path.
      2. <project>/.env                     everything else (see .env.example)
    A value already in the environment always wins. The secrets file loads first,
    so a blank GROQ_API_KEY= line in .env cannot hide the shared key.

Outputs
    ``load_settings()`` returns an immutable ``Settings``. To change a value at
    runtime (for example the provider picked in the app), use
    ``dataclasses.replace(settings, provider="ollama")``.

Threads
    None. Settings are read-only once built.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SECRETS_FILE = Path(os.environ.get("HANDOFF_SECRETS", "~/.config/handoff-ai/secrets.env")).expanduser()


def load_env_files() -> None:
    """Load the shared secrets file, then the project .env (neither overrides the real environment)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    if SECRETS_FILE.exists():
        load_dotenv(SECRETS_FILE)
    load_dotenv(PROJECT_ROOT / ".env")

PROVIDERS = ("groq", "ollama", "openrouter")
TRANSCRIBERS = ("auto", "groq", "local")
CAPTION_ENGINES = ("auto", "vosk", "whisper", "off")


def _bool(value: Optional[str], default: bool) -> bool:
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _float(value: Optional[str], default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default


def _int(value: Optional[str], default: int) -> int:
    try:
        return int(value) if value not in (None, "") else default
    except ValueError:
        return default


def _choice(value: Optional[str], allowed: tuple, default: str) -> str:
    v = (value or "").strip().lower()
    return v if v in allowed else default


@dataclass(frozen=True)
class Settings:
    # --- which AI writes the report -------------------------------------
    provider: str = "groq"                       # groq | ollama | openrouter
    groq_api_key: str = field(default="", repr=False)
    openrouter_api_key: str = field(default="", repr=False)
    groq_llm_model: str = "llama-3.3-70b-versatile"
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"
    ollama_model_pinned: bool = False             # True when OLLAMA_MODEL is set: use exactly it, never a substitute
    openrouter_model: str = "meta-llama/llama-3.2-3b-instruct:free"
    llm_timeout_s: float = 180.0                 # per request; local models are slow

    # --- final transcript (runs once, after DONE) -------------------------
    transcriber: str = "auto"                    # auto | groq | local
    groq_whisper_model: str = "whisper-large-v3-turbo"
    local_whisper_model: str = "large-v3-turbo"
    local_whisper_device: str = "cpu"            # cpu | cuda
    local_whisper_compute: str = "int8"

    # --- live listening ----------------------------------------------------
    mic_device: str = ""                         # index, or part of the name; "" = auto (prefers USB)
    caption_engine: str = "auto"                 # auto | vosk | whisper | off
    vosk_model_dir: Path = PROJECT_ROOT / "models" / "vosk-model-small-en-us-0.15"
    caption_whisper_model: str = "tiny.en"       # fallback caption model when Vosk is missing
    stop_phrase: str = "confirm report ready"
    max_recording_s: int = 600                   # auto-DONE after 10 minutes

    # --- app ---------------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 5050                             # not 5000: the old EMS-Handoff app uses 5000
    output_dir: Path = PROJECT_ROOT / "output"
    textbook_dir: Path = PROJECT_ROOT / "database"
    use_lexicon: bool = True
    use_textbook_rag: bool = True

    # --- replay (recorded call-sets through the live pipeline) -------------
    replay_mode_enabled: bool = False            # HANDOFF_REPLAY=1; off = today's app, byte for byte
    replay_callsets_dir: Path = Path("~/callsets").expanduser()
    replay_index: str = ""                       # optional corpus index for burst timing (callset_info.json comes first)
    replay_provider: str = "groq"                # per replay request; this is the default
    replay_groq_model: str = "openai/gpt-oss-120b"
    replay_speed: float = 1.0                    # gaps are divided by this; bursts always play at real time
    replay_sink: str = "handoff_replay"          # PipeWire null sink the burst audio is played into
    replay_quiet_online_s: float = 5.0
    replay_quiet_offline_s: float = 25.0

    # --- derived helpers ---------------------------------------------------
    def has_key(self, provider: Optional[str] = None) -> bool:
        p = provider or self.provider
        if p == "groq":
            return bool(self.groq_api_key)
        if p == "openrouter":
            return bool(self.openrouter_api_key)
        return True  # ollama needs no key

    def llm_model(self, provider: Optional[str] = None) -> str:
        p = provider or self.provider
        return {"groq": self.groq_llm_model,
                "ollama": self.ollama_model,
                "openrouter": self.openrouter_model}[p]

    def effective_transcriber(self) -> str:
        """'groq' or 'local'. 'auto' uses Groq when a Groq key exists."""
        if self.transcriber in ("groq", "local"):
            return self.transcriber
        return "groq" if self.groq_api_key else "local"

    def public_dict(self) -> dict:
        """Settings safe to show in the app (no secrets)."""
        return {
            "provider": self.provider,
            "model": self.llm_model(),
            "model_pinned": self.provider == "ollama" and self.ollama_model_pinned,
            "keys": {p: self.has_key(p) for p in PROVIDERS},
            "transcriber": self.effective_transcriber(),
            "caption_engine": self.caption_engine,
            "mic_device": self.mic_device or "auto",
            "stop_phrase": self.stop_phrase,
            "max_recording_s": self.max_recording_s,
            "replay_mode": self.replay_mode_enabled,
        }


def load_settings(env: Optional[Mapping[str, str]] = None) -> Settings:
    """Build Settings from environment variables (or a supplied mapping)."""
    e = os.environ if env is None else env
    g = e.get
    defaults = Settings()
    return Settings(
        provider=_choice(g("HANDOFF_PROVIDER"), PROVIDERS, defaults.provider),
        groq_api_key=(g("GROQ_API_KEY") or "").strip(),
        openrouter_api_key=(g("OPENROUTER_API_KEY") or "").strip(),
        groq_llm_model=g("GROQ_LLM_MODEL") or defaults.groq_llm_model,
        ollama_url=g("OLLAMA_URL") or defaults.ollama_url,
        ollama_model=(g("OLLAMA_MODEL") or "").strip() or defaults.ollama_model,
        ollama_model_pinned=bool((g("OLLAMA_MODEL") or "").strip()),
        openrouter_model=g("OPENROUTER_MODEL") or defaults.openrouter_model,
        llm_timeout_s=_float(g("HANDOFF_LLM_TIMEOUT_S"), defaults.llm_timeout_s),
        transcriber=_choice(g("HANDOFF_TRANSCRIBER"), TRANSCRIBERS, defaults.transcriber),
        groq_whisper_model=g("GROQ_WHISPER_MODEL") or defaults.groq_whisper_model,
        local_whisper_model=g("LOCAL_WHISPER_MODEL") or defaults.local_whisper_model,
        local_whisper_device=g("LOCAL_WHISPER_DEVICE") or defaults.local_whisper_device,
        local_whisper_compute=g("LOCAL_WHISPER_COMPUTE") or defaults.local_whisper_compute,
        mic_device=(g("HANDOFF_MIC_DEVICE") or "").strip(),
        caption_engine=_choice(g("HANDOFF_CAPTIONS"), CAPTION_ENGINES, defaults.caption_engine),
        vosk_model_dir=Path(g("VOSK_MODEL_DIR")) if g("VOSK_MODEL_DIR") else defaults.vosk_model_dir,
        caption_whisper_model=g("CAPTION_WHISPER_MODEL") or defaults.caption_whisper_model,
        stop_phrase=(g("HANDOFF_STOP_PHRASE") or defaults.stop_phrase).strip().lower(),
        max_recording_s=_int(g("HANDOFF_MAX_RECORDING_S"), defaults.max_recording_s),
        host=g("HANDOFF_HOST") or defaults.host,
        port=_int(g("HANDOFF_PORT"), defaults.port),
        output_dir=Path(g("HANDOFF_OUTPUT_DIR")) if g("HANDOFF_OUTPUT_DIR") else defaults.output_dir,
        textbook_dir=Path(g("HANDOFF_TEXTBOOK_DIR")) if g("HANDOFF_TEXTBOOK_DIR") else defaults.textbook_dir,
        use_lexicon=_bool(g("HANDOFF_USE_LEXICON"), True),
        use_textbook_rag=_bool(g("HANDOFF_USE_TEXTBOOK_RAG"), True),
        replay_mode_enabled=_bool(g("HANDOFF_REPLAY"), False),
        replay_callsets_dir=Path(g("HANDOFF_CALLSETS")).expanduser() if g("HANDOFF_CALLSETS") else defaults.replay_callsets_dir,
        replay_index=(g("HANDOFF_CALLSET_INDEX") or "").strip(),
        replay_provider=_choice(g("HANDOFF_REPLAY_PROVIDER"), PROVIDERS, defaults.replay_provider),
        replay_groq_model=g("HANDOFF_REPLAY_GROQ_MODEL") or defaults.replay_groq_model,
        replay_speed=_float(g("HANDOFF_REPLAY_SPEED"), defaults.replay_speed),
        replay_sink=g("HANDOFF_REPLAY_SINK") or defaults.replay_sink,
        replay_quiet_online_s=_float(g("HANDOFF_REPLAY_QUIET_ONLINE_S"), defaults.replay_quiet_online_s),
        replay_quiet_offline_s=_float(g("HANDOFF_REPLAY_QUIET_OFFLINE_S"), defaults.replay_quiet_offline_s),
    )
