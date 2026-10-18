"""
doctor.py — "is this machine ready?" checks, run by ``python -m ems_handoff --check``.

Checks, in the order a demo needs them:
    1. Python packages import (including PortAudio for the mic)
    2. Microphone found, and which one will be used
    3. Live caption model (Vosk) present, otherwise the Whisper fallback is used
    4. Whisper model files cached locally (needed for offline transcription)
    5. AI provider reachable with the configured model (and Groq key if used)
    6. Optional textbook store present

Output is plain text, one line per check: OK / WARN / FAIL with a fix.
Returns exit code 0 when nothing FAILs.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path

from .config import Settings


def _line(status: str, what: str, detail: str = "") -> None:
    print(f"  [{status:<4}] {what}" + (f": {detail}" if detail else ""))


def _hf_cache() -> Path:
    base = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if base:
        return Path(base)
    home = os.environ.get("HF_HOME")
    return Path(home, "hub") if home else Path.home() / ".cache" / "huggingface" / "hub"


def whisper_cached(size: str) -> bool:
    cache = _hf_cache()
    return cache.exists() and any(f"faster-whisper-{size}" in p.name for p in cache.iterdir())


def run_checks(settings: Settings) -> int:
    failures = 0
    print("HANDOFF.AI setup check\n")

    print("Packages")
    for mod in ("flask", "openai", "numpy", "scipy", "faster_whisper", "vosk", "spellchecker"):
        try:
            importlib.import_module(mod)
            _line("OK", mod)
        except Exception as e:  # noqa: BLE001
            level = "WARN" if mod in ("vosk", "spellchecker") else "FAIL"
            failures += level == "FAIL"
            _line(level, mod, f"{type(e).__name__}: {e}. Run: pip install -r requirements.txt")

    print("\nMicrophone")
    try:
        from .audio.capture import find_input_device, list_input_devices
        for i, name, rate in list_input_devices():
            _line("..", f"{i}: {name}", f"{rate:.0f} Hz")
        i, name, _ = find_input_device(settings.mic_device)
        _line("OK", "will use", f"[{i}] {name} (HANDOFF_MIC_DEVICE={settings.mic_device or 'auto'})")
    except Exception as e:  # noqa: BLE001
        failures += 1
        _line("FAIL", "microphone", str(e))

    print("\nLive captions")
    from .audio.listeners import vosk_available
    if vosk_available(settings.vosk_model_dir):
        _line("OK", "Vosk model", str(settings.vosk_model_dir))
    else:
        _line("WARN", "Vosk model missing", f"{settings.vosk_model_dir}. Captions fall back to "
              f"Whisper {settings.caption_whisper_model}. Run: python scripts/download_models.py")

    print("\nTranscription")
    mode = settings.effective_transcriber()
    if mode == "groq":
        _line("OK", "final transcript", f"Groq {settings.groq_whisper_model} (local Whisper is the backup)")
    else:
        _line("OK", "final transcript", f"local faster-whisper {settings.local_whisper_model}")
    for size in {settings.local_whisper_model, settings.caption_whisper_model}:
        if whisper_cached(size):
            _line("OK", f"Whisper {size} cached")
        else:
            _line("WARN", f"Whisper {size} not cached", "it downloads on first use (needs internet). "
                  "Run: python scripts/download_models.py")

    print("\nAI provider")
    from .llm import LLMClient, LLMError
    try:
        good, msg = LLMClient(settings).check()
    except LLMError as e:
        good, msg = False, str(e)
    failures += not good
    _line("OK" if good else "FAIL", settings.provider, msg)

    print("\nTextbook guidance (optional)")
    from .vector_store import TextbookStore
    if TextbookStore.exists(str(settings.textbook_dir)):
        _line("OK", "textbook store", str(settings.textbook_dir))
    else:
        _line("..", "no textbook store", "guidance is skipped. See README, 'Textbook guidance'.")

    print("\n" + ("Ready." if not failures else f"{failures} problem(s) to fix before a demo."))
    return 1 if failures else 0
