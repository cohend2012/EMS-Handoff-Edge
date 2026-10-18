"""
audio/listeners.py — live captions and the stop-phrase spotter.

Purpose
    Turn the live mic stream into
      1. words on screen while the paramedic talks (captions), and
      2. a DONE signal when they say the stop phrase (spotter),
    without ever blocking the recording.

Engines
    VoskCaptioner      Streaming Kaldi recognizer (offline, ~40 MB model). It
                       is built for real time on small ARM CPUs and returns
                       partial words several times a second. This is the default.
    VoskPhraseSpotter  A second Vosk recognizer limited by a grammar to the stop
                       phrase plus "[unk]". It runs in its own thread, so phrase
                       detection never waits on captions.
    WhisperCaptioner   Fallback when the Vosk model is not installed.
                       faster-whisper tiny.en, single pass (temperature 0), on a
                       window that always starts where the last committed text
                       ended. Every sample is covered and nothing is skipped.
                       The window is capped, so each pass takes bounded time.

Both captioners also report whether the stop phrase appears in the latest
words. The live session treats either source as DONE.

Inputs
    16 kHz float32 numpy blocks from AudioCapture.

Outputs
    ``Captioner.feed(block) -> CaptionUpdate`` and ``PhraseSpotter.feed(block) -> bool``.

Threads
    Each instance is used by exactly one listener thread. The Vosk model
    object is shared read-only and loaded once per process.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np

from ..config import Settings
from ..phrase import contains_phrase, words

logger = logging.getLogger(__name__)

RATE = 16000


@dataclass
class CaptionUpdate:
    final: str          # text that will not change any more
    partial: str        # the words still being recognized
    changed: bool
    heard_phrase: bool = False


def _pcm16(block: np.ndarray) -> bytes:
    return (np.clip(block, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _tail(text: str, n: int = 10) -> str:
    return " ".join(words(text)[-n:])


# ---------------------------------------------------------------------------
# Vosk
# ---------------------------------------------------------------------------
_VOSK_MODELS = {}
_VOSK_LOCK = threading.Lock()


def vosk_available(model_dir: Path) -> bool:
    try:
        import vosk  # noqa: F401
    except ImportError:
        return False
    return Path(model_dir, "conf").exists() or Path(model_dir, "am").exists()


def load_vosk_model(model_dir: Path):
    with _VOSK_LOCK:
        key = str(model_dir)
        if key not in _VOSK_MODELS:
            import vosk
            vosk.SetLogLevel(-1)
            logger.info("Loading Vosk model from %s", model_dir)
            _VOSK_MODELS[key] = vosk.Model(key)
        return _VOSK_MODELS[key]


class VoskCaptioner:
    name = "vosk"

    def __init__(self, model, phrase: str = ""):
        import vosk
        self._rec = vosk.KaldiRecognizer(model, RATE)
        self._phrase = phrase
        self._final = ""
        self._partial = ""

    def feed(self, block: np.ndarray) -> CaptionUpdate:
        before = (self._final, self._partial)
        if self._rec.AcceptWaveform(_pcm16(block)):
            text = json.loads(self._rec.Result()).get("text", "").strip()
            if text:
                self._final = f"{self._final} {text}".strip()
            self._partial = ""
        else:
            self._partial = json.loads(self._rec.PartialResult()).get("partial", "").strip()
        heard = bool(self._phrase) and contains_phrase(
            _tail(f"{self._final} {self._partial}"), self._phrase)
        return CaptionUpdate(self._final, self._partial, before != (self._final, self._partial), heard)


class VoskPhraseSpotter:
    """Grammar-limited recognizer: it can only answer 'the phrase' or '[unk]'."""

    def __init__(self, model, phrase: str, min_confidence: float = 0.65):
        import vosk
        self.phrase = phrase
        self.min_confidence = min_confidence
        self._rec = vosk.KaldiRecognizer(model, RATE, json.dumps([phrase, "[unk]"]))
        self._rec.SetWords(True)

    def feed(self, block: np.ndarray) -> bool:
        if not self._rec.AcceptWaveform(_pcm16(block)):
            return False
        result = json.loads(self._rec.Result())
        if result.get("text", "").strip() != self.phrase:
            return False
        confs = [w.get("conf", 0.0) for w in result.get("result", []) if w.get("word") != "[unk]"]
        conf = sum(confs) / len(confs) if confs else 0.0
        logger.info("Stop-phrase candidate heard (confidence %.2f)", conf)
        return conf >= self.min_confidence


# ---------------------------------------------------------------------------
# Whisper fallback
# ---------------------------------------------------------------------------
class WhisperCaptioner:
    """Gap-free rolling captions with a small Whisper model (fallback only)."""

    name = "whisper"

    def __init__(self, model, phrase: str = "", step_s: float = 1.0, commit_s: float = 8.0,
                 transcribe_fn: Optional[Callable[[np.ndarray], str]] = None):
        self._model = model
        self._phrase = phrase
        self._step = int(step_s * RATE)
        self._commit = int(commit_s * RATE)
        self._buf = np.zeros(0, np.float32)    # audio since the last commit
        self._since_run = 0
        self._final = ""
        self._partial = ""
        self._transcribe = transcribe_fn or self._whisper

    def _whisper(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language="en", beam_size=1, temperature=0.0,
                                             condition_on_previous_text=False,
                                             without_timestamps=True, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    def feed(self, block: np.ndarray) -> CaptionUpdate:
        self._buf = np.concatenate([self._buf, block])
        self._since_run += len(block)
        if self._since_run < self._step:
            return CaptionUpdate(self._final, self._partial, False)
        self._since_run = 0
        before = (self._final, self._partial)
        self._partial = self._transcribe(self._buf)
        if len(self._buf) >= self._commit:   # lock in this window; keep 0.5 s for context
            self._final = f"{self._final} {self._partial}".strip()
            self._partial = ""
            self._buf = self._buf[-RATE // 2:]
        heard = bool(self._phrase) and contains_phrase(
            _tail(f"{self._final} {self._partial}"), self._phrase)
        return CaptionUpdate(self._final, self._partial, before != (self._final, self._partial), heard)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
_WHISPER_CAPTION_MODELS = {}


def make_listeners(settings: Settings) -> Tuple[Optional[Callable], Optional[Callable], str]:
    """Return (captioner_factory, spotter_factory, description) for this machine.

    Factories build fresh recognizers per recording. The heavy model is loaded
    once and cached.
    """
    engine = settings.caption_engine
    phrase = settings.stop_phrase
    if engine in ("auto", "vosk") and vosk_available(settings.vosk_model_dir):
        model = load_vosk_model(settings.vosk_model_dir)
        return (lambda: VoskCaptioner(model, phrase),
                lambda: VoskPhraseSpotter(model, phrase),
                f"Vosk ({Path(settings.vosk_model_dir).name})")
    if engine == "vosk":
        logger.warning("Vosk captions requested but no model at %s", settings.vosk_model_dir)
    if engine in ("auto", "whisper", "vosk"):
        size = settings.caption_whisper_model

        def factory():
            if size not in _WHISPER_CAPTION_MODELS:
                from faster_whisper import WhisperModel
                _WHISPER_CAPTION_MODELS[size] = WhisperModel(size, device="cpu", compute_type="int8")
            return WhisperCaptioner(_WHISPER_CAPTION_MODELS[size], phrase)
        return factory, None, f"Whisper {size} (fallback, install the Vosk model for faster captions)"
    return None, None, "off"
