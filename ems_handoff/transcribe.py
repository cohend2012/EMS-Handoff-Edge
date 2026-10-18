"""
transcribe.py — the accurate, final transcript, made once after DONE.

Purpose
    Live captions are for the paramedic's eyes. This module produces the text
    the report is actually built from. It runs once on the whole recording.

Backends
    GroqTranscriber   Groq's hosted whisper-large-v3-turbo. Fast (about 1-2 s
                      for a minute of audio) and accurate. Needs GROQ_API_KEY
                      and internet.
    LocalTranscriber  faster-whisper on the device (default large-v3-turbo,
                      CPU int8). Works offline but is slower on a Jetson CPU.
                      Uses one decoding pass (temperature 0) with a silence
                      filter. Whisper's default 6-step temperature fallback is
                      meant for batch accuracy and can multiply the time.
    Both get the EMS vocabulary hint from lexicon_rag.whisper_hint().

Inputs
    16 kHz mono float32 numpy audio.

Outputs
    ``transcribe(audio) -> str``. Raises ``TranscriptionError`` with a message
    fit to show on screen.

Threads
    Called from the report worker thread. LocalTranscriber caches its model
    (loaded on first use, or ahead of time with ``warm_up()``).
"""

from __future__ import annotations

import io
import logging
import threading
import time
from typing import Optional

import numpy as np
import scipy.io.wavfile as wavfile

from .config import Settings
from .lexicon_rag import whisper_hint

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class TranscriptionError(RuntimeError):
    pass


def to_wav_bytes(audio: np.ndarray, rate: int = SAMPLE_RATE) -> bytes:
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    buf = io.BytesIO()
    wavfile.write(buf, rate, (pcm * 32767).astype(np.int16))
    return buf.getvalue()


class GroqTranscriber:
    name = "groq"

    def __init__(self, settings: Settings, client=None):
        self.settings = settings
        self.model = settings.groq_whisper_model
        self.hint = whisper_hint()
        if client is None:
            if not settings.groq_api_key:
                raise TranscriptionError("Groq transcription needs GROQ_API_KEY in the .env file.")
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1",
                            api_key=settings.groq_api_key, timeout=60, max_retries=1)
        self._client = client

    def describe(self) -> str:
        return f"Groq {self.model}"

    def warm_up(self) -> None:  # nothing to preload for a cloud model
        return None

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < SAMPLE_RATE // 2:
            raise TranscriptionError("The recording is too short (under half a second).")
        t0 = time.perf_counter()
        try:
            result = self._client.audio.transcriptions.create(
                file=("report.wav", to_wav_bytes(audio)), model=self.model,
                prompt=self.hint[-800:], language="en", temperature=0.0,
                response_format="json")
        except Exception as e:  # noqa: BLE001
            raise TranscriptionError(f"Groq transcription failed: {type(e).__name__}: {e}") from e
        text = (getattr(result, "text", "") or "").strip()
        logger.info("Groq transcription: %.1fs of audio in %.2fs",
                    len(audio) / SAMPLE_RATE, time.perf_counter() - t0)
        return text


class LocalTranscriber:
    name = "local"

    def __init__(self, settings: Settings, model_size: Optional[str] = None):
        self.settings = settings
        self.model_size = model_size or settings.local_whisper_model
        self.hint = whisper_hint()
        self._model = None
        self._lock = threading.Lock()

    def describe(self) -> str:
        return f"local faster-whisper {self.model_size} ({self.settings.local_whisper_device})"

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel
                t0 = time.perf_counter()
                self._model = WhisperModel(self.model_size,
                                           device=self.settings.local_whisper_device,
                                           compute_type=self.settings.local_whisper_compute)
                logger.info("Loaded faster-whisper %s in %.1fs", self.model_size,
                            time.perf_counter() - t0)
            return self._model

    def warm_up(self) -> None:
        self._get_model()

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < SAMPLE_RATE // 2:
            raise TranscriptionError("The recording is too short (under half a second).")
        t0 = time.perf_counter()
        try:
            segments, _info = self._get_model().transcribe(
                np.asarray(audio, dtype=np.float32), language="en", beam_size=1,
                temperature=0.0, condition_on_previous_text=False,
                initial_prompt=self.hint, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500})
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            raise TranscriptionError(f"Local transcription failed: {type(e).__name__}: {e}") from e
        logger.info("Local transcription: %.1fs of audio in %.2fs",
                    len(audio) / SAMPLE_RATE, time.perf_counter() - t0)
        return text


def make_transcriber(settings: Settings):
    if settings.effective_transcriber() == "groq":
        return GroqTranscriber(settings)
    return LocalTranscriber(settings)
