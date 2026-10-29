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
from typing import List, Optional

import numpy as np
import scipy.io.wavfile as wavfile

from .config import Settings
from .lexicon_rag import whisper_hint

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
GROQ_WHISPER_PREFERENCES = ["whisper-large-v3-turbo", "whisper-large-v3"]


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
        self.hint = whisper_hint(vocab=getattr(settings, 'radio_vocab', ''))
        if client is None:
            if not settings.groq_api_key:
                raise TranscriptionError("Groq transcription needs GROQ_API_KEY in the .env file.")
            from openai import OpenAI
            client = OpenAI(base_url="https://api.groq.com/openai/v1",
                            api_key=settings.groq_api_key, timeout=60, max_retries=1)
        self._client = client
        self._resolved = False

    def describe(self) -> str:
        return f"Groq {self.model}"

    def _resolve_model(self) -> None:
        """If Groq retired the configured Whisper model, use another Whisper it still offers."""
        if self._resolved:
            return
        self._resolved = True
        try:
            ids = {m.id for m in self._client.models.list().data}
        except Exception:  # noqa: BLE001 - cannot list: keep the configured model
            return
        if self.model in ids:
            return
        for candidate in GROQ_WHISPER_PREFERENCES + sorted(i for i in ids if "whisper" in i):
            if candidate in ids:
                logger.warning("Groq no longer offers %s; using %s", self.model, candidate)
                self.model = candidate
                return

    def warm_up(self) -> None:  # nothing to preload for a cloud model
        return None

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) < SAMPLE_RATE // 2:
            raise TranscriptionError("The recording is too short (under half a second).")
        self._resolve_model()
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
        self.hint = whisper_hint(vocab=getattr(settings, 'radio_vocab', ''))
        self.fallback = bool(getattr(settings, "local_whisper_fallback", False))
        self.redecode = bool(getattr(settings, "local_whisper_redecode", False))
        self.last_notes: List[str] = []
        self._model = None
        self._lock = threading.Lock()

    def describe(self) -> str:
        return (f"local faster-whisper {self.model_size} ({self.settings.local_whisper_device})"
                + (", fallback" if self.fallback else "") + (", gap re-decode" if self.redecode else ""))

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
            # One greedy pass at temperature 0. With fallback on, a window whose decode collapses (average log
            # probability under -1 or compression ratio over 2.4) is decoded again at 0.2, then 0.4, and the
            # no-speech gate is off so that window is not thrown away as silence first: on the generated set
            # (SYNTH_V1_2026-10-27.md) medium lost the middle 28 s of a report that way in 3 of 300 clips.
            decode = dict(temperature=[0.0, 0.2, 0.4], no_speech_threshold=None) if self.fallback else dict(temperature=0.0)
            x = np.asarray(audio, dtype=np.float32)
            segments = list(self._decode(x, decode))
            self.last_notes = []
            if self.redecode:
                segments = self._redecode_gaps(x, segments, decode)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:  # noqa: BLE001
            raise TranscriptionError(f"Local transcription failed: {type(e).__name__}: {e}") from e
        logger.info("Local transcription: %.1fs of audio in %.2fs",
                    len(audio) / SAMPLE_RATE, time.perf_counter() - t0)
        return text

    def _decode(self, x: np.ndarray, decode: dict):
        beam = int(getattr(self.settings, "local_whisper_beam", 1) or 1)
        vad = bool(getattr(self.settings, "local_whisper_vad", True))
        segments, _info = self._get_model().transcribe(
            x, language="en", beam_size=beam, condition_on_previous_text=False, **decode,
            initial_prompt=self.hint or None, hotwords=getattr(self, "hotwords", None) or None, vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": 500} if vad else None)
        return segments

    GAP_SECONDS = 6.0          # a stretch of audio this long with no words from the first pass
    GAP_RMS_RATIO = 0.5        # ...and at least half the level of the stretches that did get words
    MAX_WORDS_PER_SECOND = 5.0 # a span decoded alone that comes back faster than anyone talks is a loop

    def _redecode_gaps(self, x: np.ndarray, segments: list, decode: dict) -> list:
        """Whisper's decoder now and then collapses a whole 30 s window to one word or nothing, and the
        no-speech gate drops it: the transcript jumps from "sugar is" to "approximately 20 minutes" with
        the middle of the report gone (medium on the generated set, 3 clips in 300, see
        SYNTH_V1_2026-10-27.md). The collapse is a property of the window, not the audio: the same span
        decoded on its own comes out whole. So any gap between worded segments that is long and as loud as
        the speech around it is decoded again alone and spliced in. Leading and trailing silence is left."""
        worded = [s for s in segments if s.text.strip() and (s.end - s.start) > 0]
        if len(worded) < 1:
            return segments
        def rms(a: float, b: float) -> float:
            seg = x[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
            return float(np.sqrt(np.mean(seg * seg))) if len(seg) else 0.0
        speech_rms = np.median([rms(s.start, s.end) for s in worded]) if worded else 0.0
        covered = [(s.start, s.end) for s in worded]
        gaps = []
        for (a_end), (b_start) in zip([c[1] for c in covered[:-1]], [c[0] for c in covered[1:]]):
            if b_start - a_end >= self.GAP_SECONDS and rms(a_end, b_start) >= self.GAP_RMS_RATIO * speech_rms:
                gaps.append((a_end, b_start))
        # a collapsed window kept by a disabled no-speech gate shows up as a long segment with a word or two
        for s in worded:
            if (s.end - s.start) >= self.GAP_SECONDS * 2 and len(s.text.split()) <= 2:
                gaps.append((s.start, s.end))
        if not gaps:
            return segments
        from .transcript_checks import collapse_repeats   # here, not at the top: transcript_checks imports this module
        out = list(segments)
        for a, b in gaps:
            span = x[int(a * SAMPLE_RATE):int(b * SAMPLE_RATE)]
            redone = [s for s in self._decode(span, decode) if s.text.strip()]
            words = sum(len(s.text.split()) for s in redone)
            first = sum(len(s.text.split()) for s in segments if a <= s.start and s.end <= b)
            _collapsed, loop_note = collapse_repeats(" ".join(s.text.strip() for s in redone))
            if loop_note or words > self.MAX_WORDS_PER_SECOND * max(b - a, 1.0):
                # The span decoded alone can loop instead of collapsing: large-v3-turbo on the generated set gave
                # "ETA ten minutes" 25 times (SY074) and "the end of the day" 37 times (SY051), 250 words for 28 s
                # (SYSTEM_VS_BASE_2026-10-28.md). A loop is not a recovery: the first pass stays, the attempt is noted.
                self.last_notes.append(f"re-decode of {a:.1f}-{b:.1f} s discarded ({loop_note or f'{words} words in {b - a:.0f} s'}); "
                                       f"the first pass keeps its {first} words")
                logger.info("Local transcription: %s", self.last_notes[-1])
                continue
            self.last_notes.append(f"re-decoded {a:.1f}-{b:.1f} s on its own: {words} words where the first pass had {first}")
            logger.info("Local transcription: %s", self.last_notes[-1])
            out = [s for s in out if not (a <= s.start and s.end <= b)]
            for s in redone:
                out.append(_Shifted(s, a))
        return sorted(out, key=lambda s: s.start)


class _Shifted:
    """A re-decoded segment placed back on the clip's timeline."""
    def __init__(self, seg, offset: float):
        self.start, self.end, self.text = seg.start + offset, seg.end + offset, seg.text


def make_transcriber(settings: Settings):
    if settings.effective_transcriber() == "groq":
        return GroqTranscriber(settings)
    return LocalTranscriber(settings)
