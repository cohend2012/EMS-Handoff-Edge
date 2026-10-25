"""
audio/sources.py — audio sources other than the microphone.

Purpose
    Play a recorded clip into the live pipeline exactly as the mic would deliver
    it: 16 kHz mono float32 blocks of 100 ms, released at real-time pace, fanned
    out to the same subscriber queues, with the full recording kept. When the
    file ends the source tells its owner, so a session finishes the way
    pressing DONE does.

Why this design
    The mic path (capture.py) is not touched. A session takes any object with
    the AudioSource shape; FileAudioSource is the first one besides AudioCapture.
    Replay (recorded radio calls) uses it. The live mic never goes through here.

Threads
    "audio-file" worker: reads the file once at start, then releases one block
    per block_ms on a monotonic clock, calls the on_end callback after the last
    block, and exits. stop() joins it and puts None on every subscriber queue.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable, List, Optional, Protocol

import numpy as np

from .capture import TARGET_RATE, resample

logger = logging.getLogger(__name__)


class AudioSource(Protocol):
    """What LiveSession needs from an audio source. AudioCapture satisfies it as-is."""

    device_name: str
    level: float

    def start(self) -> None: ...
    def stop(self) -> np.ndarray: ...
    def subscribe(self) -> "queue.Queue[Optional[np.ndarray]]": ...
    def recording(self) -> np.ndarray: ...


class AudioFileError(RuntimeError):
    pass


def load_wav(path: Path) -> np.ndarray:
    """A WAV file as 16 kHz mono float32 in -1..1. Mono only; other rates are resampled."""
    try:
        with wave.open(str(path), "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except (OSError, wave.Error, EOFError) as e:
        raise AudioFileError(f"Cannot read {path}: {e}") from e
    if channels != 1:
        raise AudioFileError(f"{path.name} has {channels} channels; replay clips must be mono")
    if width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        audio = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif width == 1:
        audio = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise AudioFileError(f"{path.name}: unsupported sample width {width} bytes")
    if rate != TARGET_RATE:
        audio = resample(audio, rate)
    return np.ascontiguousarray(audio, dtype=np.float32)


class FileAudioSource:
    """A WAV clip -> 16 kHz float32 blocks at real-time pace -> subscribers, plus the recording."""

    def __init__(self, path, block_ms: int = 100, pace: float = 1.0,
                 on_end: Optional[Callable[[], None]] = None):
        self.path = Path(path)
        self.block_ms = block_ms
        self.pace = pace                      # 1.0 = real time; 0 = as fast as possible (tests)
        self._on_end = on_end
        self._subscribers: List[queue.Queue] = []
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._audio: Optional[np.ndarray] = None
        self.device_name = f"file: {self.path.name}"
        self.level = 0.0
        self.overflows = 0
        self.ended = False                    # the last block was delivered
        self.blocks_sent = 0

    # -- the hook a session uses to finish when the clip ends ----------------
    def set_on_end(self, callback: Callable[[], None]) -> None:
        self._on_end = callback

    # -- lifecycle (same surface as AudioCapture) -----------------------------
    def start(self) -> None:
        self._audio = load_wav(self.path)       # errors surface here, before any thread runs
        self._worker = threading.Thread(target=self._run, name="audio-file", daemon=True)
        self._worker.start()
        logger.info("Playing %s: %.1f s", self.path.name, len(self._audio) / TARGET_RATE)

    def stop(self) -> np.ndarray:
        """Stop playback (if still running), tell the listeners, return what was delivered."""
        if not self._stop.is_set():
            self._stop.set()
            if self._worker is not None and self._worker is not threading.current_thread():
                self._worker.join(timeout=5)
            for q in self._subscribers:
                q.put(None)
        return self.recording()

    def recording(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(c) for c in self._chunks) / TARGET_RATE

    @property
    def duration(self) -> float:
        """Length of the whole clip in seconds (0 before start)."""
        return 0.0 if self._audio is None else len(self._audio) / TARGET_RATE

    def subscribe(self) -> "queue.Queue[Optional[np.ndarray]]":
        q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._subscribers.append(q)
        return q

    # -- internals --------------------------------------------------------------
    def _run(self) -> None:
        audio = self._audio
        block_len = int(TARGET_RATE * self.block_ms / 1000)
        period = self.block_ms / 1000 * self.pace
        t0 = time.monotonic()
        i = 0
        for start in range(0, len(audio), block_len):
            if self._stop.is_set():
                return
            due = t0 + i * period
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            block = audio[start:start + block_len]
            self.level = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
            with self._lock:
                self._chunks.append(block)
            for q in self._subscribers:
                q.put(block)
            i += 1
            self.blocks_sent = i
        self.ended = True
        if self._on_end is not None and not self._stop.is_set():
            try:
                self._on_end()
            except Exception:  # noqa: BLE001 - the owner's problem must not kill the source thread
                logger.exception("on_end callback failed")
