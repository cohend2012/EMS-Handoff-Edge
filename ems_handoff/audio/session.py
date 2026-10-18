"""
audio/session.py — one live recording, from START to DONE (or Cancel).

Purpose
    Run the mic and the listeners for a single report and decide when it is
    finished. A recording finishes in exactly one of these ways:
        "button"   the DONE button was pressed (always works)
        "phrase"   the stop phrase was heard (by the spotter or in the captions)
        "timeout"  Settings.max_recording_s was reached. The audio is still
                   processed, never thrown away.
    Cancel is the only path that discards audio, and only on purpose.

Threads
    audio-capture     (inside AudioCapture) mic -> 16 kHz blocks
    live-captions     captioner.feed() -> "caption" events
    phrase-spotter    spotter.feed()   -> DONE when the phrase is heard
    session-watch     level meter + elapsed time + timeout, 5x per second
    The finish callback runs on whichever thread detected the end. It must
    return quickly; the app hands the real work to its report worker.

Events (sent through ``emit``)
    {"type": "caption", "final": str, "partial": str}
    {"type": "level", "rms": float, "seconds": float}
    {"type": "listening", "captions": str, "spotter": bool, "mic": str}
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from .capture import AudioCapture

logger = logging.getLogger(__name__)


class LiveSession:
    def __init__(self, capture: AudioCapture,
                 captioner_factory: Optional[Callable] = None,
                 spotter_factory: Optional[Callable] = None,
                 emit: Callable[[dict], None] = lambda e: None,
                 on_finish: Callable[[str], None] = lambda reason: None,
                 max_seconds: float = 600, captions_label: str = ""):
        self.capture = capture
        self._captioner_factory = captioner_factory
        self._spotter_factory = spotter_factory
        self._emit = emit
        self._on_finish = on_finish
        self.max_seconds = max_seconds
        self.captions_label = captions_label
        self.finish_reason: Optional[str] = None
        self.started_at = 0.0
        self.final_text = ""
        self.problems: list = []
        self._finished = threading.Event()
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        self._threads = []

    # -- control ------------------------------------------------------------
    def _make(self, factory: Optional[Callable], what: str):
        """Build a listener. Listeners are optional: if one cannot load, record without it."""
        if factory is None:
            return None
        try:
            return factory()
        except Exception as e:  # noqa: BLE001 - e.g. model missing and no internet to fetch it
            logger.warning("%s unavailable, recording without it: %s", what, e)
            self.problems.append(f"{what} unavailable ({type(e).__name__}). Use the DONE button.")
            return None

    def start(self) -> None:
        captioner = self._make(self._captioner_factory, "Live captions")
        spotter = self._make(self._spotter_factory, "Stop-phrase spotter")
        cap_q = self.capture.subscribe() if captioner else None
        spot_q = self.capture.subscribe() if spotter else None
        self.capture.start()
        self.started_at = time.time()
        if captioner:
            self._spawn("live-captions", self._caption_loop, captioner, cap_q)
        if spotter:
            self._spawn("phrase-spotter", self._spotter_loop, spotter, spot_q)
        self._spawn("session-watch", self._watch_loop)
        self._emit({"type": "listening",
                    "captions": (self.captions_label or "off") if captioner else "off",
                    "spotter": spotter is not None, "mic": self.capture.device_name,
                    "problems": self.problems})

    def finish(self, reason: str) -> bool:
        """Mark the recording finished. Returns False if it already was."""
        with self._lock:
            if self._finished.is_set():
                return False
            self.finish_reason = reason
            self._finished.set()
        logger.info("Recording finished: %s after %.1fs", reason, time.time() - self.started_at)
        self._on_finish(reason)
        return True

    def stop(self) -> np.ndarray:
        """Close the mic, stop the listeners, and return the full recording."""
        if self._stopped.is_set():
            return self.capture.recording()
        self._stopped.set()
        audio = self.capture.stop()
        for t in self._threads:
            if t is not threading.current_thread():
                t.join(timeout=3)
        return audio

    @property
    def is_finished(self) -> bool:
        return self._finished.is_set()

    # -- threads ------------------------------------------------------------
    def _spawn(self, name, target, *args):
        t = threading.Thread(target=target, args=args, name=name, daemon=True)
        self._threads.append(t)
        t.start()

    def _caption_loop(self, captioner, q: "queue.Queue") -> None:
        while not self._stopped.is_set():
            try:
                block = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if block is None:
                return
            try:
                update = captioner.feed(block)
            except Exception:  # noqa: BLE001 - captions must never kill the recording
                logger.exception("Caption engine error; captions paused")
                return
            if update.changed:
                self.final_text = update.final
                self._emit({"type": "caption", "final": update.final, "partial": update.partial})
            if update.heard_phrase and not self.is_finished:
                self.finish("phrase")

    def _spotter_loop(self, spotter, q: "queue.Queue") -> None:
        while not self._stopped.is_set():
            try:
                block = q.get(timeout=0.25)
            except queue.Empty:
                continue
            if block is None:
                return
            try:
                if spotter.feed(block) and not self.is_finished:
                    self.finish("phrase")
            except Exception:  # noqa: BLE001
                logger.exception("Stop-phrase spotter error; use the DONE button")
                return

    def _watch_loop(self) -> None:
        while not self._stopped.is_set():
            elapsed = time.time() - self.started_at
            self._emit({"type": "level", "rms": round(self.capture.level, 4),
                        "seconds": round(elapsed, 1)})
            if elapsed >= self.max_seconds and not self.is_finished:
                self.finish("timeout")
            time.sleep(0.2)
