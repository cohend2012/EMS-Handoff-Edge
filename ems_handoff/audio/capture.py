"""
audio/capture.py — one microphone stream, shared by every listener.

Purpose
    Open the mic once, convert everything to 16 kHz mono float32, keep the
    complete recording, and hand each block to any number of subscribers
    (the caption engine and the stop-phrase spotter each get their own queue).

Why this design
    The PortAudio callback only copies the block into a queue. Resampling,
    level metering, and fan-out happen on a separate worker thread, so a slow
    listener can never cause dropped audio. Subscriber queues are unbounded,
    so a slow listener falls behind without blocking the others.

Device selection (Settings.mic_device / HANDOFF_MIC_DEVICE)
    ""        auto: the first input whose name contains "USB", else the system default
    "24"      that device index
    "PnP"     the first input device whose name contains that text (case-insensitive)
    Device numbers can change after a reboot. Names do not, so prefer a name.

Sample rate
    The device's native rate is opened (raw ALSA devices, hw:X,Y, usually only
    offer 44.1/48 kHz) and resampled to 16 kHz. 16 kHz is tried if that fails.

Mains hum
    A high-pass (Settings.mic_highpass_hz / HANDOFF_MIC_HIGHPASS_HZ, default 150 Hz, 0 = off) runs on
    the worker thread after resampling, so the level meter, the recording and every listener get
    audio without the 50/60 Hz hum the USB mic on the Jetson picks up.
Threads
    PortAudio callback thread -> queue -> "audio-capture" worker thread -> subscribers.
"""

from __future__ import annotations

import logging
import queue
import threading
from math import gcd
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

TARGET_RATE = 16000


class MicError(RuntimeError):
    pass


def _sd():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio library missing
        raise MicError("Audio system unavailable (PortAudio not installed): "
                       f"{e}. Run: sudo apt install libportaudio2") from e
    return sd


def list_input_devices(devices: Optional[Sequence[dict]] = None) -> List[Tuple[int, str, float]]:
    """[(index, name, default_samplerate)] for every device with an input channel."""
    if devices is None:
        devices = _sd().query_devices()
    return [(i, d["name"], float(d.get("default_samplerate") or TARGET_RATE))
            for i, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]


def find_input_device(spec: str, devices: Optional[Sequence[dict]] = None,
                      default_index: Optional[int] = None) -> Tuple[int, str, float]:
    inputs = list_input_devices(devices)
    if not inputs:
        raise MicError("No microphone found. Plug in the USB mic and restart the app.")
    spec = (spec or "").strip()
    if spec.isdigit():
        for dev in inputs:
            if dev[0] == int(spec):
                return dev
        raise MicError(f"Mic device {spec} does not exist or has no input. "
                       f"Inputs: {', '.join(f'{i}: {n}' for i, n, _ in inputs)}")
    if spec:
        for dev in inputs:
            if spec.lower() in dev[1].lower():
                return dev
        raise MicError(f"No input device name contains '{spec}'.")
    for dev in inputs:
        if "usb" in dev[1].lower():
            return dev
    if default_index is not None:
        for dev in inputs:
            if dev[0] == default_index:
                return dev
    return inputs[0]


def resample(block: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    if src_rate == dst_rate:
        return block
    from scipy.signal import resample_poly
    g = gcd(int(src_rate), int(dst_rate))
    return resample_poly(block, dst_rate // g, src_rate // g, padtype="line").astype(np.float32)


StreamFactory = Callable[[int, int, int, Callable], object]


def _default_stream_factory(device: int, samplerate: int, blocksize: int, callback):
    sd = _sd()
    return sd.InputStream(device=device, channels=1, dtype="float32",
                          samplerate=samplerate, blocksize=blocksize, callback=callback)


class AudioCapture:
    """Mic -> 16 kHz float32 blocks -> subscribers, plus the full recording."""

    def __init__(self, device_spec: str = "", block_ms: int = 100,
                 stream_factory: Optional[StreamFactory] = None,
                 device_info: Optional[Tuple[int, str, float]] = None,
                 highpass_hz: float = 150.0):
        self.device_spec = device_spec
        self.block_ms = block_ms
        # Mains hum: the USB mic on the Jetson delivers 60 Hz at -20 dBFS with speech 10 dB under it
        # (2026-10-26). A 6th-order high-pass (about 47 dB down at 60 Hz for the 150 Hz default),
        # run block by block with its state carried over, takes the hum out before the level meter,
        # the recording and every listener see the audio. Speech intelligibility sits above 300 Hz.
        self.highpass_hz = float(highpass_hz or 0.0)
        self._hp_sos = None
        self._hp_state = None
        if self.highpass_hz > 0:
            from scipy.signal import butter, sosfilt_zi
            self._hp_sos = butter(6, self.highpass_hz, btype="highpass", fs=TARGET_RATE, output="sos")
            self._hp_state = sosfilt_zi(self._hp_sos) * 0.0
        self._stream_factory = stream_factory or _default_stream_factory
        self._device_info = device_info
        self._raw: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._subscribers: List[queue.Queue] = []
        self._chunks: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._stream = None
        self._worker: Optional[threading.Thread] = None
        self.native_rate = TARGET_RATE
        self.device_name = ""
        self.level = 0.0          # RMS of the latest block, 0..1
        self.overflows = 0

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> None:
        if self._device_info is None:
            sd = _sd()
            try:
                default_in = sd.default.device[0]
            except Exception:  # noqa: BLE001 - no default device configured
                default_in = None
            self._device_info = find_input_device(self.device_spec, None, default_in)
        index, name, native = self._device_info
        self.device_name = name
        last_error = None
        # The device's own rate first: asking a raw ALSA device for 16 kHz fails and makes
        # PortAudio print errors. Resampling 100 ms blocks costs nothing noticeable.
        rates = [int(native), TARGET_RATE, 48000, 44100]
        for rate in dict.fromkeys(rates):
            try:
                self._stream = self._stream_factory(index, rate, int(rate * self.block_ms / 1000),
                                                    self._callback)
                self._stream.start()
                self.native_rate = rate
                break
            except Exception as e:  # noqa: BLE001 - try the next rate
                last_error = e
                self._stream = None
        if self._stream is None:
            raise MicError(f"Could not open mic '{name}': {last_error}")
        logger.info("Mic open: [%s] %s at %d Hz (resampled to %d Hz)", index, name,
                    self.native_rate, TARGET_RATE)
        self._worker = threading.Thread(target=self._run, name="audio-capture", daemon=True)
        self._worker.start()

    def stop(self) -> np.ndarray:
        """Close the mic and return the whole recording (16 kHz float32)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception as e:  # noqa: BLE001
                logger.warning("Closing mic stream: %s", e)
            self._stream = None
        self._raw.put(None)
        if self._worker is not None:
            self._worker.join(timeout=5)
            self._worker = None
        for q in self._subscribers:
            q.put(None)  # tell listeners the stream ended
        return self.recording()

    def recording(self) -> np.ndarray:
        with self._lock:
            return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    @property
    def seconds(self) -> float:
        with self._lock:
            return sum(len(c) for c in self._chunks) / TARGET_RATE

    def subscribe(self) -> "queue.Queue[Optional[np.ndarray]]":
        q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue()
        self._subscribers.append(q)
        return q

    # -- internals ------------------------------------------------------------
    def _callback(self, indata, frames, time_info, status):  # PortAudio thread: keep it tiny
        if status:
            self.overflows += 1
        self._raw.put(indata[:, 0].copy() if indata.ndim == 2 else indata.copy())

    def _run(self) -> None:
        while True:
            block = self._raw.get()
            if block is None:
                return
            block = resample(np.asarray(block, dtype=np.float32), self.native_rate)
            if self._hp_sos is not None and block.size:
                from scipy.signal import sosfilt
                block, self._hp_state = sosfilt(self._hp_sos, block, zi=self._hp_state)
                block = block.astype(np.float32)
            self.level = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
            with self._lock:
                self._chunks.append(block)
            for q in self._subscribers:
                q.put(block)

    # used by tests and by the fake mic
    def feed(self, block: np.ndarray) -> None:
        self._raw.put(np.asarray(block, dtype=np.float32))
