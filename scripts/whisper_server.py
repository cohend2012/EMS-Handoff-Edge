"""
whisper_server.py — the app's local whisper decode behind a one-endpoint HTTP server, for the split pipeline.

Runs on the GPU Jetson (inside the CUDA container) and serves the other Jetson's app, which uses
HANDOFF_TRANSCRIBER=remote and HANDOFF_WHISPER_URL=http://<this box>:8765.

    LOCAL_WHISPER_DEVICE=cuda LOCAL_WHISPER_COMPUTE=float16 LOCAL_WHISPER_MODEL=large-v3 \\
        python3 scripts/whisper_server.py --port 8765

    POST /transcribe   body: WAV (16 kHz mono)  ->  {"text", "seconds", "notes", "model"}
    GET  /health                                ->  {"ok", "model", "device", "compute"}

Standard library only (the container has no Flask). One request at a time: the model is one decoder.
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ems_handoff.transcribe import SAMPLE_RATE  # noqa: E402
from ems_handoff.config import load_env_files, load_settings  # noqa: E402
from ems_handoff.transcribe import LocalTranscriber, TranscriptionError  # noqa: E402,F811

_lock = threading.Lock()


def wav_to_float(data: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(data)) as w:
        rate, ch, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"16-bit PCM expected, got {8 * width}-bit")
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if rate != SAMPLE_RATE:
        from ems_handoff.audio.capture import resample
        x = resample(x, rate, SAMPLE_RATE)
    return x


def make_handler(transcriber: LocalTranscriber, settings):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/health"):
                self._send(200, {"ok": True, "model": transcriber.model_size, "device": settings.local_whisper_device,
                                 "compute": settings.local_whisper_compute, "redecode": transcriber.redecode})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self.path.startswith("/transcribe"):
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            data = self.rfile.read(n)
            try:
                audio = wav_to_float(data)
            except Exception as e:  # noqa: BLE001
                return self._send(400, {"error": f"bad wav: {e}"})
            # Per-request decode options on the one loaded model (no second model fits beside large-v3 on this box):
            # ?hint=0 drops the vocabulary prompt, ?beam=N sets the beam, ?vad=0 turns Silero off, ?pad=S adds S seconds
            # of silence before and after the burst. A second opinion is a second decode of the same weights.
            q = parse_qs(urlparse(self.path).query)
            opt = lambda k: (q.get(k) or [None])[0]
            if opt("pad"):
                pad = np.zeros(int(float(opt("pad")) * SAMPLE_RATE), dtype=np.float32)
                audio = np.concatenate([pad, audio, pad])
            t0 = time.perf_counter()
            try:
                with _lock:
                    saved = (transcriber.hint, transcriber.settings)
                    try:
                        if opt("hint") == "0":
                            transcriber.hint = ""
                        changes = {}
                        if opt("beam"):
                            changes["local_whisper_beam"] = int(opt("beam"))
                        if opt("vad"):
                            changes["local_whisper_vad"] = opt("vad") != "0"
                        if changes:
                            transcriber.settings = dataclasses.replace(transcriber.settings, **changes)
                        text = transcriber.transcribe(audio)
                    finally:
                        transcriber.hint, transcriber.settings = saved
            except TranscriptionError as e:
                return self._send(500, {"error": str(e)})
            self._send(200, {"text": text, "seconds": round(time.perf_counter() - t0, 2),
                             "audio_s": round(len(audio) / SAMPLE_RATE, 1), "notes": list(getattr(transcriber, "last_notes", [])), "words": list(getattr(transcriber, "last_words", [])),
                             "model": transcriber.model_size})

        def log_message(self, fmt, *args):  # quieter: one line per request
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))
    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--model", help="faster-whisper size; default LOCAL_WHISPER_MODEL")
    args = ap.parse_args(argv)
    load_env_files()
    settings = load_settings()
    transcriber = LocalTranscriber(settings, model_size=args.model)
    t0 = time.perf_counter()
    transcriber.warm_up()
    print(f"whisper server: {transcriber.describe()} loaded in {time.perf_counter() - t0:.1f} s, listening on {args.host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), make_handler(transcriber, settings)).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
