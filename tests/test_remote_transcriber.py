"""The split pipeline's remote transcriber: posts WAV, reads JSON; the factory picks it; the server decodes WAV bytes."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from ems_handoff.transcribe import SAMPLE_RATE
from ems_handoff.config import Settings
from ems_handoff.transcribe import RemoteTranscriber, TranscriptionError, make_transcriber, to_wav_bytes


class _Fake(BaseHTTPRequestHandler):
    seen = {}

    def do_GET(self):
        body = json.dumps({"ok": True, "model": "large-v3", "device": "cuda", "compute": "float16"}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers["Content-Length"]); data = self.rfile.read(n)
        _Fake.seen["bytes"] = len(data); _Fake.seen["path"] = self.path; _Fake.seen["riff"] = data[:4]
        body = json.dumps({"text": "Medic 14, 67 male, chest pain.", "seconds": 0.42, "notes": ["re-decode: none"]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_remote_transcriber_posts_wav_and_returns_the_text(server):
    s = Settings(transcriber="remote", remote_whisper_url=server)
    t = make_transcriber(s)
    assert isinstance(t, RemoteTranscriber) and s.effective_transcriber() == "remote"
    t.warm_up()
    assert "large-v3" in t.describe() and "cuda" in t.describe()
    audio = np.zeros(SAMPLE_RATE * 2, dtype=np.float32)
    assert t.transcribe(audio) == "Medic 14, 67 male, chest pain."
    assert _Fake.seen["path"] == "/transcribe" and _Fake.seen["riff"] == b"RIFF" and _Fake.seen["bytes"] == len(to_wav_bytes(audio))
    assert t.last_notes == ["re-decode: none"]


def test_remote_transcriber_needs_a_url_and_reports_a_dead_server():
    with pytest.raises(TranscriptionError):
        RemoteTranscriber(Settings(transcriber="remote", remote_whisper_url=""))
    t = RemoteTranscriber(Settings(transcriber="remote", remote_whisper_url="http://127.0.0.1:9"), timeout_s=2)
    with pytest.raises(TranscriptionError):
        t.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))


def test_server_decodes_the_wav_it_is_sent():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("whisper_server", pathlib.Path("scripts/whisper_server.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    x = (np.sin(np.linspace(0, 200, SAMPLE_RATE)) * 0.5).astype(np.float32)
    y = mod.wav_to_float(to_wav_bytes(x))
    assert y.shape == x.shape and abs(float(np.abs(y - x).max())) < 1e-3
