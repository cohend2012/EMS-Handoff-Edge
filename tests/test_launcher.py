"""Launcher: own port (not the old app's 5000), next free port, and no second copy."""

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from ems_handoff.__main__ import bind, is_our_app
from ems_handoff.config import Settings
from ems_handoff.server import create_app


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _OldApp(BaseHTTPRequestHandler):
    def do_GET(self):  # the old EMS-Handoff answers with its own HTML page
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"<html>old EMS handoff</html>")

    def log_message(self, *a):
        pass


def test_default_port_is_not_the_old_apps_port():
    assert Settings().port == 5050


def test_busy_port_moves_to_the_next_free_one(tmp_path):
    port = _free_port()
    old = HTTPServer(("127.0.0.1", port), _OldApp)
    threading.Thread(target=old.serve_forever, daemon=True).start()
    try:
        assert not is_our_app("127.0.0.1", port)
        server, used, note = bind("127.0.0.1", port, create_app(Settings(output_dir=tmp_path)))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            assert used != port and f"port {port} is used by another program" in note
            assert is_our_app("127.0.0.1", used)
        finally:
            server.shutdown()
    finally:
        old.shutdown()


def test_nothing_listening_is_not_our_app():
    assert not is_our_app("127.0.0.1", _free_port())
