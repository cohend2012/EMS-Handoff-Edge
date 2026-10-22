"""
__main__.py — start HANDOFF.AI as a desktop app.

    python -m ems_handoff              start the server and open the app window
    python -m ems_handoff --no-window  start the server only (open http://127.0.0.1:5050 yourself)
    python -m ems_handoff --check      print the setup check and exit (mic, models, AI provider)

What it does
    1. Loads .env from the project root, then the choices saved from the app.
    2. Sets up logging to output/handoff.log and the terminal.
    3. Starts the local server on 127.0.0.1:5050. It is never exposed to the network.
       Port 5000 is left to the old EMS-Handoff app. If 5050 is taken by some
       other program, the next free port is used and printed. If HANDOFF.AI is
       already running there, its window is opened instead of a second copy.
    4. Opens Chromium/Chrome in "app" mode (its own window, no address bar),
       or the default browser if Chromium is not found.
    5. Runs until Quit is pressed in the app, or Ctrl+C.

Threads
    The HTTP server runs in a background thread. The main thread waits for quit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import threading
import urllib.request
import webbrowser

from . import __version__
from .config import PROJECT_ROOT, load_settings


def setup_logging(output_dir) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(output_dir / "handoff.log", mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)   # no per-request noise
    logging.getLogger("httpx").setLevel(logging.WARNING)


def open_window(url: str) -> None:
    for name in ("chromium-browser", "chromium", "google-chrome", "google-chrome-stable"):
        exe = shutil.which(name)
        if exe:
            subprocess.Popen([exe, f"--app={url}", "--new-window", "--start-maximized"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
    webbrowser.open(url)


def is_our_app(host: str, port: int) -> bool:
    """True if HANDOFF.AI (this app, not the old EMS-Handoff) answers on that port."""
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/api/status", timeout=2) as r:
            return json.load(r).get("app") == "handoff-app"
    except Exception:  # noqa: BLE001 - anything else on the port is "not us"
        return False


def bind(host: str, port: int, app, tries: int = 20):
    """Start the HTTP server on ``port`` or the next free one. Returns (server, port, note)."""
    from werkzeug.serving import make_server

    last = None
    for p in range(port, port + tries):
        try:
            with socket.socket() as probe:   # werkzeug exits the process on a busy port; check first
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind((host, p))
            server = make_server(host, p, app, threaded=True)
        except (OSError, SystemExit) as e:
            last = e
            continue
        note = "" if p == port else f"port {port} is used by another program, so port {p} is used instead"
        return server, p, note
    raise OSError(f"no free port between {port} and {port + tries - 1}: {last}")


def banner(url: str, note: str = "") -> None:
    line = "=" * 64
    print(f"\n{line}\n  HANDOFF.AI (new app) is running.\n  Open this address:  {url}")
    if note:
        print(f"  Note: {note}.")
    print(f"  http://127.0.0.1:5000 is the OLD EMS-Handoff app, not this one.\n{line}\n", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ems_handoff", description="HANDOFF.AI desktop app")
    parser.add_argument("--no-window", action="store_true", help="do not open the app window")
    parser.add_argument("--check", action="store_true", help="check the setup and exit")
    parser.add_argument("--port", type=int, help="port (default 5050)")
    args = parser.parse_args(argv)

    from .config import load_env_files
    load_env_files()

    from .server import create_app, load_user_settings
    settings = load_user_settings(load_settings())
    if args.port:
        import dataclasses
        settings = dataclasses.replace(settings, port=args.port)
    setup_logging(settings.output_dir)
    log = logging.getLogger("ems_handoff")

    if args.check:
        from .doctor import run_checks
        return run_checks(settings)

    # Never call Hugging Face at startup when the models are already on disk (works with no signal).
    from .audio.listeners import vosk_available
    from .doctor import whisper_cached
    needed = {settings.local_whisper_model}
    if not vosk_available(settings.vosk_model_dir):
        needed.add(settings.caption_whisper_model)
    if all(whisper_cached(m) for m in needed):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if is_our_app(settings.host, settings.port):
        url = f"http://{settings.host}:{settings.port}/"
        banner(url, "it was already running, so its window was opened again")
        if not args.no_window:
            open_window(url)
        return 0

    stop = threading.Event()
    app = create_app(settings, on_quit=stop.set)
    try:
        server, port, note = bind(settings.host, settings.port, app)
    except OSError as e:
        log.error("Cannot start the server: %s", e)
        return 1
    if note:
        log.warning("%s", note)
    threading.Thread(target=app.config["controller"].warm_up, name="warm-up", daemon=True).start()
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    url = f"http://{settings.host}:{port}/"
    log.info("HANDOFF.AI %s running at %s (provider %s)", __version__, url, settings.provider)
    banner(url, note)
    if not args.no_window and os.environ.get("DISPLAY", "") + os.environ.get("WAYLAND_DISPLAY", ""):
        open_window(url)
    elif not args.no_window:
        log.info("No display found; open %s in a browser.", url)
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    log.info("Shutting down")
    # shutdown() waits for the serve loop; a stuck request must not make Ctrl+C hang
    closer = threading.Thread(target=server.shutdown, daemon=True)
    closer.start()
    closer.join(timeout=3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
