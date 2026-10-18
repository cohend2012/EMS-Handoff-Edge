"""
__main__.py — start HANDOFF.AI as a desktop app.

    python -m ems_handoff              start the server and open the app window
    python -m ems_handoff --no-window  start the server only (open http://127.0.0.1:5000 yourself)
    python -m ems_handoff --check      print the setup check and exit (mic, models, AI provider)

What it does
    1. Loads .env from the project root, then the choices saved from the app.
    2. Sets up logging to output/handoff.log and the terminal.
    3. Starts the local server on 127.0.0.1:5000. It is never exposed to the network.
    4. Opens Chromium/Chrome in "app" mode (its own window, no address bar),
       or the default browser if Chromium is not found.
    5. Runs until Quit is pressed in the app, or Ctrl+C.

Threads
    The HTTP server runs in a background thread. The main thread waits for quit.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ems_handoff", description="HANDOFF.AI desktop app")
    parser.add_argument("--no-window", action="store_true", help="do not open the app window")
    parser.add_argument("--check", action="store_true", help="check the setup and exit")
    parser.add_argument("--port", type=int, help="port (default 5000)")
    args = parser.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

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

    from werkzeug.serving import make_server

    stop = threading.Event()
    app = create_app(settings, on_quit=stop.set)
    threading.Thread(target=app.config["controller"].warm_up, name="warm-up", daemon=True).start()
    try:
        server = make_server(settings.host, settings.port, app, threaded=True)
    except OSError as e:
        log.error("Cannot start on %s:%d (%s). Is HANDOFF.AI already running?",
                  settings.host, settings.port, e)
        return 1
    threading.Thread(target=server.serve_forever, name="http", daemon=True).start()
    url = f"http://{settings.host}:{settings.port}/"
    log.info("HANDOFF.AI %s running at %s (provider %s)", __version__, url, settings.provider)
    if not args.no_window and os.environ.get("DISPLAY", "") + os.environ.get("WAYLAND_DISPLAY", ""):
        open_window(url)
    elif not args.no_window:
        log.info("No display found; open %s in a browser.", url)
    try:
        stop.wait()
    except KeyboardInterrupt:
        pass
    log.info("Shutting down")
    server.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
