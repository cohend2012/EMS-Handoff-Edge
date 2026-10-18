#!/usr/bin/env python3
"""
mic_test.py — test the microphone, live captions and stop phrase from a terminal.

    python scripts/mic_test.py              record up to 30 s, show captions live
    python scripts/mic_test.py --seconds 15
    python scripts/mic_test.py --no-final   skip the final transcription step

What you should see
    * the chosen mic and a live level bar
    * your words appearing within about a second (live captions)
    * "STOP PHRASE HEARD" when you say "confirm report ready" (recording ends)
    * the final transcript and how long it took (Groq or local Whisper)

This uses exactly the same engine code as the app, so if it works here, it works there.
Press Ctrl+C to stop early. The recording is still transcribed.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description="Microphone and live-caption test")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--no-final", action="store_true")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    from ems_handoff.audio.capture import AudioCapture, MicError
    from ems_handoff.audio.listeners import make_listeners
    from ems_handoff.audio.session import LiveSession
    from ems_handoff.config import load_settings
    from ems_handoff.transcribe import TranscriptionError, make_transcriber

    s = load_settings()
    print("Loading live caption engine...")
    cap_f, spot_f, label = make_listeners(s)
    print(f"Captions: {label}. Stop phrase: '{s.stop_phrase}' (spotter: {'on' if spot_f else 'captions only'})")

    done = threading.Event()
    state = {"caption": "", "level": 0.0, "secs": 0.0, "reason": ""}

    def emit(ev):
        if ev["type"] == "caption":
            state["caption"] = (ev["final"] + " " + ev["partial"]).strip()
        elif ev["type"] == "level":
            state["level"], state["secs"] = ev["rms"], ev["seconds"]
        elif ev["type"] == "listening":
            print(f"Mic: {ev['mic']}\nTalk now. Say '{s.stop_phrase}' to finish.\n")

    def on_finish(reason):
        state["reason"] = reason
        done.set()

    session = LiveSession(AudioCapture(s.mic_device), cap_f, spot_f, emit=emit,
                          on_finish=on_finish, max_seconds=args.seconds, captions_label=label)
    try:
        session.start()
    except MicError as e:
        print(f"MIC PROBLEM: {e}")
        return 1
    try:
        while not done.wait(0.2):
            bar = "#" * min(30, int(state["level"] * 300))
            text = state["caption"][-90:]
            print(f"\r{state['secs']:5.1f}s |{bar:<30}| {text}", end="", flush=True)
    except KeyboardInterrupt:
        session.finish("ctrl-c")
    audio = session.stop()
    print(f"\n\nFinished by: {state['reason'].upper()}" +
          ("  <- STOP PHRASE HEARD" if state["reason"] == "phrase" else ""))
    print(f"Recorded {len(audio) / 16000:.1f} s")
    if args.no_final:
        return 0
    try:
        t = make_transcriber(s)
        print(f"\nFinal transcript with {t.describe()}...")
        t0 = time.perf_counter()
        text = t.transcribe(audio)
        print(f"({time.perf_counter() - t0:.1f} s)\n{text}")
    except TranscriptionError as e:
        print(f"Final transcription failed: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
