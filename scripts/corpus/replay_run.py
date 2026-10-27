#!/usr/bin/env python3
"""Run call-sets through the live server's replay endpoint and save each run's status as a
summary, so score_callsets.py can score runs that were not recorded.

    replay_run.py CS008 CS014 CS001 [--provider groq] [--speed 4] [--out ~/replays] [--repeat 1]

Each run is saved as <out>/<date>_<commit>/<callset>.summary.json (the same shape a recording's
summary has), or <callset>.run<N>.summary.json with --repeat. The server must be running with
HANDOFF_REPLAY=1. Speed above 1 shortens the gaps between bursts; the audio itself is still
played and transcribed in full, so the transcripts and reports are the same as at 1.0x.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests


def run_one(base: str, callset: str, provider: str, speed: float, timeout_s: float) -> dict:
    r = requests.post(f"{base}/api/replay/start", json={"callset": callset, "provider": provider, "speed": speed, "record": False}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if not body.get("ok"):
        raise RuntimeError(body)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(2)
        st = requests.get(f"{base}/api/replay/status", timeout=30).json()
        rep = st.get("replay") or {}
        phase = (rep.get("driver") or {}).get("phase")
        if st.get("state") != "replaying" and phase in ("done", "error"):
            return rep
        if st.get("state") in ("done", "error", "idle") and phase in ("done", "error"):
            return rep
    raise TimeoutError(f"{callset} did not finish in {timeout_s:.0f} s")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("callsets", nargs="+")
    ap.add_argument("--base", default="http://127.0.0.1:5050")
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path.home() / "replays")
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args(argv)
    for cid in args.callsets:
        for n in range(1, args.repeat + 1):
            t0 = time.time()
            rep = run_one(args.base, cid, args.provider, args.speed, args.timeout)
            folder = args.out / f"{time.strftime('%Y-%m-%d')}_{rep.get('commit', 'unknown')}"
            folder.mkdir(parents=True, exist_ok=True)
            name = f"{cid}.summary.json" if args.repeat == 1 else f"{cid}.run{n}.summary.json"
            (folder / name).write_text(json.dumps(rep, indent=1))
            report = (rep.get("session") or {}).get("report") or {}
            print(f"{cid} run {n}: {rep.get('provider')} {rep.get('model')} commit {rep.get('commit')} -> "
                  f"{report.get('alert_type')} {report.get('alert_level')} ETA {report.get('eta_minutes')} "
                  f"({time.time() - t0:.0f} s) -> {folder / name}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
