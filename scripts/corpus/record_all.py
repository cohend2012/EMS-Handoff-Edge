#!/usr/bin/env python3
"""Record every call-set as a video through the live server, one after another, on Groq.

    record_all.py [CS001 CS002 ...] [--provider groq] [--skip-done] [--max-minutes 180]

Without ids: every CS* folder under ~/callsets in order. Each one is started with record=true,
waited for, and gets a compressed share copy (<id>.share.mp4, under 30 MB for the chat). The
batch stops when a run fell back from Groq to the local model (the day's quota is gone: the
video would not be the Groq one), or when a recording fails. --skip-done skips a call-set that
already has a sync pass on the server's current commit. Progress goes to stdout and to
~/recordings/BATCH.log.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


def log(msg: str, path: Path) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with path.open("a") as f:
        f.write(line + "\n")


def wait_done(base: str, timeout_s: float) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(3)
        st = requests.get(f"{base}/api/replay/status", timeout=30).json()
        rep = st.get("replay") or {}
        rec = rep.get("recording") or {}
        if rec.get("state") == "failed" or (rec.get("state") == "done" and "sync" in rec):   # sync lands just after "done"
            return rep
        if (rep.get("driver") or {}).get("phase") == "error" and rec.get("state") not in ("recording",):
            return rep
    raise TimeoutError("run did not finish")


def retry_after_s(reason: str) -> float | None:
    """Seconds to wait from Groq's 'Please try again in 12m13.968s' (day-cap and rate-limit replies), else None."""
    m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", reason)
    if not m or not any(m.groups()):
        return None
    h, mi, s = (float(x) if x else 0.0 for x in m.groups())
    return h * 3600 + mi * 60 + s + 45


def share_copy(src: Path) -> Path:
    dst = src.with_suffix(".share.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                    "-maxrate", "800k", "-bufsize", "1600k", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "96k", str(dst)], check=True)
    return dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("callsets", nargs="*")
    ap.add_argument("--base", default="http://127.0.0.1:5050")
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--recordings", type=Path, default=Path.home() / "recordings")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--max-minutes", type=float, default=240)
    ap.add_argument("--max-retries", type=int, default=6, help="re-records of one call-set after a quota fallback")
    args = ap.parse_args(argv)
    logp = args.recordings / "BATCH.log"
    ids = args.callsets or sorted(p.name for p in args.callsets_dir.iterdir() if p.is_dir() and p.name.startswith("CS"))
    commit = requests.get(f"{args.base}/api/settings", timeout=30).json().get("app_commit") or ""
    started = time.time()
    fallbacks: dict = {}
    log(f"batch of {len(ids)} call-sets on {args.provider}, server commit {commit or '?'}", logp)
    ids = list(ids)
    i = -1
    while i + 1 < len(ids):
        i += 1
        cid = ids[i]
        if time.time() - started > args.max_minutes * 60:
            log(f"stopping: {args.max_minutes:.0f} minutes used", logp)
            return 2
        if args.skip_done and commit:
            done = list(args.recordings.glob(f"*_{commit}/{cid}.sync.json"))
            if done and json.loads(done[0].read_text()).get("pass"):
                log(f"{cid}: already recorded with a sync pass on {commit}, skipped", logp)
                continue
        r = requests.post(f"{args.base}/api/replay/start", json={"callset": cid, "provider": args.provider, "speed": 1.0, "record": True}, timeout=60)
        body = r.json() if r.content else {}
        if not body.get("ok"):
            log(f"{cid}: could not start: {body.get('error') or r.text[:200]}", logp)
            return 1
        t0 = time.time()
        try:
            rep = wait_done(args.base, timeout_s=1800)
        except TimeoutError as e:
            log(f"{cid}: {e}", logp)
            return 1
        rec = rep.get("recording") or {}
        report = (rep.get("session") or {}).get("report") or {}
        sync = rec.get("sync") or {}
        line = (f"{cid}: {rec.get('state')} {rec.get('path')} {(rec.get('bytes') or 0) / 1048576:.1f} MB "
                f"{rec.get('duration_s', 0):.0f} s sync {'pass' if sync.get('pass') else 'FAIL'} "
                f"worst {sync.get('max_abs_offset_ms')} ms -> {report.get('alert_type')} {report.get('alert_level')} ETA {report.get('eta_minutes')} "
                f"({time.time() - t0:.0f} s)")
        log(line, logp)
        if rec.get("state") == "done" and rec.get("path"):
            share = share_copy(Path(rec["path"]))
            log(f"{cid}: share copy {share} {share.stat().st_size / 1048576:.1f} MB", logp)
        if rep.get("fallback"):
            reason = str(rep["fallback"].get("reason") or "")
            wait = retry_after_s(reason)
            tries = fallbacks.get(cid, 0) + 1
            fallbacks[cid] = tries
            if wait is None or tries > args.max_retries:
                log(f"{cid}: fell back from {rep['fallback'].get('from')} to local ({reason[:160]}); stopping the batch, "
                    f"this one must be redone on {args.provider}", logp)
                return 3
            log(f"{cid}: fell back to local ({reason[:120]}); waiting {wait / 60:.0f} min for the quota, then recording it again "
                f"(try {tries} of {args.max_retries})", logp)
            time.sleep(wait)
            ids.insert(ids.index(cid) + 1, cid) if cid not in ids[ids.index(cid) + 1:] else None
            continue
        if rec.get("state") != "done":
            log(f"{cid}: recording {rec.get('state')}: {rec.get('error')}; stopping", logp)
            return 1
    log("batch complete", logp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
