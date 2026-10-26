#!/usr/bin/env python3
"""Record every call-set twice, on Groq and on the local model, through the live server.

    demo_batch.py [CS001 ...] [--providers groq,ollama] [--max-hours 12]

One replay runs at a time. Groq's day cap is a rolling 24 h window, so when a Groq run falls
back to local the driver notes Groq's "try again in N min" and records local call-sets until
that time has passed, then returns to the Groq queue. A Groq run that fell back is not counted
and is queued again. A call-set is skipped when a recording for that provider already exists on
the server's current commit with a sync pass and no fallback.

Each recording lands where the server puts it (~/recordings/<date>_<commit>/<id>.mp4 with
.summary.json and .sync.json) plus a share copy; collect_demo.py gathers them per call-set.
Progress: stdout and ~/recordings/BATCH.log.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

LOG = Path.home() / "recordings" / "BATCH.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def retry_after_s(reason: str) -> Optional[float]:
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


def already_done(recordings: Path, commit: str, cid: str, provider: str) -> Optional[Path]:
    """A passing, non-fallback recording of cid on this provider and commit, if one exists."""
    for summ in recordings.glob(f"*_{commit}/{cid}.summary.json"):
        d = json.loads(summ.read_text())
        if d.get("provider") != provider or d.get("fallback"):
            continue
        sync = summ.with_name(f"{cid}.sync.json")
        if sync.exists() and json.loads(sync.read_text()).get("pass"):
            return summ
    return None


def wait_done(base: str, timeout_s: float = 2400) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(3)
        st = requests.get(f"{base}/api/replay/status", timeout=30).json()
        rep = st.get("replay") or {}
        rec = rep.get("recording") or {}
        if rec.get("state") == "failed" or (rec.get("state") == "done" and "sync" in rec):
            return rep
    raise TimeoutError("run did not finish")


def record(base: str, cid: str, provider: str) -> dict:
    r = requests.post(f"{base}/api/replay/start", json={"callset": cid, "provider": provider, "speed": 1.0, "record": True}, timeout=60)
    body = r.json() if r.content else {}
    if not body.get("ok"):
        raise RuntimeError(body.get("error") or r.text[:200])
    return wait_done(base)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("callsets", nargs="*")
    ap.add_argument("--base", default="http://127.0.0.1:5050")
    ap.add_argument("--providers", default="groq,ollama")
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--recordings", type=Path, default=Path.home() / "recordings")
    ap.add_argument("--max-hours", type=float, default=12)
    ap.add_argument("--max-retries", type=int, default=10)
    args = ap.parse_args(argv)
    ids = args.callsets or sorted(p.name for p in args.callsets_dir.iterdir() if p.is_dir() and p.name.startswith("CS"))
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    commit = ""
    try:
        commit = requests.get(f"{args.base}/api/status", timeout=30).json().get("commit") or ""
    except Exception:  # noqa: BLE001
        pass
    queues: Dict[str, List[str]] = {}
    for prov in providers:
        queues[prov] = [c for c in ids if not already_done(args.recordings, commit, c, prov)]
        skipped = len(ids) - len(queues[prov])
        log(f"{prov}: {len(queues[prov])} to record" + (f", {skipped} already done on {commit}" if skipped else ""))
    retries: Dict[str, int] = {}
    groq_free_at = 0.0
    started = time.time()
    while any(queues.values()):
        if time.time() - started > args.max_hours * 3600:
            log(f"stopping after {args.max_hours} h; left: " + ", ".join(f"{p} {q}" for p, q in queues.items() if q))
            return 2
        prov = None
        if queues.get("groq") and time.time() >= groq_free_at:
            prov = "groq"
        else:
            for p in providers:
                if p != "groq" and queues.get(p):
                    prov = p
                    break
        if prov is None:
            wait = max(30.0, groq_free_at - time.time())
            log(f"only Groq left and its quota is closed; sleeping {wait / 60:.0f} min")
            time.sleep(wait)
            continue
        cid = queues[prov][0]
        t0 = time.time()
        try:
            rep = record(args.base, cid, prov)
        except Exception as e:  # noqa: BLE001
            log(f"{cid} on {prov}: {e}; waiting 2 min and trying again")
            time.sleep(120)
            continue
        rec = rep.get("recording") or {}
        report = (rep.get("session") or {}).get("report") or {}
        sync = rec.get("sync") or {}
        fb = rep.get("fallback")
        log(f"{cid} on {prov}: {rec.get('state')} {(rec.get('bytes') or 0) / 1048576:.1f} MB {rec.get('duration_s', 0):.0f} s "
            f"sync {'pass' if sync.get('pass') else 'FAIL'} worst {sync.get('max_abs_offset_ms')} ms -> "
            f"{report.get('alert_type')} {report.get('alert_level')} ETA {report.get('eta_minutes')} ({time.time() - t0:.0f} s)"
            + (f" FELL BACK to {fb.get('to')} after burst {fb.get('bursts')}" if fb else ""))
        if rec.get("state") == "done" and rec.get("path") and not fb:
            share = share_copy(Path(rec["path"]))
            log(f"{cid} on {prov}: share copy {share.name} {share.stat().st_size / 1048576:.1f} MB")
        if fb:
            wait = retry_after_s(str(fb.get("reason") or "")) or 900.0
            groq_free_at = time.time() + wait
            retries[cid] = retries.get(cid, 0) + 1
            if retries[cid] > args.max_retries:
                log(f"{cid} on {prov}: gave up after {args.max_retries} quota retries")
                queues[prov].pop(0)
            else:
                log(f"{cid} on {prov}: Groq closed for about {wait / 60:.0f} min; recording local call-sets meanwhile")
            continue
        if rec.get("state") != "done":
            log(f"{cid} on {prov}: recording {rec.get('state')}: {rec.get('error')}; waiting 2 min and trying again")
            time.sleep(120)
            continue
        queues[prov].pop(0)
    log("demo batch complete: every call-set recorded on " + " and ".join(providers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
