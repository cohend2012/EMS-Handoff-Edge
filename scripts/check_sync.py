#!/usr/bin/env python3
"""Check a replay recording's audio/video sync: clicks in the audio vs flashes in the video, per burst.

    python scripts/check_sync.py ~/recordings/<date>_<commit>/CS008.mp4 [--expected 5] [--json out.json]

Pass rule: every burst within 150 ms (ems_handoff/sync_check.py)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ems_handoff.sync_check import check, format_report  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("mp4")
ap.add_argument("--expected", type=int)
ap.add_argument("--json")
ap.add_argument("--pass-ms", type=float, default=150.0)
a = ap.parse_args()
r = check(Path(a.mp4), a.expected, a.pass_ms)
print(format_report(r))
if a.json:
    Path(a.json).write_text(json.dumps(r, indent=1))
sys.exit(0 if r["pass"] else 1)
