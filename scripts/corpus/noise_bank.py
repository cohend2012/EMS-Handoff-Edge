#!/usr/bin/env python3
"""
noise_bank.py — cut the corpus' between-words floor into a noise bed per feed, for radioize.py --noise-file.

    python scripts/corpus/noise_bank.py                 # $HANDOFF_DATASETS/noise/<feed>_floor.wav (skips beds that exist)
    python scripts/corpus/noise_bank.py --force --out /tmp/beds --min-run 4

What a "floor" frame is, from the spectral survey's per-frame CSVs ($HANDOFF_DATASETS/spectral/frames/, no re-decode
of the audio for the selection): the carrier is open (level above --closed dBFS; squelch-closed gaps are digital
silence below -90), it is well under the speech (at least --under dB below the file's median voiced level), it is
not voiced (voicing < 0.3) and the frame is not labelled speech, tone or hum. Runs shorter than --min-run frames
(16 ms each) are dropped. The runs are cut from the decoded call audio, 10 ms raised-cosine crossfades between
them, and concatenated per feed. Writes <feed>_floor.wav (16 kHz mono) and INDEX.tsv with the seconds each
source file contributed. The beds that were in noise/ before this script (2026-10-26) were cut by hand.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from spectral_survey import HOP, SR, decode  # noqa: E402

FRAME_S = HOP / SR


def floor_runs(rows: List[dict], closed_db: float, under_db: float, min_run: int) -> List[Tuple[int, int]]:
    """(first frame, frame count) of every run of floor frames in one file's survey rows."""
    lvl = np.array([float(r["level_db"]) for r in rows])
    voic = np.array([float(r["voicing"]) for r in rows])
    lab = np.array([r["label"] for r in rows])
    voiced = voic > 0.30
    speech_ref = float(np.percentile(lvl[voiced], 50)) if voiced.any() else float(np.percentile(lvl, 90))
    ok = (lvl > closed_db) & (lvl < speech_ref - under_db) & (voic < 0.30) & ~np.isin(lab, ["speech", "tone", "hum"])
    runs, start = [], None
    for i, flag in enumerate(list(ok) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            if i - start >= min_run:
                runs.append((start, i - start))
            start = None
    return runs


def cut(x: np.ndarray, runs: List[Tuple[int, int]], fade: int) -> List[np.ndarray]:
    out = []
    for start, n in runs:
        a, b = start * HOP, start * HOP + n * HOP
        seg = x[a:b].copy()
        if len(seg) <= 2 * fade:
            continue
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade))
        seg[:fade] *= ramp
        seg[-fade:] *= ramp[::-1]
        out.append(seg)
    return out


def concat_crossfade(segs: List[np.ndarray], fade: int) -> np.ndarray:
    if not segs:
        return np.zeros(0, dtype=np.float32)
    y = segs[0]
    for s in segs[1:]:
        y = np.concatenate([y[:-fade], y[-fade:] + s[:fade], s[fade:]])
    return y


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--frames", type=Path, default=paths.spectral() / "frames")
    ap.add_argument("--calls", type=Path, default=Path.home() / "calls", help="the raw recordings the survey read")
    ap.add_argument("--out", type=Path, default=paths.noise())
    ap.add_argument("--closed", type=float, default=-80.0, help="dBFS; quieter frames are the squelch closed")
    ap.add_argument("--under", type=float, default=20.0, help="dB below the file's voiced median a floor frame must be")
    ap.add_argument("--min-run", type=int, default=4, help="frames (16 ms) a run must last")
    ap.add_argument("--force", action="store_true", help="overwrite a bed that exists")
    args = ap.parse_args()

    by_feed: Dict[str, List[Tuple[str, List[Tuple[int, int]]]]] = defaultdict(list)
    for csv_path in sorted(args.frames.glob("*.csv")):
        rows = list(csv.DictReader(csv_path.open()))
        if not rows:
            continue
        feed = csv_path.stem.split("-")[1]
        runs = floor_runs(rows, args.closed, args.under, args.min_run)
        if runs:
            by_feed[feed].append((csv_path.stem, runs))

    args.out.mkdir(parents=True, exist_ok=True)
    fade = int(0.010 * SR)
    index_rows = []
    for feed in sorted(by_feed):
        dst = args.out / f"feed{feed}_floor.wav"
        if dst.exists() and not args.force:
            print(f"[skip] {dst} exists (--force to overwrite)")
            continue
        segs: List[np.ndarray] = []
        for stem, runs in by_feed[feed]:
            src = args.calls / f"{stem}.m4a"
            if not src.exists():
                print(f"[warn] {src} missing", file=sys.stderr)
                continue
            x = decode(src)
            got = cut(x, runs, fade)
            segs += got
            index_rows.append((feed, stem, len(got), round(sum(len(g) for g in got) / SR, 2)))
        y = concat_crossfade(segs, fade)
        if len(y) < SR:
            print(f"[warn] feed {feed}: only {len(y) / SR:.1f} s of floor; not written", file=sys.stderr)
            continue
        from radioize import write_wav
        write_wav(dst, y)
        rms = 20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-9)
        print(f"[OK] {dst}: {len(y) / SR:.1f} s from {len(segs)} runs in {len(by_feed[feed])} files, {rms:.1f} dBFS")
    with (args.out / "INDEX.tsv").open("a") as f:
        if f.tell() == 0:
            f.write("feed\tsource\truns\tseconds\n")
        for r in index_rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
