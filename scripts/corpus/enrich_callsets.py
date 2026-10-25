#!/usr/bin/env python3
"""Write per-burst timing into every callset_info.json so a call-set folder replays on its own.

    enrich_callsets.py [--callsets ~/callsets] [--index ~/labels/callsets.csv] [--dry-run]

Adds a "bursts" list to callset_info.json: for each segment in "segments" order, its
segment_id, source_file, feed, epoch, duration_sec (from the WAV), gap_from_previous_sec
(from the index when it has one, else epoch minus the previous burst's end) and words.
Existing keys are kept. Run once after the index changes; idempotent."""
import argparse, csv, json, re, sys, wave
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--callsets", default="~/callsets"); ap.add_argument("--index", default="~/labels/callsets.csv")
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()
root, index_path = Path(args.callsets).expanduser(), Path(args.index).expanduser()
PATTERN = re.compile(r"mocomdps-(\d+)-(\d{9,11})")

index = {}
if index_path.exists():
    with open(index_path, newline="") as f:
        index = {r["segment_id"]: r for r in csv.DictReader(f)}
else:
    print(f"index {index_path} not found; using filenames only", file=sys.stderr)

def duration(wav):
    with wave.open(str(wav), "rb") as w:
        return round(w.getnframes() / w.getframerate(), 2)

changed = 0
for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p / "callset_info.json").exists()):
    info = json.loads((folder / "callset_info.json").read_text())
    bursts, prev_end = [], None
    for sid in info.get("segments") or []:
        row = index.get(sid, {})
        src = row.get("source_file", "")
        m = PATTERN.search(src)
        feed, epoch = (m.group(1), int(m.group(2))) if m else (row.get("feed") or info.get("feed", ""), None)
        if epoch is None and row.get("epoch"):
            epoch = int(float(row["epoch"]))
        wav = folder / f"{sid}.wav"
        dur = duration(wav) if wav.exists() else None
        txt = folder / f"{sid}.txt"
        words = len(txt.read_text().split()) if txt.exists() else int(row.get("words") or 0)
        gap = None
        if row.get("gap_from_previous_sec", "") not in ("", None):
            gap = float(row["gap_from_previous_sec"])
        elif epoch is not None and prev_end is not None:
            gap = round(epoch - prev_end, 1)
        bursts.append({"segment_id": sid, "source_file": src, "feed": feed, "epoch": epoch,
                       "duration_sec": dur, "gap_from_previous_sec": gap, "words": words})
        if epoch is not None and dur is not None:
            prev_end = max(prev_end or 0, epoch + dur)
    if info.get("bursts") == bursts:
        continue
    info["bursts"] = bursts
    changed += 1
    print(f"{info.get('callset_id', folder.name)}: {len(bursts)} bursts, "
          f"{sum(1 for b in bursts if b['epoch'] is not None)} with epoch")
    if not args.dry_run:
        (folder / "callset_info.json").write_text(json.dumps(info, indent=1) + "\n")
print(f"{changed} callset_info.json file(s) {'would be ' if args.dry_run else ''}updated")
