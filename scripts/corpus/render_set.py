#!/usr/bin/env python3
"""
render_set.py — read a generated set aloud and put it through the radio channel, laid out for replay mode.

    python scripts/corpus/render_set.py                              # all of synth_v1, Piper voice, SNR inf 20 15 10 5
    python scripts/corpus/render_set.py --split test --snr inf 10     # only the test split, two SNRs
    python scripts/corpus/render_set.py --voice xtts_cs008 --clean-dir ~/datasets/synth_v1/audio/xtts_cs008/clean
                                                                     # clips rendered elsewhere (a GPU box): channel only

Per script id and voice, under <set>/audio/<voice>/:
    clean/<id>.wav          the voice reading <id>.spoken.txt, 16 kHz mono (Piper here; --clean-dir for other voices)
    snr<X>/<id>.wav         radioize: corpus EQ, 4.6 kHz brick wall, -26 dBFS, soft limiter, then the real feed-4390
                            floor at X dB under the speech; snrinf = the channel with no noise bed
and under <set>/callsets/<id>_<voice>_snr<X>/ a one-burst call-set (callset_info.json, <id>_01.wav as a link to the
clip, <id>_01.txt and combined_transcript.txt = the script text) so the app's replay mode plays it with
HANDOFF_CALLSETS=<set>/callsets. Existing files are kept (re-run to fill gaps); --force redoes them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from radioize import decode, radioize, read_spectra, write_wav  # noqa: E402
from spectral_survey import SR  # noqa: E402

SNRS_DEFAULT = ["inf", "20", "15", "10", "5"]


def snr_tag(s: str) -> str:
    return "snrinf" if s == "inf" else f"snr{int(float(s))}"


def piper_render(spoken_txt: Path, dst: Path, piper: Path, voice: Path, length_scale: float = 1.15) -> None:
    tmp = dst.with_suffix(".piper.wav")
    subprocess.run([str(piper), "-m", str(voice), "--length-scale", str(length_scale), "--sentence-silence", "0.4", "-f", str(tmp)],
                   input=spoken_txt.read_bytes(), check=True, capture_output=True)
    write_wav(dst, decode(tmp))            # 22.05 kHz -> 16 kHz mono
    tmp.unlink()


def callset_for(set_dir: Path, sid: str, voice: str, tag: str, wav: Path, text: str, gold: dict, seconds: float) -> Path:
    cid = f"{sid}_{voice}_{tag}"
    folder = set_dir / "callsets" / cid
    folder.mkdir(parents=True, exist_ok=True)
    seg = f"{sid}_01"
    link = folder / f"{seg}.wav"
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(os.path.relpath(wav, folder))
    (folder / f"{seg}.txt").write_text(text)
    (folder / "combined_transcript.txt").write_text(text)
    stated = gold.get("stated", {})
    age = next((f["expect"] for f in gold["fields"] if f["path"] == "patient.age" and f["kind"] == "int"), None)
    sex = next((f["expect"] for f in gold["fields"] if f["path"] == "patient.sex"), None)
    when = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=int(sid[2:]) * 10)
    info = {
        "callset_id": cid, "feed": "synth", "hospital": [stated.get("hospital", "")], "units": [stated.get("unit", "")],
        "patient": {"ages": [f"{age}y"] if age else [], "sex": [sex.lower()] if sex else []},
        "complaint_keywords": [], "first_burst_utc": when.strftime("%Y-%m-%d %H:%M:%S"), "last_burst_utc": when.strftime("%Y-%m-%d %H:%M:%S"),
        "num_bursts": 1, "total_words": len(text.split()), "total_audio_sec": round(seconds, 1), "over_cap": False,
        "has_clear_opening": True, "has_clear_closing": True, "appears_truncated": "no", "fragments": [], "segments": [seg],
        "related_callset": "", "related_evidence": "",
        "bursts": [{"segment_id": seg, "source_file": str(wav.relative_to(set_dir)), "feed": "synth", "epoch": int(when.timestamp()),
                    "duration_sec": round(seconds, 2), "gap_from_previous_sec": 0.0, "words": len(text.split())}],
        "synth": {"script": sid, "voice": voice, "snr": tag, "gold": f"gold/{sid}.json", "family": gold.get("family")},
    }
    (folder / "callset_info.json").write_text(json.dumps(info, indent=1) + "\n")
    return folder


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", type=Path, default=paths.datasets() / "synth_v1")
    ap.add_argument("--voice", default="piper", help="voice name for the audio/<voice> folder")
    ap.add_argument("--clean-dir", type=Path, help="clean 16 kHz clips rendered elsewhere, <id>.wav; default: render with Piper")
    ap.add_argument("--snr", nargs="+", default=SNRS_DEFAULT, help="inf and/or dB values")
    ap.add_argument("--split", choices=["train", "val", "test"], help="only the ids of this split")
    ap.add_argument("--ids", nargs="*", help="only these ids")
    ap.add_argument("--noise-file", type=Path, default=paths.noise() / "feed4390_floor.wav")
    ap.add_argument("--spectra", type=Path, default=paths.spectral() / "spectra.csv")
    ap.add_argument("--piper", type=Path, default=paths.piper())
    ap.add_argument("--piper-voice", type=Path, default=paths.piper_voice())
    ap.add_argument("--no-callsets", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ids: List[str] = sorted(p.stem for p in (args.set / "scripts").glob("SY*.txt") if not p.name.endswith(".spoken.txt"))
    if args.split:
        ids = [i for i in ids if i in set(json.loads((args.set / "splits.json").read_text())[args.split])]
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    target_db, _ = read_spectra(args.spectra)
    bed = decode(args.noise_file)
    clean_dir = args.clean_dir or (args.set / "audio" / args.voice / "clean")
    clean_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); made = 0
    for n, sid in enumerate(ids, 1):
        text = (args.set / "scripts" / f"{sid}.txt").read_text()
        gold = json.loads((args.set / "gold" / f"{sid}.json").read_text())
        clean = clean_dir / f"{sid}.wav"
        if not clean.exists() or (args.force and not args.clean_dir):
            if args.clean_dir:
                print(f"[warn] {clean} missing; skipped", file=sys.stderr); continue
            piper_render(args.set / "scripts" / f"{sid}.spoken.txt", clean, args.piper, args.piper_voice)
        x: Optional[np.ndarray] = None
        for s in args.snr:
            tag = snr_tag(s)
            dst = args.set / "audio" / args.voice / tag / f"{sid}.wav"
            if not dst.exists() or args.force:
                if x is None:
                    x = decode(clean)
                y, _eq = radioize(x, target_db, bed if s != "inf" else None, None if s == "inf" else float(s))
                write_wav(dst, y); made += 1
            if not args.no_callsets:
                import wave
                with wave.open(str(dst)) as w:
                    seconds = w.getnframes() / w.getframerate()
                callset_for(args.set, sid, args.voice, tag, dst, text, gold, seconds)
        if n % 10 == 0 or n == len(ids):
            print(f"[{n}/{len(ids)}] {sid} done, {made} clips written, {time.time() - t0:.0f} s", flush=True)
    print(f"[OK] {len(ids)} scripts x {len(args.snr)} SNRs under {args.set / 'audio' / args.voice}; call-sets in {args.set / 'callsets'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
