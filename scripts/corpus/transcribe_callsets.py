#!/usr/bin/env python3
"""transcribe_callsets.py — every burst of every call-set through the transcriber stacks, reference or not.

    python scripts/corpus/transcribe_callsets.py --cloud                    # Groq rows, all call-sets, paced
    python scripts/corpus/transcribe_callsets.py --local --ids CS012 CS013  # medium rows for a subset (one box of two)
    python scripts/corpus/transcribe_callsets.py --md out.md                # table of what is cached

Same stacks and the same cache as real_stack.py (~/labels/reference/<CS>/<seg>.stack.json), so a call-set
that later gets a human reference is scored by real_stack.py without transcribing again. Rows: local_base
(medium, no prompt), local_ours (medium + hint + gap re-decode + corrector), cloud_base (large-v3-turbo, no
prompt), cloud_ours (large-v3-turbo + hint + corrector), cloud_large_ours (large-v3 + hint + corrector).
The split by --ids is how two Jetsons share one run: each takes half the call-sets and writes its own caches.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from draft_reference import load_wav  # noqa: E402
from real_stack import LABEL, transcribers  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--reference-dir", type=Path, default=Path.home() / "labels" / "reference")
    ap.add_argument("--ids", nargs="*", help="call-set ids; default every folder in --callsets-dir")
    ap.add_argument("--cloud", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--only", nargs="*", help="stack names to run, e.g. local_ours")
    ap.add_argument("--pace", type=float, default=3.2, help="seconds between cloud calls")
    ap.add_argument("--md", type=Path, help="write a table of cached transcripts here")
    args = ap.parse_args(argv)
    which = [w for w, on in (("cloud", args.cloud), ("local", args.local)) if on]
    trs, corrector = transcribers(which) if which else ({}, None)
    if args.only:
        trs = {k: v for k, v in trs.items() if k in args.only}
    sets = args.ids or sorted(d.name for d in args.callsets_dir.iterdir() if (d / "callset_info.json").exists())
    t_start = time.time(); done = 0
    for cid in sets:
        info = json.loads((args.callsets_dir / cid / "callset_info.json").read_text())
        (args.reference_dir / cid).mkdir(parents=True, exist_ok=True)
        for b in info["bursts"]:
            seg = b["segment_id"]
            cache = args.reference_dir / cid / f"{seg}.stack.json"
            got = json.loads(cache.read_text()) if cache.exists() else {}
            audio = None
            for name, (t, corr) in trs.items():
                if name in got:
                    continue
                audio = load_wav(args.callsets_dir / cid / f"{seg}.wav") if audio is None else audio
                t0 = time.time()
                try:
                    text = t.transcribe(audio)
                    if corr:
                        text = corrector(text).text
                    got[name] = {"text": text, "seconds": round(time.time() - t0, 1), "notes": getattr(t, "last_notes", None)}
                except Exception as e:  # noqa: BLE001
                    print(f"[error] {cid} {seg} {name}: {e}", flush=True); continue
                done += 1
                print(f"{cid} {seg} {name}: {got[name]['seconds']} s  {text[:70]}", flush=True)
                cache.write_text(json.dumps(got, indent=1, ensure_ascii=False))
                if name.startswith("cloud"):
                    time.sleep(args.pace)
    print(f"[OK] {done} transcripts in {time.time() - t_start:.0f} s", flush=True)
    if args.md:
        lines = ["| call-set | burst | audio s | " + " | ".join(LABEL.get(k, k) for k in LABEL) + " |",
                 "|---|---|---|" + "---|" * len(LABEL)]
        for cid in sets:
            info = json.loads((args.callsets_dir / cid / "callset_info.json").read_text())
            for b in info["bursts"]:
                seg = b["segment_id"]
                cache = args.reference_dir / cid / f"{seg}.stack.json"
                got = json.loads(cache.read_text()) if cache.exists() else {}
                lines.append(f"| {cid} | {seg} | {b.get('duration_sec', '')} | " +
                             " | ".join(got.get(k, {}).get("text", "").replace("|", "/") for k in LABEL) + " |")
        args.md.write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
