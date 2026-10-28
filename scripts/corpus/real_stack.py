#!/usr/bin/env python3
"""real_stack.py — the transcriber stacks on the real call-sets that have a human reference.

    python scripts/corpus/real_stack.py --cloud            # Groq rows (27 calls, paced)
    python scripts/corpus/real_stack.py --local            # faster-whisper medium rows (CPU, ~1 min per burst)
    python scripts/corpus/real_stack.py --md docs/reports/REAL_STACK.md   # tables only, from the cache

Rows: the base model alone (no prompt, no re-decode, no corrector) against the app's stack (the radio-phrase
hint, the gap re-decode locally, the transcript corrector), locally (medium) and in the cloud (large-v3-turbo,
and large-v3 with the stack). Every burst's transcript is cached in ~/labels/reference/<CS>/<seg>.stack.json
so a rerun scores without transcribing. Scores: WER (edits over reference words, pooled), mean 1- to 4-gram F
and 4-gram recall (word-weighted), per call-set and over all bursts.
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
from ngram_scores import scores as ngram  # noqa: E402
from score_callsets import wer  # noqa: E402

CLOUD = ("cloud_base", "cloud_ours", "cloud_large_ours")
LOCAL = ("local_base", "local_ours")
LABEL = {"local_base": "local medium, base (no prompt)", "local_ours": "local medium + hint + gap re-decode + corrector",
         "cloud_base": "cloud large-v3-turbo, base (no prompt)", "cloud_ours": "cloud large-v3-turbo + hint + corrector",
         "cloud_large_ours": "cloud large-v3 + hint + corrector"}


def transcribers(which):
    from ems_handoff.config import load_env_files, load_settings
    from ems_handoff.transcribe import GroqTranscriber, LocalTranscriber
    from ems_handoff.transcript_corrector import correct_transcript
    load_env_files(); settings = load_settings()
    out = {}
    if "cloud" in which:
        base = GroqTranscriber(settings); base.hint = ""
        ours = GroqTranscriber(settings)
        large = GroqTranscriber(settings); large.model = "whisper-large-v3"
        out["cloud_base"] = (base, False); out["cloud_ours"] = (ours, True); out["cloud_large_ours"] = (large, True)
    if "local" in which:
        base = LocalTranscriber(settings, model_size="medium"); base.hint = ""; base.redecode = False
        ours = LocalTranscriber(settings, model_size="medium"); ours.redecode = True
        out["local_base"] = (base, False); out["local_ours"] = (ours, True)
    return out, correct_transcript


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--reference-dir", type=Path, default=Path.home() / "labels" / "reference")
    ap.add_argument("--cloud", action="store_true"); ap.add_argument("--local", action="store_true")
    ap.add_argument("--pace", type=float, default=3.2, help="seconds between cloud calls")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    which = [w for w, on in (("cloud", args.cloud), ("local", args.local)) if on]
    trs, corrector = transcribers(which) if which else ({}, None)
    sets = sorted(d.name for d in args.reference_dir.iterdir() if d.is_dir() and list(d.glob("*.ref.txt")))
    rows = []   # (cid, seg, config, ref, hyp)
    for cid in sets:
        info = json.loads((args.callsets_dir / cid / "callset_info.json").read_text())
        for b in info["bursts"]:
            seg = b["segment_id"]
            ref_path = args.reference_dir / cid / f"{seg}.ref.txt"
            if not ref_path.exists():
                continue
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
                print(f"{cid} {seg} {name}: {got[name]['seconds']} s  {text[:70]}", flush=True)
                cache.write_text(json.dumps(got, indent=1, ensure_ascii=False))
                if name.startswith("cloud"):
                    time.sleep(args.pace)
            ref = ref_path.read_text()
            for name, d in got.items():
                rows.append((cid, seg, name, ref, d["text"]))

    def table(select):
        agg = {}
        for cid, seg, name, ref, hyp in select:
            w, e, n = wer(ref, hyp); g = ngram(ref, hyp)
            a = agg.setdefault(name, {"e": 0, "n": 0, "f": 0.0, "r4": 0.0, "bursts": 0})
            a["e"] += e; a["n"] += n; a["f"] += g["avg_f"] * n; a["r4"] += g["r4"] * n; a["bursts"] += 1
        out = ["| stack | bursts | words | WER | avg n-gram F | 4-gram recall |", "|---|---|---|---|---|---|"]
        for name in sorted(agg, key=lambda k: agg[k]["e"] / max(agg[k]["n"], 1)):
            a = agg[name]
            out.append(f"| {LABEL.get(name, name)} | {a['bursts']} | {a['n']} | {100 * a['e'] / a['n']:.1f} % ({a['e']}/{a['n']}) | "
                       f"{a['f'] / a['n']:.3f} | {a['r4'] / a['n']:.2f} |")
        return "\n".join(out)

    md = [f"# The transcriber stacks on the real call-sets with a reference ({', '.join(sets)})", "",
          "Base = the model alone, no prompt, no re-decode, no corrector. Ours = the app's radio-phrase hint, the gap",
          "re-decode (local only) and the transcript corrector. WER pooled over all reference words; n-gram scores",
          "word-weighted. Cached transcripts: ~/labels/reference/<CS>/<seg>.stack.json (scripts/corpus/real_stack.py).", "",
          "## All bursts", "", table(rows)]
    for cid in sets:
        md += ["", f"## {cid}", "", table([r for r in rows if r[0] == cid])]
    text = "\n".join(md) + "\n"
    print(text)
    if args.md:
        args.md.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
