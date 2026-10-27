#!/usr/bin/env python3
"""
eval_errors.py — where a transcriber's errors on the generated set come from.

For one eval folder (evals/<label>) and one SNR tag, aligns every transcript with its written script and
prints the term occurrences missed, the most frequent edit spans (script words -> transcript words), and
the worst clips. Two eval folders side by side show what a change (hint, corrector, model) fixed and broke.

    .venv/bin/python scripts/corpus/eval_errors.py ~/datasets/synth_v1/evals/medium_hint --snr inf
    .venv/bin/python scripts/corpus/eval_errors.py ~/datasets/synth_v1/evals/medium_hint ~/datasets/synth_v1/evals/groq_hint --snr 5 --top 25
"""
from __future__ import annotations

import argparse
import difflib
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_callsets import normalize  # noqa: E402


def analyse(folder: Path, tag: str, set_dir: Path) -> Dict[str, object]:
    terms = [" ".join(normalize(t)) for t in (set_dir / "terms.txt").read_text().splitlines() if t.strip()]
    lines = (folder / "rows.tsv").read_text().splitlines()
    cols = lines[0].split("\t")
    rows = [dict(zip(cols, l.split("\t"))) for l in lines[1:] if l.strip()]
    rows = [r for r in rows if r["snr"] == tag]
    missed: Counter = Counter()
    edits: Counter = Counter()
    ref_words = 0
    spans = 0
    for r in rows:
        ref = normalize((set_dir / "scripts" / f"{r['id']}.txt").read_text())
        hyp = normalize((folder / "transcripts" / f"{r['id']}_{r['voice']}_{tag}.txt").read_text())
        refs, hyps = " ".join(ref), " ".join(hyp)
        for t in terms:
            c, h = refs.count(t), hyps.count(t)
            if c > h:
                missed[t] += c - h
        sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
        for op, a0, a1, b0, b1 in sm.get_opcodes():
            if op != "equal":
                edits[(" ".join(ref[a0:a1]), " ".join(hyp[b0:b1]))] += 1
                spans += 1
        ref_words += len(ref)
    worst = sorted(rows, key=lambda r: -float(r["wer"]))[:8]
    return {"rows": rows, "missed": missed, "edits": edits, "ref_words": ref_words, "spans": spans, "worst": worst}


def report(folder: Path, a: Dict[str, object], top: int) -> str:
    out = [f"## {folder.name} at {a['rows'][0]['snr'] if a['rows'] else '?'}: {len(a['rows'])} clips, "
           f"{a['spans']} edit spans over {a['ref_words']} reference words", ""]
    out.append("Terms missed: " + (", ".join(f"{t} {n}" for t, n in a["missed"].most_common()) or "none"))
    out.append("")
    out.append("| n | script | transcript |")
    out.append("|---|---|---|")
    for (r, h), n in a["edits"].most_common(top):
        out.append(f"| {n} | {r or '(nothing)'} | {h or '(dropped)'} |")
    out.append("")
    out.append("Worst clips: " + ", ".join(f"{r['id']} {r['family']} {float(r['wer']):.1%}" for r in a["worst"]))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("folders", nargs="+", type=Path)
    ap.add_argument("--snr", default="inf")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--set", type=Path, help="the set directory (default: the folder's grandparent)")
    args = ap.parse_args()
    tag = "snrinf" if args.snr == "inf" else f"snr{int(float(args.snr))}"
    for f in args.folders:
        set_dir = args.set or f.resolve().parent.parent
        print(report(f, analyse(f, tag, set_dir), args.top))
        print()
    if len(args.folders) == 2:
        a, b = (analyse(f, tag, args.set or f.resolve().parent.parent) for f in args.folders)
        fixed = a["missed"] - b["missed"]
        broke = b["missed"] - a["missed"]
        print(f"## {args.folders[0].name} -> {args.folders[1].name}")
        print("Terms recovered: " + (", ".join(f"{t} {n}" for t, n in fixed.most_common()) or "none"))
        print("Terms lost: " + (", ".join(f"{t} {n}" for t, n in broke.most_common()) or "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
