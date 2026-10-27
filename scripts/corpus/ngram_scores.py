#!/usr/bin/env python3
"""
ngram_scores.py — n-gram agreement with the reference, per transcriber, so "ours" can be set against
off-the-shelf Whisper (Groq turbo, Groq large, local medium, Vosk) on the gold call-sets.

    python scripts/corpus/ngram_scores.py                    # CS001, CS008, CS014 from $HANDOFF_LABELS/reference
    python scripts/corpus/ngram_scores.py --summaries ~/recordings/2026-10-26_d1f4373   # add the app's runs

For every burst with a reference, and every candidate transcript (the `<seg>.candidates.json` written by
draft_reference.py: groq_turbo, groq_turbo_radio, groq_large, local_medium, vosk_small, ...), plus the
app's own final transcript from recording summaries when --summaries is given:
    n-gram precision  share of the candidate's n-grams that occur in the reference (n = 1..4)
    n-gram recall     share of the reference's n-grams that occur in the candidate
    avg n-gram F      mean over n of the F score (the single "average n-gram" number)
    WER               from score_callsets.wer, for the usual comparison
Words are normalised as in score_callsets (lower case, digits for number words, units split).
Output: a table per call-set and a corpus-wide mean per transcriber, as markdown on stdout.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from score_callsets import burst_texts_from_session, normalize, wer  # noqa: E402

REF_DIR = paths.labels() / "reference"
CALLSETS = paths.callsets()


def ngrams(words: List[str], n: int) -> Counter:
    return Counter(tuple(words[i: i + n]) for i in range(len(words) - n + 1))


def ngram_pr(ref: str, hyp: str, n: int) -> Tuple[float, float]:
    r, h = ngrams(normalize(ref), n), ngrams(normalize(hyp), n)
    if not r or not h:
        return 0.0, 0.0
    common = sum((r & h).values())
    return common / sum(h.values()), common / sum(r.values())


def scores(ref: str, hyp: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    fs = []
    for n in (1, 2, 3, 4):
        p, r = ngram_pr(ref, hyp, n)
        f = 2 * p * r / (p + r) if p + r else 0.0
        out[f"p{n}"], out[f"r{n}"] = p, r
        fs.append(f)
    out["avg_f"] = sum(fs) / len(fs)
    out["wer"] = wer(ref, hyp)[0]
    return out


def app_transcripts(summaries: Path, cid: str, segs: List[str]) -> Dict[str, Dict[str, str]]:
    """{label: {seg: text}} from <CS>.<provider>.summary.json recordings (one app run per provider)."""
    out: Dict[str, Dict[str, str]] = {}
    for f in sorted(summaries.glob(f"{cid}.*.summary.json")) + sorted(summaries.glob(f"{cid}.summary.json")):
        d = json.loads(f.read_text())
        transcript = ((d.get("session") or {}).get("transcript")) or d.get("transcript") or ""
        texts = burst_texts_from_session(transcript)
        provider = f.name.split(".")[1] if f.name.count(".") > 2 else "app"
        label = f"app ({provider} run)"
        for seg, text in zip(segs, texts):
            out.setdefault(label, {})[seg] = text
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--callsets", nargs="*", default=["CS001", "CS008", "CS014"])
    ap.add_argument("--summaries", type=Path, help="recordings folder with <CS>.<provider>.summary.json")
    args = ap.parse_args()

    per_tr: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for cid in args.callsets:
        refs = {p.name.split(".")[0]: p.read_text().strip() for p in sorted((REF_DIR / cid).glob("*.ref.txt"))}
        if not refs:
            print(f"{cid}: no reference"); continue
        # the app's session lists bursts in time order (callset_info), not in segment-name order
        info_file = CALLSETS / cid / "callset_info.json"
        if info_file.exists():
            order = [b["segment_id"] for b in json.loads(info_file.read_text()).get("bursts", []) if b["segment_id"] in refs]
            refs = {seg: refs[seg] for seg in order + [s for s in refs if s not in order]}
        cands: Dict[str, Dict[str, str]] = defaultdict(dict)
        for seg in refs:
            cf = REF_DIR / cid / f"{seg}.candidates.json"
            if cf.exists():
                for name, text in json.loads(cf.read_text()).items():
                    if not name.startswith("_") and isinstance(text, str) and not text.startswith("<error"):
                        cands[name][seg] = text
            s3 = CALLSETS / cid / f"{seg}.txt"
            if s3.exists():
                cands["stage3 (groq turbo, corpus)"][seg] = s3.read_text()
        if args.summaries:
            for label, texts in app_transcripts(args.summaries, cid, list(refs)).items():
                cands[label].update(texts)
        print(f"\n### {cid} ({len(refs)} burst{'s' if len(refs) != 1 else ''}, {sum(len(normalize(r)) for r in refs.values())} reference words)\n")
        print("| transcriber | WER | 1-gram P/R | 2-gram P/R | 3-gram P/R | 4-gram P/R | avg n-gram F |")
        print("|---|---|---|---|---|---|---|")
        for name, texts in sorted(cands.items()):
            rows = [scores(refs[s], texts[s]) for s in refs if s in texts]
            if not rows:
                continue
            m = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
            per_tr[name].append(m)
            print(f"| {name} | {m['wer']:.0%} | " + " | ".join(f"{m[f'p{n}']:.2f}/{m[f'r{n}']:.2f}" for n in (1, 2, 3, 4)) + f" | {m['avg_f']:.3f} |")
    print("\n### Mean over call-sets\n")
    print("| transcriber | call-sets | WER | avg n-gram F | 4-gram recall |")
    print("|---|---|---|---|---|")
    for name, ms in sorted(per_tr.items(), key=lambda kv: -sum(m["avg_f"] for m in kv[1]) / len(kv[1])):
        k = len(ms)
        print(f"| {name} | {k} | {sum(m['wer'] for m in ms)/k:.0%} | {sum(m['avg_f'] for m in ms)/k:.3f} | {sum(m['r4'] for m in ms)/k:.2f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
