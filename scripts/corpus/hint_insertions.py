#!/usr/bin/env python3
"""
hint_insertions.py — how often Whisper's radio-phrase hint shows up in a transcript that the reference does not support.

    python scripts/corpus/hint_insertions.py ~/datasets/synth_v1/evals/medium_hint ~/datasets/synth_v1/evals/medium_nohint

AIR_2026-10-26.md: on degraded audio Whisper wrote "LAMS, Cincinnati, last known well" (the tail of the app's hint,
ems_handoff.lexicon_rag.whisper_hint) where the crew had said vitals. For every eval_set.py run given, this counts, per
SNR, the clips whose transcript contains a distinctive hint phrase that the script text does not, and lists the phrases.
Phrases: the hint's multi-word items and its rare single words; common words (BP, pulse, temp) are not counted.
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_callsets import normalize  # noqa: E402

PHRASES = ["last known well", "lams", "cincinnati", "racemic epinephrine", "dexamethasone", "blow by oxygen", "blow by",
           "12 lead unremarkable", "normal sinus with pvcs", "eta 10 minutes", "aspirin 324", "nitroglycerin sublingual",
           "glasgow coma scale", "end tidal carbon dioxide", "mean arterial pressure", "capillary refill", "st elevation mi",
           "how do you copy", "priority 1", "priority 2", "priority 3"]


def norm(s: str) -> str:
    return " " + " ".join(normalize(s)) + " "


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 2
    for ev in sys.argv[1:]:
        ev = Path(ev)
        set_dir = ev.parent.parent
        per_snr: dict = defaultdict(lambda: [0, 0, Counter()])
        for t in sorted((ev / "transcripts").glob("*.txt")):
            m = re.match(r"(SY\d+)_(.+)_(snr\w+)\.txt", t.name)
            if not m:
                continue
            sid, _voice, snr = m.groups()
            ref = norm((set_dir / "scripts" / f"{sid}.txt").read_text())
            hyp = norm(t.read_text())
            inserted = [p for p in PHRASES if f" {p} " in hyp and f" {p} " not in ref]
            per_snr[snr][0] += 1
            if inserted:
                per_snr[snr][1] += 1
                per_snr[snr][2].update(inserted)
        print(f"\n## {ev.name}\n\n| SNR | clips | clips with a hint phrase the script lacks | phrases |\n|---|---|---|---|")
        for snr in sorted(per_snr, key=lambda s: (0, 0) if s == "snrinf" else (1, -float(s[3:]))):
            n, k, c = per_snr[snr]
            print(f"| {snr[3:]} | {n} | {k} | {', '.join(f'{p} x{v}' for p, v in c.most_common(6))} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
