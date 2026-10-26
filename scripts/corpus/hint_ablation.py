#!/usr/bin/env python3
"""Which Whisper hint transcribes the real calls best? Every burst of the given call-sets, each
hint variant, N repeats on Groq whisper-large-v3-turbo, word error rate against the reference.

    hint_ablation.py CS008 CS014 CS001 [--repeats 3] [--md FILE]

Variants:
    none         no prompt at all
    terms        the lexicon terminology only (the hint before 2026-10-25)
    radio        terminology + the radio phrases (priority 1/2/3, how do you copy, the drugs)
    radio+names  radio + the local names (HANDOFF_RADIO_VOCAB)
The table shows the mean WER per burst and over all words, so a variant that helps one call and
hurts another is visible. Whisper on Groq is not deterministic at temperature 0, hence repeats.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ems_handoff.audio.sources import load_wav  # noqa: E402
from ems_handoff.config import load_env_files, load_settings  # noqa: E402
from ems_handoff.lexicon_rag import whisper_hint  # noqa: E402
from ems_handoff.transcribe import to_wav_bytes  # noqa: E402

spec = importlib.util.spec_from_file_location("score_callsets", ROOT / "scripts" / "corpus" / "score_callsets.py")
sc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sc)


def terms_only() -> str:
    import ems_handoff.lexicon_rag as lr
    saved = lr.RADIO_PHRASES
    try:
        lr.RADIO_PHRASES = ""
        return whisper_hint().rstrip(". ") + "."
    finally:
        lr.RADIO_PHRASES = saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("callsets", nargs="+")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--reference-dir", type=Path, default=Path.home() / "labels" / "reference")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    load_env_files()
    settings = load_settings()
    from openai import OpenAI
    client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key, timeout=120, max_retries=2)
    variants = {"none": None, "terms": terms_only(), "radio": whisper_hint(), "radio+names": whisper_hint(vocab=settings.radio_vocab)}
    results = {}   # (variant) -> list of (edits, words) per burst
    lines = ["# Whisper hint ablation (Groq whisper-large-v3-turbo, temperature 0, WER against ~/labels/reference)", ""]
    for cid in args.callsets:
        info = json.loads((args.callsets_dir / cid / "callset_info.json").read_text())
        segs = [b["segment_id"] for b in info["bursts"]]
        lines.append(f"## {cid}")
        lines.append("")
        lines.append("| variant | " + " | ".join(segs) + " | all |")
        lines.append("|---|" + "---|" * (len(segs) + 1))
        for name, prompt in variants.items():
            cells, edits, words = [], 0, 0
            for seg in segs:
                ref_path = args.reference_dir / cid / f"{seg}.ref.txt"
                if not ref_path.exists():
                    cells.append("-")
                    continue
                ref = ref_path.read_text()
                audio = load_wav(args.callsets_dir / cid / f"{seg}.wav")
                wers = []
                for _ in range(args.repeats):
                    kw = {"prompt": prompt[-800:]} if prompt else {}
                    for attempt in range(3):
                        try:
                            r = client.audio.transcriptions.create(file=("burst.wav", to_wav_bytes(audio)), model="whisper-large-v3-turbo",
                                                                   language="en", temperature=0.0, response_format="json", **kw)
                            break
                        except Exception as e:  # noqa: BLE001
                            if attempt == 2:
                                raise
                            time.sleep(5)
                    w, e, n = sc.wer(ref, (r.text or "").strip())
                    wers.append(w)
                    edits += e
                    words += n
                mean = statistics.mean(wers)
                spread = f"±{(max(wers) - min(wers)) * 50:.0f}" if len(wers) > 1 and max(wers) != min(wers) else ""
                cells.append(f"{mean * 100:.0f}%{spread}")
                print(f"{cid} {seg} {name}: {[round(x * 100) for x in wers]}", flush=True)
            results.setdefault(name, []).append((edits, words))
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {100 * edits / max(1, words):.0f}% |")
        lines.append("")
    lines.append("## All call-sets")
    lines.append("")
    lines.append("| variant | WER over all words |")
    lines.append("|---|---|")
    for name, pairs in results.items():
        e, n = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        lines.append(f"| {name} | {100 * e / max(1, n):.1f}% ({e}/{n}) |")
    text = "\n".join(lines)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
