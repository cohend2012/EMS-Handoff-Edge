#!/usr/bin/env python3
"""
word_confidence.py — the production decoder's own per-word confidence as a wrong-word detector.

faster-whisper (what the app runs) exposes no hidden states, but with word timestamps it reports a probability for
every word, and per segment an average log-probability, a no-speech probability and a compression ratio. This script
runs the app's LocalTranscriber model over generated-set clips with those outputs on, aligns the words against the
script, labels each word correct / substituted / inserted, and saves everything, so the gauge can answer: how well does
the probability the production decoder already computes detect a wrong word, and how well does a segment's average
log-probability detect a collapsed window?

    # inside the faster-whisper container on the GPU Jetson, repo at /app, set at /data
    LOCAL_WHISPER_DEVICE=cuda LOCAL_WHISPER_COMPUTE=float16 python3 scripts/corpus/word_confidence.py \
        --set /data --split train --snr inf 5 --model small --out /data/trajectories/fw_small_train

Output: <out>/<id>_<snr>.json with words [{word, start, end, probability, label, segment}], segments [{start, end,
avg_logprob, no_speech_prob, compression_ratio, words, text}], the text, the reference and the deleted spans;
<out>/summary.tsv with WER and label counts per clip. Decode: greedy, VAD on, no prompt by default (the same as the
trusted scorer's --no-hint row), so the WER here must match eval_set's for the same model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "corpus"))

from score_callsets import normalize, wer  # noqa: E402
from trajectories import align_words, load_wav, loop_words  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", type=Path, default=Path.home() / "datasets" / "synth_v1")
    ap.add_argument("--split", choices=["train", "val", "test"], default="train")
    ap.add_argument("--snr", nargs="+", default=["inf", "5"])
    ap.add_argument("--voice", default="piper")
    ap.add_argument("--model", default="small")
    ap.add_argument("--hint", action="store_true", help="use the app's whisper hint (default: none, like the trusted no-hint row)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    from ems_handoff.config import load_env_files, load_settings
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE
    load_env_files()
    settings = load_settings()
    t = LocalTranscriber(settings, model_size=args.model)
    if not args.hint:
        t.hint = ""
    model = t._get_model()
    splits = json.loads((args.set / "splits.json").read_text())
    ids = sorted(splits[args.split])
    if args.limit:
        ids = ids[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in ids:
        ref_text = (args.set / "scripts" / f"{sid}.txt").read_text().strip()
        for snr in args.snr:
            tag = f"snr{snr}"
            wav = args.set / "audio" / args.voice / tag / f"{sid}.wav"
            if not wav.exists():
                continue
            x = load_wav(wav)
            t0 = time.perf_counter()
            segs, _info = model.transcribe(x, language="en", beam_size=1, temperature=0.0, condition_on_previous_text=False,
                                           initial_prompt=t.hint or None, vad_filter=True,
                                           vad_parameters={"min_silence_duration_ms": 500}, word_timestamps=True)
            segments, words = [], []
            for si, s in enumerate(segs):
                ws = [{"word": w.word, "start": w.start, "end": w.end, "probability": float(w.probability), "segment": si} for w in (s.words or [])]
                segments.append({"start": s.start, "end": s.end, "avg_logprob": float(s.avg_logprob), "no_speech_prob": float(s.no_speech_prob),
                                 "compression_ratio": float(s.compression_ratio), "words": len(ws), "text": s.text.strip()})
                words.extend(ws)
            text = " ".join(s["text"] for s in segments).strip()
            # labels: normalise each word, align the flat normalised sequence, give a raw word its first label
            ref_norm = normalize(ref_text)
            per = [normalize(w["word"]) for w in words]
            flat = [x_ for ws_ in per for x_ in ws_]
            labels_flat, deleted = align_words(ref_norm, flat)
            loops = loop_words(flat)
            k = 0
            for w, ws_ in zip(words, per):
                if not ws_:
                    w["label"] = "punct"; continue
                lab = labels_flat[k]
                if any((k + i) in loops for i in range(len(ws_))):
                    lab = "loop"
                w["label"] = lab; k += len(ws_)
            rate, edits, nref = wer(ref_text, text)
            (args.out / f"{sid}_{tag}.json").write_text(json.dumps({"id": sid, "snr": tag, "text": text, "reference": ref_text, "words": words,
                                                                     "segments": segments, "deleted": deleted, "wer": rate, "edits": edits, "ref_words": nref,
                                                                     "seconds": round(time.perf_counter() - t0, 2)}))
            counts = {l: sum(1 for w in words if w.get("label") == l) for l in ("correct", "substituted", "inserted", "loop")}
            rows.append({"id": sid, "snr": tag, "wer": rate, "edits": edits, "ref_words": nref, "words": len(words), **counts,
                         "segments": len(segments), "deleted_words": sum(b - a + 1 for a, b, _ in deleted), "seconds": round(time.perf_counter() - t0, 2)})
            print(f"{sid} {tag:<7} wer {rate:.3f} words {len(words)} correct {counts['correct']} sub {counts['substituted']} ins {counts['inserted']} "
                  f"loop {counts['loop']} del {rows[-1]['deleted_words']} segs {len(segments)} ({rows[-1]['seconds']} s)", flush=True)
    if rows:
        keys = list(rows[0].keys())
        with open(args.out / "summary.tsv", "w") as f:
            f.write("\t".join(keys) + "\n")
            for r in rows:
                f.write("\t".join(str(r[k]) for k in keys) + "\n")
    print(f"done: {len(rows)} clips -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
