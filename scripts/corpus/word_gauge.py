#!/usr/bin/env python3
"""
word_gauge.py — how well the production decoder's own confidence detects its own mistakes, from word_confidence.py dumps.

Two questions, both answered on scripts held out by script id (70 / 30 split), with a bootstrap interval over scripts
and a label-permutation control:

1. Word level: does faster-whisper's per-word probability (lower = less sure) detect a wrong word (substituted, inserted,
   looped)? Also -avg_logprob of the word's segment, and a logistic regression over [1 - p, -avg_logprob, no_speech_prob,
   compression_ratio].
2. Segment level: does the segment's average log-probability detect a bad segment (30 % or more of its words wrong, or a
   segment whose words are under half of what its duration implies at 2 words a second: a collapse)? Also the no-speech
   probability and the compression ratio.

Deleted reference words have no word to score; they are reported as a rate so the reader knows what the word-level
detector cannot see.

    python3 scripts/corpus/word_gauge.py --dump /data/trajectories/fw_small_train --out /data/trajectories/fw_small_gauge
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

BAD = {"substituted", "inserted", "loop"}


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort"); s = score[order]
    ranks = np.empty(len(score), dtype=np.float64); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1; i = j + 1
    pos = y == 1; n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def oriented(score, y):
    a = auroc(score, y); return (a, False) if a >= 0.5 else (1 - a, True)


def logistic(Xtr, ytr, Xte, steps=600):
    """A small logistic regression in numpy (no torch in the faster-whisper container)."""
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6; Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    w = np.zeros(Xtr.shape[1]); b = 0.0; pw = (1 - ytr.mean()) / max(ytr.mean(), 1e-6); lr = 0.1
    for _ in range(steps):
        z = Xtr @ w + b; p = 1 / (1 + np.exp(-z)); wt = np.where(ytr == 1, pw, 1.0)
        g = (p - ytr) * wt; w -= lr * (Xtr.T @ g) / len(ytr); b -= lr * g.mean()
    return Xte @ w + b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    clips = [json.load(open(p)) for p in sorted(glob.glob(str(args.dump / "*.json")))]
    sids = sorted({c["id"] for c in clips}); rng.shuffle(sids)
    train_ids = set(sids[:int(0.7 * len(sids))])
    W, S = [], []          # word rows, segment rows
    deleted_words = sum(sum(b - a + 1 for a, b, _ in c["deleted"]) for c in clips); ref_words = sum(c["ref_words"] for c in clips)
    for c in clips:
        segs = c["segments"]
        bad_in_seg = [0] * len(segs); words_in_seg = [0] * len(segs)
        for w in c["words"]:
            lab = w.get("label", "punct")
            if lab in ("punct", "special"):
                continue
            s = segs[w["segment"]]
            W.append({"sid": c["id"], "snr": c["snr"], "y": 1 if lab in BAD else 0, "p": w["probability"], "avg_logprob": s["avg_logprob"],
                      "no_speech": s["no_speech_prob"], "cr": s["compression_ratio"]})
            words_in_seg[w["segment"]] += 1; bad_in_seg[w["segment"]] += 1 if lab in BAD else 0
        for k, s in enumerate(segs):
            dur = max(s["end"] - s["start"], 0.1); expected = 2.0 * dur
            collapsed = words_in_seg[k] < 0.5 * expected and dur >= 6.0
            frac_bad = bad_in_seg[k] / words_in_seg[k] if words_in_seg[k] else 1.0
            S.append({"sid": c["id"], "y": 1 if (frac_bad >= 0.30 or collapsed) else 0, "avg_logprob": s["avg_logprob"], "no_speech": s["no_speech_prob"],
                      "cr": s["compression_ratio"], "wps": words_in_seg[k] / dur})
    def split(rows):
        tr = [r for r in rows if r["sid"] in train_ids]; te = [r for r in rows if r["sid"] not in train_ids]; return tr, te
    Wtr, Wte = split(W); Str, Ste = split(S)
    res = {"clips": len(clips), "words": len(W), "test_words": len(Wte), "test_bad_words": int(sum(r["y"] for r in Wte)),
           "deleted_reference_words": deleted_words, "deleted_share_of_reference": round(deleted_words / max(ref_words, 1), 4),
           "segments": len(S), "test_segments": len(Ste), "test_bad_segments": int(sum(r["y"] for r in Ste)), "word_auroc": {}, "segment_auroc": {}}
    yw = np.array([r["y"] for r in Wte]); ys = np.array([r["y"] for r in Ste])
    feats_w = {"1 - word probability": np.array([1 - r["p"] for r in Wte]), "-segment avg_logprob": np.array([-r["avg_logprob"] for r in Wte]),
               "segment no_speech_prob": np.array([r["no_speech"] for r in Wte]), "segment compression_ratio": np.array([r["cr"] for r in Wte])}
    for k, v in feats_w.items():
        a, inv = oriented(v, yw); res["word_auroc"][k + (" (inverted)" if inv else "")] = a
    Xtr = np.array([[1 - r["p"], -r["avg_logprob"], r["no_speech"], r["cr"]] for r in Wtr]); Xte = np.array([[1 - r["p"], -r["avg_logprob"], r["no_speech"], r["cr"]] for r in Wte])
    ytr = np.array([r["y"] for r in Wtr])
    res["word_auroc"]["logistic(word p, segment stats)"] = auroc(logistic(Xtr, ytr, Xte), yw)
    # controls and intervals for the headline word detector
    sc = feats_w["1 - word probability"]; sid_te = np.array([r["sid"] for r in Wte]); uniq = np.array(sorted(set(sid_te.tolist())))
    boots = []
    for _ in range(300):
        pick = rng.choice(uniq, len(uniq)); mask = np.concatenate([np.where(sid_te == s)[0] for s in pick])
        if yw[mask].sum() and (1 - yw[mask]).sum():
            boots.append(auroc(sc[mask], yw[mask]))
    res["word_checks"] = {"bootstrap 95 % over test scripts": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                          "permuted labels (control)": auroc(sc, rng.permutation(yw)),
                          "per level": {lvl: auroc(sc[np.array([r["snr"] for r in Wte]) == lvl], yw[np.array([r["snr"] for r in Wte]) == lvl]) for lvl in sorted({r["snr"] for r in Wte})}}
    feats_s = {"-avg_logprob": np.array([-r["avg_logprob"] for r in Ste]), "no_speech_prob": np.array([r["no_speech"] for r in Ste]),
               "compression_ratio": np.array([r["cr"] for r in Ste]), "-words per second": np.array([-r["wps"] for r in Ste])}
    for k, v in feats_s.items():
        a, inv = oriented(v, ys); res["segment_auroc"][k + (" (inverted)" if inv else "")] = a
    if len(Str) and ys.sum():
        Xs_tr = np.array([[-r["avg_logprob"], r["no_speech"], r["cr"], -r["wps"]] for r in Str]); Xs_te = np.array([[-r["avg_logprob"], r["no_speech"], r["cr"], -r["wps"]] for r in Ste])
        res["segment_auroc"]["logistic(all segment stats)"] = auroc(logistic(Xs_tr, np.array([r["y"] for r in Str]), Xs_te), ys)
    # a conformal threshold on 1 - p at 5 % false alarm, from the training split's correct words
    q = float(np.quantile(np.array([1 - r["p"] for r in Wtr if r["y"] == 0]), 0.95))
    res["word_conformal"] = {"alpha": 0.05, "threshold_1_minus_p": q, "test_false_alarm": float((sc[yw == 0] > q).mean()), "test_detection": float((sc[yw == 1] > q).mean())}
    args.out.mkdir(parents=True, exist_ok=True); (args.out / "results.json").write_text(json.dumps(res, indent=1))
    md = ["# Production decoder confidence as a detector (word_gauge)", "",
          f"Clips {res['clips']}; test words {res['test_words']}, bad {res['test_bad_words']}; deleted reference words {deleted_words} ({100 * res['deleted_share_of_reference']:.1f} % of the reference, invisible to a word detector); test segments {res['test_segments']}, bad {res['test_bad_segments']}.", "",
          "| word-level score | AUROC |", "|---|---|"] + [f"| {k} | {v:.3f} |" for k, v in res["word_auroc"].items()] + \
         ["", f"Checks on 1 - p: {json.dumps(res['word_checks'])}", f"Conformal: {json.dumps(res['word_conformal'])}", "", "| segment-level score | AUROC |", "|---|---|"] + \
         [f"| {k} | {v:.3f} |" for k, v in res["segment_auroc"].items()]
    (args.out / "results.md").write_text("\n".join(md) + "\n"); print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
