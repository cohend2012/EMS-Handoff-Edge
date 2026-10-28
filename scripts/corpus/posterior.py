#!/usr/bin/env python3
"""posterior.py — the statistics the abstract named, computed on our rows: a Beta-Binomial posterior
probability that one run beats another on a per-trial outcome, and a paired bootstrap interval on the
per-clip score difference.

    python scripts/corpus/posterior.py --evals medium_nohint medium_hint_rd_corr      # term recall, WER
    python scripts/corpus/posterior.py --reports medium_nohint__qwen3-4b-instruct medium_hint_rd_corr__qwen3-4b-instruct

Trials for an eval run: each of the 200 term occurrences in the test scripts is one Bernoulli trial (the
transcript has the term or not), so term recall ~ Binomial; with a flat Beta(1,1) prior the posterior is
Beta(1 + hits, 1 + misses) and P(B > A) is estimated by sampling. For a report run the trials are the gold's
stated facts (recall = hit / stated) and the report's asserted facts (precision = hit / (hit + wrong +
hallucinated)), the benchmark's own definitions. The bootstrap resamples clips with replacement, keeping the
pairing, and reports the 95 % interval of the mean difference (B minus A) in WER or weighted score.
Trials inside one clip are not independent (one collapsed window loses many terms at once), so the posterior
is optimistic about certainty; the clip-level bootstrap is the conservative companion. Both are printed.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np

SET = Path.home() / "datasets" / "synth_v1"
RNG = np.random.default_rng(0)


def rows(path: Path):
    return list(csv.DictReader(open(path), delimiter="\t"))


def p_b_gt_a(ha, ma, hb, mb, n=200000) -> float:
    a = RNG.beta(1 + ha, 1 + ma, n); b = RNG.beta(1 + hb, 1 + mb, n)
    return float((b > a).mean())


def boot_diff(xa, xb, n=10000):
    xa, xb = np.asarray(xa, float), np.asarray(xb, float)
    d = xb - xa
    idx = RNG.integers(0, len(d), (n, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def by_snr(rs):
    out = {}
    for r in rs:
        out.setdefault(r["snr"], {})[r["id"]] = r
    return out


def evals(a: str, b: str) -> str:
    A, B = by_snr(rows(SET / "evals" / a / "rows.tsv")), by_snr(rows(SET / "evals" / b / "rows.tsv"))
    out = [f"### Transcriber: `{b}` against `{a}`", "",
           "| SNR | clips | term recall A | term recall B | P(B > A) | WER A | WER B | mean diff (B - A) | 95 % interval |",
           "|---|---|---|---|---|---|---|---|---|"]
    tot = dict(ha=0, ma=0, hb=0, mb=0)
    for snr in ("snrinf", "snr20", "snr15", "snr10", "snr5"):
        ids = sorted(set(A.get(snr, {})) & set(B.get(snr, {})))
        if not ids:
            continue
        ta = sum(int(A[snr][i]["term_total"]) for i in ids); ma = sum(int(A[snr][i]["term_missing"]) for i in ids)
        tb = sum(int(B[snr][i]["term_total"]) for i in ids); mb = sum(int(B[snr][i]["term_missing"]) for i in ids)
        ha, hb = ta - ma, tb - mb
        for k, v in zip(("ha", "ma", "hb", "mb"), (ha, ma, hb, mb)):
            tot[k] += v
        wa = [float(A[snr][i]["wer"]) for i in ids]; wb = [float(B[snr][i]["wer"]) for i in ids]
        wera = sum(int(A[snr][i]["edits"]) for i in ids) / sum(int(A[snr][i]["ref_words"]) for i in ids)
        werb = sum(int(B[snr][i]["edits"]) for i in ids) / sum(int(B[snr][i]["ref_words"]) for i in ids)
        m, lo, hi = boot_diff(wa, wb)
        out.append(f"| {snr[3:]} | {len(ids)} | {100 * ha / ta:.1f} % ({ha}/{ta}) | {100 * hb / tb:.1f} % ({hb}/{tb}) | "
                   f"{100 * p_b_gt_a(ha, ma, hb, mb):.1f} % | {100 * wera:.1f} % | {100 * werb:.1f} % | {100 * m:+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}] |")
    ha, ma, hb, mb = tot["ha"], tot["ma"], tot["hb"], tot["mb"]
    out.append(f"| all | | {100 * ha / (ha + ma):.1f} % | {100 * hb / (hb + mb):.1f} % | {100 * p_b_gt_a(ha, ma, hb, mb):.1f} % | | | | |")
    return "\n".join(out)


def reports(a: str, b: str) -> str:
    A, B = by_snr(rows(SET / "reports" / a / "rows.tsv")), by_snr(rows(SET / "reports" / b / "rows.tsv"))
    out = [f"### Report stage: `{b}` against `{a}`", "",
           "| SNR | reports | recall A | recall B | P(B > A) | precision A | precision B | P(B > A) | weighted A | weighted B | mean diff | 95 % interval |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for snr in ("snrinf", "snr20", "snr15", "snr10", "snr5"):
        ids = sorted(set(A.get(snr, {})) & set(B.get(snr, {})))
        if not ids:
            continue
        def counts(R):
            h = sum(int(R[snr][i]["hit"]) for i in ids); o = sum(int(R[snr][i]["omission"]) for i in ids)
            w = sum(int(R[snr][i]["wrong"]) for i in ids); x = sum(int(R[snr][i]["hallucination"]) for i in ids)
            return h, o, w, x
        ha, oa, wa, xa = counts(A); hb, ob, wb, xb = counts(B)
        sa = [float(A[snr][i]["weighted"]) for i in ids]; sb = [float(B[snr][i]["weighted"]) for i in ids]
        m, lo, hi = boot_diff(sa, sb)
        out.append(f"| {snr[3:]} | {len(ids)} | {100 * ha / (ha + oa + wa):.1f} % | {100 * hb / (hb + ob + wb):.1f} % | "
                   f"{100 * p_b_gt_a(ha, oa + wa, hb, ob + wb):.1f} % | {100 * ha / (ha + wa + xa):.1f} % | {100 * hb / (hb + wb + xb):.1f} % | "
                   f"{100 * p_b_gt_a(ha, wa + xa, hb, wb + xb):.1f} % | {np.mean(sa):.1f} | {np.mean(sb):.1f} | {m:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--evals", nargs=2, metavar=("A", "B"), action="append", default=[])
    ap.add_argument("--reports", nargs=2, metavar=("A", "B"), action="append", default=[])
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    parts = [evals(a, b) for a, b in args.evals] + [reports(a, b) for a, b in args.reports]
    text = "\n\n".join(parts) + "\n"
    print(text)
    if args.md:
        args.md.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
