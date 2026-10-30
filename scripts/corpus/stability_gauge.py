#!/usr/bin/env python3
"""
stability_gauge.py — Phase 2 of docs/plans/LYAPUNOV_STT_PLAN.md: learn an energy function V over the decoder's
hidden states from the Phase 1 trajectories, and test V's decrease (V_t+1 - V_t) as a detector of a wrong next
token against the cheap baselines the plan names. The plan's stop rule is applied by this script.

    python3 scripts/corpus/stability_gauge.py --traj /data/trajectories/small_train --out /data/trajectories/gauge_small

Data: every token of every clip, with its 768-dim hidden state, log-probability, entropy, cross-attention entropy
and windowed mass, and its label (correct / substituted / inserted / loop / punct / special). A token is "bad" when
its label is substituted, inserted or loop; punctuation and special tokens are dropped. The split is by script
(70 % of the scripts train, 30 % test), never by token, so no script's tokens appear on both sides.

Baselines (per token, no training): -logprob, entropy, attention entropy, -attention mass; plus a logistic
regression over those four (the strongest cheap detector). V: a small network on the hidden state, trained with
the Lyapunov-style loss  max(0, margin + V(x_t+1) - V(x_t)) on transitions into a correct token (V must fall),
max(0, margin - (V(x_t+1) - V(x_t))) on transitions into a bad token (V must not fall), plus a small positivity
term. The detector score is dV = V(x_t+1) - V(x_t): a step that does not lower the energy is suspect.

Also computed: a conformal threshold on dV from the training split's correct tokens at a chosen false-alarm
rate, and the realised false-alarm and detection rates on the test split (the plan's provable piece).

Output: <out>/results.json and results.md with AUROC per detector, the stop-rule verdict, and the conformal
numbers. Runs on CPU or GPU; the network is a few thousand parameters and trains in a minute.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

BAD = {"substituted", "inserted", "loop"}
GOOD = {"correct"}


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the ROC: probability a random bad token scores higher than a random good one (rank statistic)."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    # average ranks for ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load(traj_dir: Path):
    clips = []
    for p in sorted(glob.glob(str(traj_dir / "*.npz"))):
        d = np.load(p, allow_pickle=True)
        sid = os.path.basename(p).split("_")[0]
        lab = np.array([str(x) for x in d["label"]])
        keep = np.array([l in BAD or l in GOOD for l in lab])
        if keep.sum() < 2:
            continue
        clips.append({"sid": sid, "file": os.path.basename(p), "hidden": d["hidden"].astype(np.float32),
                      "logprob": d["logprob"].astype(np.float32), "entropy": d["entropy"].astype(np.float32),
                      "attn_entropy": d["attn_entropy"].astype(np.float32), "attn_mass": d["attn_mass_window"].astype(np.float32),
                      "bad": np.array([l in BAD for l in lab]), "keep": keep})
    return clips


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--alarm", type=float, default=0.05, help="conformal false-alarm rate on correct tokens")
    args = ap.parse_args(argv)
    import torch
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clips = load(args.traj)
    sids = sorted({c["sid"] for c in clips}); rng.shuffle(sids)
    n_train = int(0.7 * len(sids)); train_ids = set(sids[:n_train])
    train = [c for c in clips if c["sid"] in train_ids]; test = [c for c in clips if c["sid"] not in train_ids]

    def transitions(cs):
        """Pairs (x_t, x_t+1, bad_t+1) over consecutive kept tokens of each clip, plus the baseline features of x_t+1."""
        X0, X1, Y, F = [], [], [], []
        for c in cs:
            idx = np.where(c["keep"])[0]
            for a, b in zip(idx[:-1], idx[1:]):
                X0.append(c["hidden"][a]); X1.append(c["hidden"][b]); Y.append(int(c["bad"][b]))
                F.append([-c["logprob"][b], c["entropy"][b], c["attn_entropy"][b], -c["attn_mass"][b]])
        F = np.nan_to_num(np.array(F, dtype=np.float32), nan=0.0, posinf=50.0, neginf=-50.0)   # no NaN reaches the regressions
        return np.nan_to_num(np.array(X0)), np.nan_to_num(np.array(X1)), np.array(Y), F
    X0tr, X1tr, Ytr, Ftr = transitions(train); X0te, X1te, Yte, Fte = transitions(test)
    mu, sd = X1tr.mean(0), X1tr.std(0) + 1e-6
    norm = lambda X: (X - mu) / sd

    # --- baselines
    res = {"n_scripts": len(sids), "train_scripts": n_train, "test_scripts": len(sids) - n_train,
           "train_transitions": int(len(Ytr)), "test_transitions": int(len(Yte)), "test_bad": int(Yte.sum()),
           "test_bad_rate": float(Yte.mean()), "auroc": {}}
    names = ["-logprob", "entropy", "attn_entropy", "-attn_mass"]
    for k, nm in enumerate(names):
        a = auroc(Fte[:, k], Yte)
        res["auroc"][nm] = a if a >= 0.5 else 1 - a            # a feature that works inverted is still a detector; the sign is noted
        if a < 0.5:
            res.setdefault("inverted", []).append(nm)
    # logistic regression over the four baseline features (torch, so no sklearn dependency)
    fmu, fsd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Ftr_t = torch.tensor((Ftr - fmu) / fsd, device=device); Fte_t = torch.tensor((Fte - fmu) / fsd, device=device)
    Ytr_t = torch.tensor(Ytr, dtype=torch.float32, device=device)
    lr = torch.nn.Linear(4, 1).to(device); opt = torch.optim.Adam(lr.parameters(), lr=0.05)
    w_pos = float((1 - Ytr.mean()) / max(Ytr.mean(), 1e-6))
    for _ in range(400):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(lr(Ftr_t).squeeze(1), Ytr_t, pos_weight=torch.tensor(w_pos, device=device)); loss.backward(); opt.step()
    with torch.no_grad():
        res["auroc"]["logistic(4 baselines)"] = auroc(lr(Fte_t).squeeze(1).cpu().numpy(), Yte)

    # --- V on the hidden state
    class V(torch.nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(d, 64), torch.nn.Softplus(), torch.nn.Linear(64, 64), torch.nn.Softplus(), torch.nn.Linear(64, 1))
        def forward(self, x):
            return torch.nn.functional.softplus(self.net(x)).squeeze(1)      # V >= 0
    v = V(X0tr.shape[1]).to(device); opt = torch.optim.Adam(v.parameters(), lr=1e-3)
    x0 = torch.tensor(norm(X0tr), device=device); x1 = torch.tensor(norm(X1tr), device=device); y = torch.tensor(Ytr, dtype=torch.float32, device=device)
    m = args.margin; bs = 1024; t0 = time.perf_counter()
    for ep in range(args.epochs):
        perm = torch.randperm(len(y), device=device); tot = 0.0
        for i in range(0, len(y), bs):
            b = perm[i:i + bs]
            dv = v(x1[b]) - v(x0[b])
            good = torch.relu(m + dv) * (1 - y[b])                    # into a correct token: V must drop by the margin
            bad = torch.relu(m - dv) * y[b] * w_pos                   # into a bad token: V must not drop (weighted, bad tokens are rare)
            loss = (good + bad).mean() + 1e-3 * (v(x0[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()) * len(b)
    res["v_train_seconds"] = round(time.perf_counter() - t0, 1); res["v_last_loss"] = round(tot / len(y), 4)
    with torch.no_grad():
        dv_te = (v(torch.tensor(norm(X1te), device=device)) - v(torch.tensor(norm(X0te), device=device))).cpu().numpy()
        dv_tr = (v(x1) - v(x0)).cpu().numpy()
    res["auroc"]["dV (learned V)"] = auroc(dv_te, Yte)
    # V plus the baselines in one logistic regression: does V add information?
    Gtr = np.concatenate([(Ftr - fmu) / fsd, dv_tr[:, None]], 1).astype(np.float32); Gte = np.concatenate([(Fte - fmu) / fsd, dv_te[:, None]], 1).astype(np.float32)
    lr2 = torch.nn.Linear(5, 1).to(device); opt = torch.optim.Adam(lr2.parameters(), lr=0.05)
    Gtr_t = torch.tensor(Gtr, device=device)
    for _ in range(400):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(lr2(Gtr_t).squeeze(1), Ytr_t, pos_weight=torch.tensor(w_pos, device=device)); loss.backward(); opt.step()
    with torch.no_grad():
        res["auroc"]["logistic(4 baselines + dV)"] = auroc(lr2(torch.tensor(Gte, device=device)).squeeze(1).cpu().numpy(), Yte)

    # --- conformal threshold on dV: alarm when dV exceeds the (1 - alpha) quantile of the training split's correct transitions
    q = float(np.quantile(dv_tr[Ytr == 0], 1 - args.alarm))
    res["conformal"] = {"alpha": args.alarm, "threshold_dV": q,
                        "test_false_alarm_rate": float((dv_te[Yte == 0] > q).mean()),
                        "test_detection_rate": float((dv_te[Yte == 1] > q).mean()),
                        "guarantee": "P(alarm | correct token) <= alpha + 1/(n_cal+1) on exchangeable test tokens; tokens within a clip are not independent, so read the realised rate"}

    best_single = max(v_ for k, v_ in res["auroc"].items() if k in names)
    logistic = res["auroc"]["logistic(4 baselines)"]; dv = res["auroc"]["dV (learned V)"]
    res["stop_rule"] = {"best_single_baseline": best_single, "logistic_baselines": logistic, "dV": dv,
                        "continue": bool(dv >= best_single + 0.05 and dv > logistic),
                        "rule": "continue only if dV beats the best single baseline by 0.05 AUROC and beats the logistic regression over the baselines"}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(json.dumps(res, indent=1))
    md = ["# Stability gauge, Phase 2 results", "",
          f"Scripts: {res['n_scripts']} ({res['train_scripts']} train / {res['test_scripts']} test, split by script). Test transitions: {res['test_transitions']}, bad {res['test_bad']} ({100 * res['test_bad_rate']:.1f} %).", "",
          "| detector | AUROC on the test scripts |", "|---|---|"]
    for k, v_ in res["auroc"].items():
        md.append(f"| {k} | {v_:.3f} |")
    c = res["conformal"]
    md += ["", f"Conformal threshold on dV at alpha = {c['alpha']}: {c['threshold_dV']:.4f}; realised false-alarm rate on correct test tokens {100 * c['test_false_alarm_rate']:.1f} %, detection rate on bad test tokens {100 * c['test_detection_rate']:.1f} %.", "",
           f"Stop rule: best single baseline {best_single:.3f}, logistic over baselines {logistic:.3f}, dV {dv:.3f}: **{'continue' if res['stop_rule']['continue'] else 'stop'}**."]
    (args.out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
