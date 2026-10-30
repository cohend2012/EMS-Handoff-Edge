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


def check_dataset(clips, traj_dir: Path, expected_wer: float = 0.09, tol: float = 0.05) -> dict:
    """Invariants that would have caught the two 10-30 mistakes (30 s truncation, NaN entropies). Any failure aborts.

    1. tokens per reference word within [0.8, 3.0] on every clip longer than 30 words (truncation drops it below 0.6);
    2. no NaN or infinity in any feature column;
    3. the dump's mean per-clip WER on the long scripts within `tol` of the trusted scorer's WER for the same model;
    4. bad-token rate between 3 and 25 %.
    """
    import csv
    problems = []
    rows = list(csv.DictReader(open(traj_dir / "summary.tsv"), delimiter="\t"))
    ratios = [(r["id"], r["snr"], int(r["tokens"]) / max(1, int(r["ref_words"]))) for r in rows if int(r["ref_words"]) > 30]
    low = [(i, s, round(x, 2)) for i, s, x in ratios if x < 0.8]; high = [(i, s, round(x, 2)) for i, s, x in ratios if x > 3.0]
    share_low = len(low) / max(1, len(ratios))
    if share_low > 0.30 or high:
        problems.append(f"tokens per reference word out of [0.8, 3.0] on {len(low) + len(high)} of {len(ratios)} long clips (a truncation or runaway signature): {(low + high)[:5]}")
    notes = [f"{len(low)} of {len(ratios)} long clips ({100 * share_low:.0f} %) emitted under 0.8 tokens per reference word: the decoder's own collapses and early stops, kept as data"]
    for c in clips:
        for k in ("hidden", "logprob", "entropy", "attn_entropy", "attn_mass"):
            if not np.isfinite(c[k]).all():
                problems.append(f"{c['file']}: non-finite values in {k}")
                break
    long_wer = [float(r["wer"]) for r in rows if int(r["ref_words"]) > 30]
    mean_wer = float(np.mean(long_wer)) if long_wer else float("nan")
    if not (expected_wer - tol <= mean_wer <= expected_wer + tol):
        problems.append(f"mean WER on long scripts {mean_wer:.3f} is outside {expected_wer:.2f} +/- {tol:.2f}: the dump does not match the trusted scorer")
    bad = sum(int(c["bad"][c["keep"]].sum()) for c in clips); tot = sum(int(c["keep"].sum()) for c in clips)
    rate = bad / max(1, tot)
    if not (0.03 <= rate <= 0.25):
        problems.append(f"bad-token rate {rate:.3f} outside [0.03, 0.25]: labels suspect")
    report = {"clips": len(clips), "tokens": tot, "bad_rate": round(rate, 4), "mean_wer_long_scripts": round(mean_wer, 4),
              "tokens_per_ref_word_min_max": (round(min(x for _, _, x in ratios), 2), round(max(x for _, _, x in ratios), 2)) if ratios else None,
              "problems": problems, "notes": notes}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--alarm", type=float, default=0.05, help="conformal false-alarm rate on correct tokens")
    ap.add_argument("--expected-wer", type=float, default=0.09, help="the trusted scorer's WER for this model on these scripts (guard 3)")
    ap.add_argument("--check-only", action="store_true", help="run the dataset guard and stop")
    ap.add_argument("--force", action="store_true", help="train even if the guard fails (diagnostics only; the results say so)")
    args = ap.parse_args(argv)
    import torch
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clips = load(args.traj)
    guard = check_dataset(clips, args.traj, expected_wer=args.expected_wer)
    print("dataset guard:", json.dumps(guard, indent=1))
    if guard["problems"] and not args.force:
        print("ABORT: the dataset fails its invariants; nothing was trained.")
        return 2
    if args.check_only:
        return 0
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
    res = {"guard": guard, "forced_past_guard": bool(guard["problems"]), "n_scripts": len(sids), "train_scripts": n_train, "test_scripts": len(sids) - n_train,
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
    # --- Idea 1: is there information in the hidden state beyond the model's own uncertainty? Classifiers on x alone and on
    # [x, phi], against the logistic regression on phi. If C(x, phi) - logistic(phi) < 0.02 the hidden state adds nothing and any
    # energy over it is capped at the uncertainty baseline.
    def train_clf(Xtr_, ytr_, Xte_, hidden=64, epochs=15):
        net = torch.nn.Sequential(torch.nn.Linear(Xtr_.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1)).to(device)
        o = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); Xt = torch.tensor(Xtr_, device=device); yt = torch.tensor(ytr_, dtype=torch.float32, device=device)
        pw = torch.tensor(float((1 - ytr_.mean()) / max(ytr_.mean(), 1e-6)), device=device)
        for _ in range(epochs):
            perm = torch.randperm(len(yt), device=device)
            for i in range(0, len(yt), 1024):
                b = perm[i:i + 1024]; o.zero_grad()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[b]).squeeze(1), yt[b], pos_weight=pw); loss.backward(); o.step()
        with torch.no_grad():
            return net(torch.tensor(Xte_, device=device)).squeeze(1).cpu().numpy()
    phi_tr = ((Ftr - fmu) / fsd).astype(np.float32); phi_te = ((Fte - fmu) / fsd).astype(np.float32)
    res["auroc"]["C(x): classifier on the hidden state"] = auroc(train_clf(norm(X1tr).astype(np.float32), Ytr, norm(X1te).astype(np.float32)), Yte)
    res["auroc"]["C(x, phi): hidden state + 4 baselines"] = auroc(train_clf(np.concatenate([norm(X1tr), phi_tr], 1).astype(np.float32), Ytr, np.concatenate([norm(X1te), phi_te], 1).astype(np.float32)), Yte)
    # caution checks on idea 1: a label-permutation control (must fall to about 0.5), a bootstrap interval over test scripts,
    # the per-noise-level AUROC, and a second seed
    sc_cx = train_clf(norm(X1tr).astype(np.float32), Ytr, norm(X1te).astype(np.float32))
    perm_y = rng.permutation(Ytr)
    sc_perm = train_clf(norm(X1tr).astype(np.float32), perm_y, norm(X1te).astype(np.float32))
    torch.manual_seed(args.seed + 1)
    sc_cx2 = train_clf(norm(X1tr).astype(np.float32), Ytr, norm(X1te).astype(np.float32))
    torch.manual_seed(args.seed)
    # script id and snr per test transition, for the bootstrap and the per-level split
    sid_te, snr_te = [], []
    for c in test:
        idx = np.where(c["keep"])[0]; n_ = max(0, len(idx) - 1)
        sid_te += [c["sid"]] * n_; snr_te += [c["file"].split("_")[1][:-4]] * n_
    sid_te = np.array(sid_te); snr_te = np.array(snr_te)
    boots = []
    uniq = np.array(sorted(set(sid_te.tolist())))
    for _ in range(300):
        pick = rng.choice(uniq, len(uniq)); mask = np.concatenate([np.where(sid_te == s)[0] for s in pick])
        if Yte[mask].sum() and (1 - Yte[mask]).sum():
            boots.append(auroc(sc_cx[mask], Yte[mask]))
    res["idea1_checks"] = {"C(x) seed 0": auroc(sc_cx, Yte), "C(x) seed 1": auroc(sc_cx2, Yte),
                           "C(x) with permuted training labels (control, expect about 0.5)": auroc(sc_perm, Yte),
                           "C(x) bootstrap 95 % over test scripts": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                           "C(x) per level": {lvl: auroc(sc_cx[snr_te == lvl], Yte[snr_te == lvl]) for lvl in sorted(set(snr_te.tolist()))},
                           "logistic(phi) per level": {lvl: auroc(lr(Fte_t).squeeze(1).detach().cpu().numpy()[snr_te == lvl], Yte[snr_te == lvl]) for lvl in sorted(set(snr_te.tolist()))}}
    res["idea1"] = {"C(x,phi) minus logistic(phi)": round(res["auroc"]["C(x, phi): hidden state + 4 baselines"] - res["auroc"]["logistic(4 baselines)"], 4),
                    "verdict": "hidden state carries information beyond uncertainty" if res["auroc"]["C(x, phi): hidden state + 4 baselines"] - res["auroc"]["logistic(4 baselines)"] >= 0.02 else "hidden state adds nothing beyond uncertainty (< 0.02)"}

    # --- Idea 2: an energy with a known target. E_t = number of bad tokens among the next 16 kept tokens of the same clip, a
    # potential that falls through a good stretch and rises before a bad one. V_E regresses E from the hidden state (MSE);
    # the detector is dV_E, and Spearman(V_E, E) on the test transitions says how well V_E tracks the target.
    def energy_targets(cs):
        X, E = [], []
        for c in cs:
            idx = np.where(c["keep"])[0]; bad = c["bad"][idx].astype(np.float32)
            for j in range(len(idx)):
                X.append(c["hidden"][idx[j]]); E.append(float(bad[j:j + 16].sum()))
        return np.array(X), np.array(E, dtype=np.float32)
    XE_tr, E_tr = energy_targets(train); XE_te, E_te = energy_targets(test)
    reg = torch.nn.Sequential(torch.nn.Linear(XE_tr.shape[1], 64), torch.nn.Softplus(), torch.nn.Linear(64, 64), torch.nn.Softplus(), torch.nn.Linear(64, 1)).to(device)
    o = torch.optim.Adam(reg.parameters(), lr=1e-3, weight_decay=1e-4); Xt = torch.tensor(norm(XE_tr).astype(np.float32), device=device); Et = torch.tensor(E_tr, device=device)
    for _ in range(20):
        perm = torch.randperm(len(Et), device=device)
        for i in range(0, len(Et), 1024):
            b = perm[i:i + 1024]; o.zero_grad(); loss = torch.nn.functional.mse_loss(reg(Xt[b]).squeeze(1), Et[b]); loss.backward(); o.step()
    with torch.no_grad():
        VE_te = reg(torch.tensor(norm(XE_te).astype(np.float32), device=device)).squeeze(1).cpu().numpy()
        VE0 = reg(torch.tensor(norm(X0te).astype(np.float32), device=device)).squeeze(1).cpu().numpy()
        VE1 = reg(torch.tensor(norm(X1te).astype(np.float32), device=device)).squeeze(1).cpu().numpy()
    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b)); return float(np.corrcoef(ra, rb)[0, 1])
    res["auroc"]["dV_E (energy regressed on bad-tokens-ahead)"] = auroc(VE1 - VE0, Yte)
    res["auroc"]["V_E itself at the next token"] = auroc(VE1, Yte)
    res["idea2"] = {"spearman(V_E, E) on test tokens": round(spearman(VE_te, E_te), 4), "mean E test": float(E_te.mean()),
                    "verdict": "continue" if (VE1 - VE0 >= 0).any() and auroc(VE1 - VE0, Yte) > res["auroc"]["logistic(4 baselines)"] else "dV_E does not beat the four-feature regression"}

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

    # the Lyapunov property itself, measured: on the test transitions, how often V fell into a correct token and how often it
    # failed to fall into a bad one; and the same with the training margin
    good_te, bad_te = dv_te[Yte == 0], dv_te[Yte == 1]
    res["decrease_property"] = {
        "P(dV < 0 | correct)": float((good_te < 0).mean()), "P(dV <= -margin | correct)": float((good_te <= -m).mean()),
        "P(dV >= 0 | bad)": float((bad_te >= 0).mean()), "P(dV >= margin | bad)": float((bad_te >= m).mean()),
        "mean dV | correct": float(good_te.mean()), "mean dV | bad": float(bad_te.mean()),
        "train P(dV < 0 | correct)": float((dv_tr[Ytr == 0] < 0).mean()), "train P(dV >= 0 | bad)": float((dv_tr[Ytr == 1] >= 0).mean())}
    np.savez_compressed(args.out / "dv_test.npz", dv=dv_te, bad=Yte, features=Fte)
    # --- Idea 3: detect the collapse at the level it happens. Label a clip bad when its WER > 0.30 (the window collapses
    # and early stops that lose whole stretches), score it from quantities the decoder gives for free, aggregated over the
    # clip: tokens per audio second, mean and min log-probability, mean and max entropy, mean attention entropy, and the
    # fraction of windows that emitted fewer than one token per second. Leave-scripts-out: the same 70 / 30 split by script.
    import csv as _csv
    summ = {(r["id"], r["snr"]): r for r in _csv.DictReader(open(args.traj / "summary.tsv"), delimiter="\t")}
    def clip_feats(c):
        d = np.load(args.traj / c["file"], allow_pickle=True)
        win = d["window"]; nwin = int(win.max()) + 1 if len(win) else 1
        secs = float(summ[(c["sid"], c["file"].split("_")[1][:-4])]["ref_words"]) / 2.5   # the script's spoken length, about 2.5 words a second
        per_win = [int((win == k).sum()) for k in range(nwin)]
        slow = np.mean([1.0 if toks < max(8.0, min(30.0, secs - 30 * k)) else 0.0 for k, toks in enumerate(per_win)])
        lp = c["logprob"][c["keep"]]; en = c["entropy"][c["keep"]]; ae = np.nan_to_num(c["attn_entropy"][c["keep"]], nan=0.0)
        return [len(win) / max(secs, 1.0), float(lp.mean()), float(lp.min()), float(en.mean()), float(en.max()), float(ae.mean()), float(slow)]
    cf_names = ["tokens/s", "mean logprob", "min logprob", "mean entropy", "max entropy", "mean attn entropy", "slow-window fraction"]
    Ctr = np.array([clip_feats(c) for c in train], dtype=np.float32); Cte = np.array([clip_feats(c) for c in test], dtype=np.float32)
    ytr_c = np.array([1 if float(summ[(c["sid"], c["file"].split("_")[1][:-4])]["wer"]) > 0.30 else 0 for c in train])
    yte_c = np.array([1 if float(summ[(c["sid"], c["file"].split("_")[1][:-4])]["wer"]) > 0.30 else 0 for c in test])
    res["clip_collapse"] = {"test_clips": int(len(yte_c)), "test_bad": int(yte_c.sum()), "auroc": {}}
    for k, nm in enumerate(cf_names):
        a = auroc(Cte[:, k], yte_c); res["clip_collapse"]["auroc"][nm] = (a if a >= 0.5 else 1 - a)
    cmu, csd = Ctr.mean(0), Ctr.std(0) + 1e-6
    lr3 = torch.nn.Linear(len(cf_names), 1).to(device); opt = torch.optim.Adam(lr3.parameters(), lr=0.05)
    Ctr_t = torch.tensor((Ctr - cmu) / csd, device=device); ytr_ct = torch.tensor(ytr_c, dtype=torch.float32, device=device)
    wpos = float((1 - ytr_c.mean()) / max(ytr_c.mean(), 1e-6))
    for _ in range(600):
        opt.zero_grad(); loss = torch.nn.functional.binary_cross_entropy_with_logits(lr3(Ctr_t).squeeze(1), ytr_ct, pos_weight=torch.tensor(wpos, device=device)); loss.backward(); opt.step()
    with torch.no_grad():
        res["clip_collapse"]["auroc"]["logistic(all clip features)"] = auroc(lr3(torch.tensor((Cte - cmu) / csd, device=device)).squeeze(1).cpu().numpy(), yte_c)
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
    cc = res["clip_collapse"]
    md += ["", f"Clip-level collapse detector (label: clip WER > 0.30; {cc['test_bad']} of {cc['test_clips']} test clips):", "", "| clip score | AUROC |", "|---|---|"]
    for k, v_ in cc["auroc"].items():
        md.append(f"| {k} | {v_:.3f} |")
    md += ["", f"Idea 1: {res['idea1']}", f"Idea 1 checks: {json.dumps(res['idea1_checks'])}", f"Idea 2: {res['idea2']}", f"Guard: {'FORCED past a failing guard: ' + '; '.join(res['guard']['problems']) if res['forced_past_guard'] else 'passed'}; {res['guard']['notes'][0]}"]
    dp = res["decrease_property"]
    md += ["", "Decrease property on the test transitions: " + "; ".join(f"{k} = {v_:.3f}" for k, v_ in dp.items()), ""]
    md += [f"Conformal threshold on dV at alpha = {c['alpha']}: {c['threshold_dV']:.4f}; realised false-alarm rate on correct test tokens {100 * c['test_false_alarm_rate']:.1f} %, detection rate on bad test tokens {100 * c['test_detection_rate']:.1f} %.", "",
           f"Stop rule: best single baseline {best_single:.3f}, logistic over baselines {logistic:.3f}, dV {dv:.3f}: **{'continue' if res['stop_rule']['continue'] else 'stop'}**."]
    (args.out / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
