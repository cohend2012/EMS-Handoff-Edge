# The stability gauge: the mathematics, the measured properties, and what "works" means (2026-10-30)

Companion to `LYAPUNOV_PHASE2_2026-10-30.md`. Everything here is computed by `scripts/corpus/stability_gauge.py` on the
Phase 1 trajectories; the numbers are from the 13:40 run.

## 1. Objects

For one clip, the decoder emits tokens t = 1..T. At each step we recorded

- x_t in R^768: the last decoder layer's hidden state at the position that produced token t;
- l_t = log p(token_t | audio, tokens_<t): the log-probability the model assigned to the token it chose;
- H_t = - sum_v p_t(v) log p_t(v): the entropy of the model's next-token distribution at step t (suppressed tokens have
  p = 0 and contribute 0);
- a_t in R^1500: the last layer's cross-attention over the 30 s window's encoder frames, averaged over heads and
  normalised; its entropy H(a_t), its peak frame, and the mass within 25 frames (0.5 s) of the peak;
- y_t in {0, 1}: 1 when token t belongs to a word labelled substituted, inserted or in a loop by the alignment against the
  script, 0 when it belongs to a correct word. Punctuation and special tokens are dropped.

A transition is (x_t, x_t+1, y_t+1). Test set: 9,730 transitions from 42 scripts the training never saw, 1,028 of them
bad (10.6 %).

## 2. The energy function and its training objective

V_theta : R^768 -> R_>=0, V(x) = softplus(f_theta(z)), z = (x - mu) / sigma standardised per dimension on the training
transitions, f_theta a two-hidden-layer network (768 -> 64 -> 64 -> 1, softplus activations; 53,000 parameters).

The Lyapunov-style requirement, with margin m = 0.1:

    into a correct token:   dV_t = V(x_t+1) - V(x_t) <= -m
    into a bad token:       dV_t >= +m

Hinge loss over the training transitions, with the bad class up-weighted by w = (1 - r) / r where r is the bad rate
(so both classes count equally), plus a small positivity regulariser:

    L(theta) = mean_t [ (1 - y_t+1) * max(0, m + dV_t)  +  y_t+1 * w * max(0, m - dV_t) ]  +  1e-3 * mean_t V(x_t)^2

Adam, lr 1e-3, 30 epochs, batches of 1024. This is the "learned scalar with a monotonicity constraint" the plan
described; it is not a certificate of anything. A true Lyapunov function would need dV < 0 on every correct step of a
closed dynamical system, and the decoder is not one: it is driven by the audio and emits discrete tokens.

## 3. The detector score and how it is judged

Score s_t = dV_t: a step that does not lower the energy is suspect. For a baseline, the score is the raw quantity
(-l_t+1, H_t+1, H(a_t+1) with its sign flipped where that makes it a detector, -mass_t+1), or a logistic regression
sigma(w . phi + b) fitted on the training transitions to predict y_t+1 from phi = the four baselines (and, in the
combined row, dV).

AUROC(s) = P(s_bad > s_correct) for a random bad and a random correct test transition, computed as the Mann-Whitney
rank statistic with ties averaged. 0.5 is a coin; 1.0 is perfect separation.

Conformal threshold: q = the (1 - alpha) quantile of dV over the training split's correct transitions (alpha = 0.05).
Flag when dV > q. For exchangeable tokens P(flag | correct) <= alpha + 1/(n_cal + 1); tokens inside one clip are not
exchangeable, so the realised rate is what counts.

## 4. The measured properties (test transitions, 42 held-out scripts)

| property | value | reading |
|---|---|---|
| P(dV < 0 given correct) | 0.594 | V fell on 59 % of correct steps: barely better than a coin |
| P(dV <= -m given correct) | 0.180 | the trained margin is met on 18 % of correct steps |
| P(dV >= 0 given bad) | 0.779 | V failed to fall on 78 % of bad steps (training split: 97.5 %) |
| P(dV >= m given bad) | 0.509 | |
| mean dV given correct / given bad | -0.015 / +0.098 | the means separate, the distributions overlap heavily |
| AUROC(dV) | 0.744 | |
| AUROC(H_t), entropy | 0.807 | best single free quantity |
| AUROC(-l_t), log-probability | 0.787 | |
| AUROC(H(a_t)) inverted | 0.677 | sharper attention when about to be wrong |
| AUROC(logistic over the four) | 0.826 | |
| AUROC(logistic over the four + dV) | 0.834 | dV adds 0.008 |
| conformal threshold q (alpha 0.05) | 0.159 | realised false-alarm 6.7 %, detection 32.0 % |

The decrease property does not hold out of sample: a decision rule "dV >= 0 means suspect" would detect 78 % of bad
tokens while flagging 41 % of correct ones. The gap between the training split (bad steps not decreasing 97.5 % of the
time) and the test split (77.9 %) says the network fit the training scripts' bad steps rather than a property of the
decoder.

## 5. So, does it work?

- **As a Lyapunov function: no.** V does not reliably decrease along correct trajectories (59 %), and the margin it was
  trained to is met on 18 % of them. There is no decrease property to certify.
- **As a wrong-token detector: it is a detector, but a worse one than the model's own entropy** (0.744 against 0.807),
  and it adds 0.008 to a regression that already has the free quantities. The plan's stop rule (beat the best baseline by
  0.05 and beat the regression) fires.
- **What does work:** the model's next-token entropy and log-probability, and a four-feature logistic regression over
  them and the attention statistics, detect a wrong token at AUROC 0.826 on scripts never trained on, for free. That is
  the confidence signal to carry.

## 6. Reproduce

    # on cohendynamics, PyTorch container ems-torch:hf; the whisper server must be stopped for the dump (memory)
    python3 scripts/corpus/trajectories.py --set /data --split train --snr inf 5 --model openai/whisper-small --out /data/trajectories/small_train
    python3 scripts/corpus/stability_gauge.py --traj /data/trajectories/small_train --out /data/trajectories/gauge_small
    # results.json, results.md, dv_test.npz (per-transition dV, label and the four baseline features)
