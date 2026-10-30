# Lyapunov stability gauge: Phase 1 and Phase 2 results (2026-10-30)

**Status 14:30: superseded in part by LYAPUNOV_MATH section 8: the hidden state does carry information beyond uncertainty (classifier AUROC 0.915 against 0.826), the Lyapunov formulation was what failed; a clip-level collapse detector from mean log-probability reaches 0.942.** Earlier note: A dataset guard added after these numbers were written finds the Phase 1 dump failing two
invariants (19 clips with far fewer tokens than reference words, down to 0.27 per word; long-script WER 0.173 against a
trusted band near 0.09). The numbers below are therefore from a dump that still loses output on some clips, and the
verdict is not final until a dump passes the guard. See LYAPUNOV_MATH_2026-10-30.md section 7.

Plan: `docs/plans/LYAPUNOV_STT_PLAN.md`. Scripts: `scripts/corpus/trajectories.py` (Phase 1), `scripts/corpus/stability_gauge.py`
(Phase 2). Hardware: the PyTorch 2.7 container on cohendynamics (GPU); the whisper server was stopped for the dump.

## Phase 1: the trajectory dataset

whisper-small (Hugging Face, PyTorch), no prompt, over the 140 training scripts at clean and 5 dB, decoded in 30 s windows:
280 clips, 34,120 emitted tokens. Per token: the decoder's last hidden state (768), log-probability, entropy (with
suppressed-token logits counted as zero probability), cross-attention peak frame, entropy and windowed mass, the window
index, and a label from the word alignment against the script: 30,448 correct, 3,018 substituted, 649 inserted, 5 in
loops; 2,527 reference words deleted (no token to label). Mean per-clip WER 0.16 / 0.17, inflated by the 12-word
fragment scripts and the unprompted decode. A first run truncated every clip to 30 s (the feature extractor's default)
and was discarded; a second had NaN entropies (0 x -inf) and was rerun.

## Phase 2: V as a detector, against the baselines

# Stability gauge, Phase 2 results

Scripts: 140 (98 train / 42 test, split by script). Test transitions: 9730, bad 1028 (10.6 %).

| detector | AUROC on the test scripts |
|---|---|
| -logprob | 0.787 |
| entropy | 0.807 |
| attn_entropy | 0.677 |
| -attn_mass | 0.512 |
| logistic(4 baselines) | 0.826 |
| dV (learned V) | 0.744 |
| logistic(4 baselines + dV) | 0.834 |

Conformal threshold on dV at alpha = 0.05: 0.1594; realised false-alarm rate on correct test tokens 6.7 %, detection rate on bad test tokens 32.0 %.

Stop rule: best single baseline 0.807, logistic over baselines 0.826, dV 0.744: **stop**.

AUROC is the probability that a random wrong token scores above a random correct one. Split by script (98 train, 42 test),
so no script's tokens appear on both sides. "entropy" and "attn_entropy" are reported with the sign that makes them
detectors (attention entropy works inverted: the decoder attends more sharply when it is about to be wrong).

## Verdict

The plan's stop rule was: continue only if V's step dV beats the best single baseline by at least 0.05 AUROC and beats a
logistic regression over the baselines. dV scored 0.744; the model's own token entropy alone scored 0.807, the four
baselines together 0.826, and adding dV to them gained 0.008 (0.834). **The learned energy function is a worse detector
than the model's own uncertainty, so the Lyapunov line stops here as the plan said it would.** What the exercise bought:

- a per-token wrong-word detector at AUROC 0.826 from quantities the decoder already produces (log-probability, entropy,
  attention statistics), with a conformal threshold recipe; this is the confidence signal the patient-state layer was
  waiting for, and it needs no new model;
- the finding that faster-whisper, which the product uses, exposes none of these per token; it exposes a per-segment
  average log-probability and no-speech probability, so the product-side confidence will be those two, calibrated on this
  dataset's segment-level labels;
- the Phase 1 dataset itself (34,120 labelled tokens on cohendynamics, `trajectories/small_train`), reusable for any
  later detector.

What was not tried and could reopen the question: a larger V, V on the concatenation of hidden state and the baseline
features, whisper-medium or large-v3 trajectories, or the real bursts (which need human-verified references first).
None of these is scheduled; the writer's speed and the human verification of the real-call references rank ahead.
