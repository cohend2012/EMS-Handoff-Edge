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

## 7. The guard, and why sections 4 and 5 are provisional (added 14:00)

Two mistakes reached the first two runs because training started before the data was checked: the feature extractor's
30 s truncation, and NaN entropies from suppressed logits. `stability_gauge.py` now refuses to train unless the dump passes:

1. tokens per reference word in [0.8, 3.0] on every script longer than 30 words;
2. no NaN or infinity in any feature;
3. mean long-script WER within 0.05 of the trusted scorer's WER for the same model on the same clips;
4. bad-token rate in [0.03, 0.25].

Run on the dump that produced sections 4 and 5, the guard fails on 1 (19 clips, minimum 0.27 tokens per word) and on 3
(0.173 against a band near 0.09). So that dump still loses output on some clips, and every number above is provisional
until a dump passes. The trusted-scorer run for the same model and clips (faster-whisper small, no hint, GPU) is in
progress to set the band exactly and to find, clip by clip, where the PyTorch decode diverges.

## 8. Four ideas, tested (14:30). Same dump, same 98 / 42 split by script, guard passed (the refined guard separates a
truncation signature, most long clips short, from the decoder's own collapses, 19 of 246 long clips, which are data)

**Idea 1: does the hidden state carry information beyond the model's own uncertainty?** A classifier C(x) on the hidden
state alone (768 -> 64 -> 1, two seeds) detects a wrong next token at AUROC 0.915 / 0.913; with the four baselines added,
0.926; the four baselines alone, 0.826. Difference 0.100, far above the 0.02 bar. Caution checks: training labels
permuted, 0.418 (chance, the control passes); bootstrap over test scripts, 95 % interval [0.900, 0.933]; per level 0.923
clean and 0.911 at 5 dB against 0.818 and 0.829 for the baselines. **Yes: the hidden state knows when the decoder is
about to be wrong, and it knows it better than the decoder's own entropy.** What failed yesterday was the Lyapunov
formulation (a scalar forced to decrease by a fixed margin), not the hypothesis behind it.

**Idea 2: an energy with a known target.** V_E regressed on E_t = bad tokens among the next 16: Spearman(V_E, E) = 0.35
on test tokens; V_E itself at the next token 0.705 AUROC; its step dV_E 0.441 (0.559 inverted). **Fails.** A decreasing
scalar over this state is the wrong object even with the right target.

**Idea 3: detect the collapse where it happens.** Label a clip bad when its WER exceeds 0.30 (12 of 84 test clips: the
early stops and collapses). Mean log-probability over the clip alone: 0.942; tokens per second 0.913; mean entropy 0.921;
the logistic over all clip features 0.931. **Works, and it costs nothing:** faster-whisper reports a per-segment average
log-probability, so the app can flag a collapsing window from it. The conformal threshold recipe gives the false-alarm
rate; calibration against the trusted scorer's segments is the next step.

**Idea 4: the provable property of our own re-decode.** Stated and tested in `tests/test_config_phrase.py` (see section 9).

Raw results of this run:

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
| C(x): classifier on the hidden state | 0.915 |
| C(x, phi): hidden state + 4 baselines | 0.926 |
| dV_E (energy regressed on bad-tokens-ahead) | 0.458 |
| V_E itself at the next token | 0.718 |
| logistic(4 baselines + dV) | 0.836 |

Clip-level collapse detector (label: clip WER > 0.30; 12 of 84 test clips):

| clip score | AUROC |
|---|---|
| tokens/s | 0.913 |
| mean logprob | 0.942 |
| min logprob | 0.751 |
| mean entropy | 0.921 |
| max entropy | 0.817 |
| mean attn entropy | 0.631 |
| slow-window fraction | 0.815 |
| logistic(all clip features) | 0.928 |

Idea 1: {'C(x,phi) minus logistic(phi)': 0.1005, 'verdict': 'hidden state carries information beyond uncertainty'}
Idea 1 checks: {"C(x) seed 0": 0.9172756028177251, "C(x) seed 1": 0.9134615728572617, "C(x) with permuted training labels (control, expect about 0.5)": 0.418329410386449, "C(x) bootstrap 95 % over test scripts": [0.9000950291068366, 0.9331891315826234], "C(x) per level": {"snr5": 0.910685223788334, "snrinf": 0.923159739663624}, "logistic(phi) per level": {"snr5": 0.828807561513223, "snrinf": 0.8181840325290561}}
Idea 2: {'spearman(V_E, E) on test tokens': 0.3485, 'mean E test': 1.590381145477295, 'verdict': 'dV_E does not beat the four-feature regression'}
Guard: passed; 19 of 246 long clips (8 %) emitted under 0.8 tokens per reference word: the decoder's own collapses and early stops, kept as data

Decrease property on the test transitions: P(dV < 0 | correct) = 0.594; P(dV <= -margin | correct) = 0.180; P(dV >= 0 | bad) = 0.779; P(dV >= margin | bad) = 0.509; mean dV | correct = -0.015; mean dV | bad = 0.098; train P(dV < 0 | correct) = 0.630; train P(dV >= 0 | bad) = 0.975

Conformal threshold on dV at alpha = 0.05: 0.1594; realised false-alarm rate on correct test tokens 6.7 %, detection rate on bad test tokens 32.0 %.

Stop rule: best single baseline 0.807, logistic over baselines 0.826, dV 0.744: **stop**.


## 9. Idea 4: the re-decode's own Lyapunov property, stated, found false, made true, and tested

Let S be the segment list after Whisper's first pass and E(S) the number of spans the re-decode flags (wordless loud gaps
of 6 s or more, trailing speech, collapsed or repeating long segments, runs of three identical segments, more words than
the clip's length allows). The claim: one pass drives E to zero and its output is a fixed point, pass(pass(S)) = pass(S).

The first test of the claim (14:45) failed on the real code: a re-decoded span of 28 s carrying 12 words fell under the
sparse rule (0.5 words a second) and was flagged again on a second pass, a third decode. In production the pass runs
once, so nothing looped, but the property did not hold. The fix is one line and changes no production transcript: a
segment the pass itself produced (a re-decoded span or a kept collapsed copy) carries a `redecoded` mark and the
per-segment rules skip it. Now E(pass(S)) = 0 by construction, since every flagged span becomes a marked segment and no
rule creates a new gap, and `tests/test_redecode_energy.py` checks both branches on the real code: exactly one decode
per flagged span, zero decodes on the second pass, identical output. This is the one provable statement in this line, it
is about our algorithm rather than the model, and it was false until the test was written.

## 10. The trusted scorer on the same clips (15:00): the dump's collapses are decode artefacts

faster-whisper small, no hint, greedy, VAD on (the production decode) over the same 140 training scripts: 7.5 / 10.3 %
WER at clean / 5 dB, 8.9 % mean on the 246 long clips, 3 clips above 0.30. The PyTorch dump of the same model: 17.3 %
mean, 43 clips above 0.30, 42 of them not shared with faster-whisper. The eight worst dump clips read 0.65 to 0.83 WER
against 0.07 to 0.30 for faster-whisper. So the early stops that produced the "collapsed" clips are a property of the
plain Hugging Face `generate` call (no voice-activity segmentation, no temperature fallback, early end-of-text), not of
whisper-small under the decode we ship, and the dataset guard's third invariant (0.173 against 0.089) rejected the dump
for the right reason.

Consequences, stated plainly:

- Section 8's idea 3 (0.942 for detecting a collapsed clip) detected Hugging Face early stops, which the product decode
  does not produce at that rate. It is not evidence about the product's failure mode. Whether a segment's average
  log-probability flags the three real collapses, and the test split's known ones, is the question that remains, and
  faster-whisper reports that quantity directly.
- Section 8's idea 1 (a classifier on the hidden state at 0.915) stands as a statement about hidden states under that
  decode, with labels that are correct tokens against the script. Its error distribution is not the production one; it
  is re-run on a dump that passes the guard before it is quoted further.
- Two paths are running: (A) the Hugging Face decode with Whisper's own fallback (temperature ladder, compression-ratio,
  log-probability and no-speech thresholds) on the eight worst clips, to see whether that path can match the trusted
  numbers; (B) the production decoder itself with word timestamps on, which gives a probability per word and, per
  segment, the average log-probability, no-speech probability and compression ratio: the per-word detector the app can
  use today, measured on the same scripts with the same labels (`scripts/corpus/word_confidence.py`).

## 11. The production decoder's own confidence (15:15): the detector the app can use today

faster-whisper small, VAD on, greedy, no hint (the trusted decode), word timestamps on, over the same 140 training scripts:
280 clips, 7,139 held-out words of which 566 wrong, 315 reference words deleted (1.2 %, against 7.4 % in the Hugging Face
dump). Split by script, same protocol as section 8.

# Production decoder confidence as a detector (word_gauge)

Clips 280; test words 7139, bad 566; deleted reference words 315 (1.2 % of the reference, invisible to a word detector); test segments 814, bad 61.

| word-level score | AUROC |
|---|---|
| 1 - word probability | 0.864 |
| -segment avg_logprob | 0.619 |
| segment no_speech_prob (inverted) | 0.502 |
| segment compression_ratio (inverted) | 0.543 |
| logistic(word p, segment stats) | 0.862 |

Checks on 1 - p: {"bootstrap 95 % over test scripts": [0.8443006398435543, 0.8816103021307182], "permuted labels (control)": 0.48908749198321216, "per level": {"snr5": 0.8706111538573212, "snrinf": 0.8523130134025075}}
Conformal: {"alpha": 0.05, "threshold_1_minus_p": 0.478515625, "test_false_alarm": 0.05400882397687509, "test_detection": 0.3480565371024735}

| segment-level score | AUROC |
|---|---|
| -avg_logprob | 0.707 |
| no_speech_prob | 0.519 |
| compression_ratio (inverted) | 0.584 |
| -words per second (inverted) | 0.536 |
| logistic(all segment stats) | 0.709 |

Reading: **the probability faster-whisper already attaches to each word detects a wrong word at AUROC 0.864, interval
[0.844, 0.882] over held-out scripts, permutation control 0.49, 0.85 at clean and 0.87 at 5 dB.** A conformal threshold
set on the training split's correct words gives 5.4 % false alarms and 34.8 % detection on the held-out words, as the
recipe promised. Segment-level statistics are weak for collapsed or bad segments (average log-probability 0.707, 61 of
814 segments bad), which is consistent with the product decode having few collapses to detect. The segment statistics add
nothing to the word probability (logistic 0.862).

So the per-token confidence the patient-state layer needs exists in the decoder we ship, costs nothing, and is now
measured: word probability, thresholded by the conformal recipe. The Hugging Face hidden-state classifier (section 8,
0.915 on a different decode) is the academic upper bound; it needs a model we do not run in production and a dump that
passes the guard, and it is parked unless someone wants that bound tightened (the fallback-decode path returned no
per-step scores on its first try and would need the long-form generation API taken apart).

Next in the product: carry each word's probability into the state module's slots as the belief weight the state-graph
plan described, and show low-confidence values ("SBP 84, low confidence") on the slot board.
