# Plan: a learned stability gauge for the transcriber (build and test)

Written 2026-10-26. The idea under test: put a learned energy function V(x) over Whisper's hidden
states so that trajectories that are converging on real words make V fall, and trajectories that are
drifting (hallucinated phrases, repeated loops, dropped stretches) do not; then use V as a gate and as a
steering signal for medical vocabulary. This note says what is real in that idea, what we can prove, and
how we build and test it on the Jetson with the rig we already have.

## 1. Reality check first

- **The pasted "blueprints" test nothing.** They feed random vectors and hand-written dynamics
  (f = -1.2x for "stable", f = 2.5x for "unstable") into an untrained V. The sign of V-dot there is decided
  by the made-up f, not by any property of speech or of the model. We do not run those.
- **Whisper's decoder is not an autonomous system x-dot = f(x).** It is driven by the audio through
  cross-attention and emits discrete tokens. A "Lyapunov function" over its hidden states is a learned
  scalar with a monotonicity constraint: a confidence or anomaly score shaped so that it decreases along
  good trajectories. That is a legitimate thing to learn (CBF-LLM, 2024, does the barrier version for LLM
  alignment). It is not a proof that the model converges to the right word.
- **What is provable, honestly:** (a) a decrease property of V on a bounded region of a small surrogate
  dynamics, by sampling plus Lipschitz bounds (SMT certification at Whisper's 768-dim states is out of
  reach); (b) a conformal guarantee on the detector built from V: a calibrated threshold with a guaranteed
  false-alarm rate on the test distribution; (c) forward invariance of the report's barrier rules (ETA,
  allergies, code status: null unless spoken). Everything else is an empirical curve.
- **The strong, cheap baselines it must beat:** per-token log-probability and entropy, Whisper's
  compression ratio and no-speech probability (the standard hallucination flags), our own
  transcript_checks (loop collapse, silence gate, words per second, captions disagreement), and
  contextual biasing of the medical vocabulary (hotword boosting / shallow fusion). If V does not beat
  these on detection and the steering does not beat hotword boosting on term error, the idea stops there.
- **Where it can run:** Groq's Whisper exposes no hidden states. The gauge applies to the local model
  (faster-whisper is CTranslate2 and exposes none either; the PyTorch Whisper from Hugging Face does). So
  in the product it is a second opinion on the local path or a detector that triggers a re-decode, not a
  change to the cloud transcript. On the 8 GB Orin Nano with about 4 GB usable, whisper-small in fp16 is
  about 0.5 GB plus activations; V is a few thousand parameters; audio is pre-rendered, so XTTS is never
  loaded at test time.

## 2. What we already have that makes this testable

Gold scripts with exact text (samples.py), two cloned medic voices and a Piper voice, the radio channel
filter with the corpus' real floor at any SNR (`radioize.py`), the over-the-air rig (`air_check.py`),
the word-error scorer with spoken-unit normalisation, and the real call-sets with references and gold
reports as the final check. Missing: volume (three scripts is not a dataset) and a train/test split.

## 3. Build and test, in phases with stop rules

**Phase 0, the generated gold set (week 1, needed for everything after).** Status 2026-10-27: scripts, gold, splits
and the Piper render at five SNRs exist (`~/datasets/synth_v1`, 200 scripts, 1000 clips; gen_scripts.py, render_set.py);
the transcriber baselines over the test split are in SYNTH_V1_2026-10-27.md (WER clean / 5 dB, terms missed at
clean: local medium 4.8 / 7.0 / 10.0 %, small 6.2 / 9.6 / 17.5 %, Groq turbo 4.3 / 6.4 / 10.0 %, Groq large-v3
3.9 / 4.9 / 6.5 %, all with the hint; hint off, tiny.en, local large-v3-turbo, corrector and hotword rows follow).
The deliverable table exists; the break is gentle on Piper audio down to 5 dB and the loss is vocabulary, not
English. Cloned voices wait for a GPU (XTTS on this CPU is 8 min per script).
- `scripts/corpus/gen_scripts.py`: template-based radio reports with sampled patient, complaint, vitals,
  interventions, drugs (a 200-item EMS drug and procedure list), units, hospitals, ETA, priority, and
  the corpus' phrasing habits ("how do you copy", "priority two", "be advised"). Each script writes the
  text, its spoken form, and a gold JSON in the `tests/gold` field format so `benchmark.py` scores it.
  Target 200 scripts.
- Render: XTTS in 3-4 cloned voices (about 8 min per script on the CPU, so render on the laptop or
  overnight), then radioize at SNR infinity, 20, 15, 10, 5 dB with the feed-4390 floor, plus a 4.6 kHz
  brick wall. About 200 x 4 x 5 = 4000 clips, 80 hours of audio; subsample to 1000 for the first pass.
- Split by script and by voice: 70 % train, 30 % test, test voices never seen in train.
- Metrics, per clip and aggregated: WER; medical-term error rate (MTER) over the term list;
  hallucination rate (content words in the transcript absent from the script); loop rate (three or more
  repeated n-grams); dropped-tail rate (last 20 % of script words missing); report field score.
- Baselines on the test split: Groq large-v3-turbo, local whisper-small and base.en, each with and
  without the radio-phrase hint, and with hotword boosting of the term list.
- Deliverable: a table of WER and MTER against SNR per model. This alone answers "where does the
  transcriber break", and it is the dataset the Tuesday enhancer experiment also needs.

**Phase 1, trajectories (week 2).**
- Run Hugging Face whisper-small on the test clips with hooks that dump, per emitted token, the decoder's
  last hidden state, log-probability, entropy, and the cross-attention mass over time; per 20 ms frame,
  the encoder output.
- Align the output to the gold text (the scorer's edit alignment) and label every token: correct,
  substituted, inserted (hallucinated), in a loop, or inside a dropped stretch (the tokens that should
  have been there and were not).
- Deliverable: a trajectory dataset, about 1000 clips x 150 tokens, with labels.

**Phase 2, learn V and test it as a detector (week 3). First stop rule.**
- V is a small input-convex network (two hidden layers of 64) on the 768-dim decoder state. Loss:
  V(x_t+1) - V(x_t) <= -margin on tokens labelled correct; >= 0 on tokens labelled inserted or looped;
  plus the usual positivity shaping. Trained on the train split, on the laptop or the Jetson CPU.
- Test as a detector of the next token being wrong, on the test split: AUROC and precision at a
  fixed false-alarm rate, against log-probability, entropy, compression ratio, and our checks.
- Stop rule: continue only if V's AUROC beats the best single baseline by at least 0.05 and beats a
  logistic regression over all baseline features. Otherwise V is a worse version of what we have.
- The provable piece here: a conformal threshold on V-dot giving a guaranteed false-alarm rate on the
  test distribution, with its short proof written down.

**Phase 3, use it (week 4). Second stop rule.**
- Gate: when V-dot stays >= 0 for k tokens, trigger a fallback: re-decode the segment with the hint and
  term boosting, or substitute the Vosk captions for that stretch (the captions carried the words in
  every dropped-tail case we have seen). Measure WER, MTER, hallucination and dropped-tail rates with
  and without the gate, at each SNR.
- Steer: add a penalty proportional to V-dot of each candidate next state to the logits of the top-k
  tokens, with term-list tokens as basins. Compare against plain hotword boosting at the same
  compute.
- Stop rule: keep only what reduces MTER at 10 dB SNR by at least 25 % relative with no WER regression
  at clean, over the hotword baseline. Then validate on the 22 real call-sets.

**Phase 4, the write-up.** Curves of WER and MTER against SNR and enrollment time; the detector's
ROC and its conformal threshold; the region-of-attraction estimate for V on the surrogate dynamics;
the barrier rules' invariance. Claims limited to what the curves show.

## 4. Effort, risks, and what to decide now

- Effort: about four weeks of part-time work, most of it Phase 0 and the rendering. Phase 0 is worth
  doing regardless of the Lyapunov question.
- Risks: CTranslate2 and Groq expose no states, so the product impact is on the local path only; the
  Jetson GPU needs NVIDIA's PyTorch wheel for Whisper hooks to be fast (the CPU works, slowly);
  contextual biasing may already take most of the available gain, in which case the honest result is
  "the baseline was enough".
- Decide now: the size of the first gold set (200 scripts, 1000 clips), the term list source, and
  whether rendering runs on the laptop overnight or on the Jetson over several days.
