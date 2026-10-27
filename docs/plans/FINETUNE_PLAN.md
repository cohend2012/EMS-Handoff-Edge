# Plan: fine-tuning on the generated data to prove the architecture

Written 2026-10-26. Companion to LYAPUNOV_STT_PLAN.md (which supplies the generated gold set, Phase 0).
This plan says how that data is used to fine-tune the two learned stages of HANDOFF.AI, and how the
results are turned into evidence about the architecture rather than about one model.

## 1. What "prove the architecture" means here

The architecture's claims, each of which becomes a hypothesis with a measurement:

| claim | hypothesis | measurement |
|---|---|---|
| Stages are separable: captions, final transcript, extraction, rules, each behind a fixed contract | H1: fine-tuning one stage improves the end-to-end score without changing or retraining any other stage | end-to-end field score before and after, with every other stage frozen; the API and JSON contracts unchanged (the test suite stays green) |
| The report never asserts what was not said | H2: hallucination rate stays near zero at every noise level because of the barrier rules, even when the models get worse | insertions per report with the rules on and off, across SNR |
| Captions are a safety net for the final transcript | H3: substituting captions for dropped stretches lowers the missing-field rate | dropped-tail rate and missing-field rate with the fallback on and off |
| The device can run offline at near-cloud quality | H4: a fine-tuned local transcriber and report model close most of the gap to Groq on radio audio at matched SNR | WER, term error and field score, local versus Groq, per SNR |
| Abstain rather than guess | H5: calibrated "unknown" answers reach a target coverage with a known error rate | conformal calibration on the validation split, coverage on the test split |
| It fits the Orin Nano | H6: memory and report latency stay inside budget after fine-tuning | peak RSS, GPU memory, report time per clip on the device |

The proof is the set of curves and tables that answer these, produced by the replay harness on held-out
audio, with confidence intervals. Nothing is claimed that a curve does not show.

## 2. Data and splits

- **Generated set** (Phase 0 of the companion plan; exists since 2026-10-27 as `~/datasets/synth_v1`, Piper voice only
  so far, splits.json 140/30/30 by script): about 200 scripts, each with text, spoken form and a
  gold JSON in the `tests/gold` field format; rendered in the cloned voices and radio-filtered at SNR
  infinity, 20, 15, 10 and 5 dB with the corpus floor. First pass about 1000 clips.
- **Training pairs come from the pipeline itself.** For the transcriber: (clip, script text). For the
  report model: (the transcript the *baseline* transcriber actually produced for that clip, gold JSON),
  so the extractor learns to work from real transcription errors and to leave null what the transcript
  does not say. Also a share of (clean script text, gold JSON) pairs for the pasted-text path.
- **Splits:** 70 / 15 / 15 train / validation / test, split by script and by voice; one voice appears
  only in test. The 22 real call-sets are the final test only and never enter training. The clones made
  from CS008 and CS014 are in the generated set, so those two call-sets are reported separately from
  the other twenty.
- **Forgetting control:** 10 % of transcriber training audio is clean, general English (the sample
  scripts in the Piper voice plus a public read-speech sample), and the clean sample cases are scored
  after every run, so a radio-tuned model is not allowed to lose ordinary speech.

## 3. What gets fine-tuned, and how

**3a. The local transcriber (Whisper).**
- Model: whisper-small (or base.en if memory on the device demands it), LoRA on the encoder and
  decoder attention, learning rate 1e-5 to 1e-4, 3 to 5 epochs over about 1000 clips; mixed with the
  forgetting-control audio. Trained with Hugging Face transformers and PEFT on a rented GPU (T4 or A10,
  one to two hours) or overnight on the laptop; the Jetson is for evaluation, not training.
- Deployment: merge the LoRA, convert with `ct2-transformers-converter` to CTranslate2 int8, drop it
  in as `LOCAL_WHISPER_MODEL`. The app changes by one setting; that is part of H1.
- Compared against: the baseline model, the baseline with the radio-phrase hint, and the baseline with
  hotword boosting of the term list (the cheap competitor that fine-tuning must beat).
- Expected: the large gain is channel adaptation at 10 and 5 dB SNR and on the term list; near zero
  gain at clean.

**3b. The report model (qwen3:4b-instruct, or qwen3:1.7b for headroom).**
- QLoRA on (noisy transcript, gold JSON) pairs, about 1000 to 2000 examples plus 200 pasted-text
  pairs and 100 "nothing stated" pairs whose gold is all null, 2 to 3 epochs, on a 24 GB GPU for one
  to three hours. The prompt stays the production extraction prompt so the model is tuned to the
  prompt it will see.
- Deployment: merge, convert to GGUF, `ollama create handoff-qwen3:4b` from a Modelfile; the app
  selects it by `OLLAMA_MODEL`. Again one setting.
- Compared against: the base model with the current prompt, and the base model with the prompt
  rewritten using the same examples as few-shot (prompting is the cheap competitor here).
- Targets: field score up, alert-type accuracy up (the sepsis-as-ENDOCRINE case is in the training
  distribution now), hallucinated interventions and medications at zero with the rules on, and the
  null-unless-spoken behaviour for ETA, allergies and code status learned rather than only enforced.

**3c. Not fine-tuned, but tested the same way.** Vosk captions (no practical fine-tuning path; swap
candidate: the fine-tuned whisper tiny as the caption engine, scored on caption WER and latency); the
transcript corrector and lexicon (grown from the term list; precision of corrections measured); the
barrier rules (coverage measured: which hallucinations they catch and which they miss).

## 4. The experiment protocol

1. Freeze a baseline commit. Run the whole stack in replay mode over the test split and the real
   call-sets: audio in, captions, transcript, report out, one folder per run (`~/reports_replay`), as
   the demo batch already does. Record the per-clip metrics and the device's memory and timing.
2. Fine-tune 3a. Re-run the stack with only the transcriber changed. Then 3b with only the report model
   changed. Then both. Then each with the captions fallback and the rules switched off (for H2 and H3).
3. Statistics: paired bootstrap over clips for every before-and-after difference, 95 % intervals;
   Clopper-Pearson intervals for rates (hallucination, dropped tail); conformal calibration of the
   abstention threshold on the validation split, coverage reported on test.
4. Report: one table per hypothesis, WER and term-error curves against SNR, field score per alert type,
   and the device budget. The real call-sets get their own column so synthetic gains are checked
   against reality.
5. **The off-the-shelf bar.** Every transcript table carries the same columns for plain Whisper with
   nothing of ours in front of it: Groq whisper-large-v3-turbo and large-v3 without the hint, the
   local faster-whisper medium and small without hint or corrector, and Vosk alone. Beside WER, the
   average n-gram F (mean over 1- to 4-gram precision/recall against the reference,
   `scripts/corpus/ngram_scores.py`) and 4-gram recall, which is the share of whole phrases that came
   through intact and is what the report model actually reads. "Ours" (hint, corrector, caption repair,
   checks, and later the fine-tuned model) has to beat that bar on the real call-sets, not only on the
   generated set; a change that does not is not shipped. On the three gold call-sets today, the hint
   alone takes Groq turbo from 20 % to 14 % WER and 0.72 to 0.75 average n-gram F
   (NGRAM_2026-10-26.md). On the generated set's test split (SYNTH_V1_2026-10-27.md) the bare models stand at
   5.3 % WER / 9.0 % terms missed (Groq turbo, clean) and 4.6 / 7.5 (Groq large-v3); hint plus corrector takes
   turbo to 4.2 / 8.5, and the hint alone takes large-v3 to 3.9 / 6.5. The local medium bar (hint off) is the
   `medium_nohint` row of that report.

## 5. Stop rules and what counts as proof

- 3a is kept only if term error at 10 dB falls by at least 25 % relative over hotword boosting with no
  loss at clean; otherwise the hint plus boosting is the production answer and the finding is recorded.
- 3b is kept only if the field score rises on the real call-sets as well as the synthetic test split;
  a gain on synthetic only means the generator is being learned, not the task.
- The architecture is "proven" to the extent H1 to H6 hold with intervals that exclude zero. A
  hypothesis that fails is reported as failed, with the curve; that is also evidence, about where the
  design has to change.

## 6. Effort, compute, risks

- Effort: three to four weeks after the gold set exists; most of it the protocol runs on the device
  (the demo batch took a day for 22 call-sets on two providers; the test split is larger, so budget
  several days of unattended replay).
- Compute: tens of dollars of rented GPU for both fine-tunes; the Jetson for everything else.
- Risks: overfitting to the cloned voices (held-out voice and the real call-sets guard it); forgetting
  general English (the clean mix and the clean sample scores guard it); the report model breaking its
  JSON (the existing schema validation catches it, and the rate is measured); GGUF and CTranslate2
  conversions changing behaviour (the same evaluation is run on the converted model, never on the
  PyTorch one); the synthetic noise having only one type, so robustness claims are limited to the
  corpus floor until a siren and a cabin recording are added.
