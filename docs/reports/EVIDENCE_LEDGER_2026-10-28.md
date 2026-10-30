# Evidence ledger: what we hold, what we have run, and where that leaves the abstract (2026-10-28)

One place for every dataset, every gold set, every evaluation run since the takeover on 2026-10-18, how long each
kind of run takes, and the bare-Whisper comparison. Companion to SYSTEM_VS_BASE_2026-10-28.md (the matrix and
the claim-by-claim table), CALCULATIONS_2026-10-28.md (the arithmetic) and POSTERIOR_2026-10-28.md (the statistics).

## 1. Gold data and references

| set | what it is | size | made by | used for |
|---|---|---|---|---|
| `tests/gold` | four sample handoffs (quick, STEMI, trauma, sepsis) with a gold JSON each | 4 cases | hand-written | the pasted-text path; regression on every commit |
| `~/labels/gold` | gold JSON for real call-sets CS001, CS008, CS014 | 3 calls | hand-labelled from the audio | report scoring on real calls |
| `~/labels/reference` | reference transcripts for the bursts of those three calls (assistant-drafted from several transcriber outputs, judgement calls marked; not human-verified) | 9 bursts, 612 normalised words | typed and corrected by hand from candidate transcripts | WER and n-gram scoring on real audio |
| `~/datasets/synth_v1` | 200 generated radio scripts, each with text, spoken form and gold JSON; rendered in one Piper voice at clean, 20, 15, 10, 5 dB through the measured radio channel | 200 scripts, 1000 clips (plus 200 practice renders), split 140 / 30 / 30 by script | generated 2026-10-27 | transcriber curves by SNR, term error, report scoring; the only set large enough for intervals |
| `~/datasets/synth_v1/terms.txt` | the medical term list for MTER | 149 terms, 200 occurrences in the 30 test scripts | from the lexicon and the scripts | term recall |
| real call corpus | public radio feed, cut into call-sets with the corpus pipeline | 22 call-sets, 51 bursts, 1869 s of audio, 22 labelled rows | stages 2 to 6 of `~/labels/scripts` | replay, demo batch, hallucination review; 19 still lack a human reference |

What does not exist: human verification of any reference, references for the other 19 calls, a second annotator on anything, a term list for real audio,
a second synthetic voice (XTTS clones wait for a GPU), and any audio from a real ambulance cabin.

## 2. Every evaluation run, in order

Times are wall clock on the Jetson Orin Nano (CPU int8 faster-whisper; qwen3:4b-instruct on the GPU through Ollama)
unless marked cloud.

| date | run | size | time | headline | where |
|---|---|---|---|---|---|
| 10-24 | Baseline: pipeline over the real corpus, local 4B | 45 of 73 reports (28 missing: crashes, refusals, wrong-model rejections) | 52.7 s mean LLM time per report | first measured point; 120 claims not in the transcript over 45 reports | BASELINE.md |
| 10-24 | Post-fix round, same corpus | 73 of 73 | about 1 h | claims not in transcript 120 to 67 over the 45 shared reports; invented allergies 36 to 50 (worse), code status 34 to 23; four sample cases 83.5 to 88.6 weighted | POSTFIX.md |
| 10-24 | Round 2, same corpus, grounding rules | 73 of 73 | about 1 h | unsupported claims 98 to 32; invented allergies and code status 50 / 23 to 0 / 0; crashes 4 to 0; samples 91.3 | ROUND2.md |
| 10-24 | Cloud report writer on the same transcripts (gpt-oss-120b) | 73, then 16 (daily token cap) | minutes | 0 / 0 invented allergies and code status before and after; the comparison files | reports_groq, reports_groq_round2 |
| 10-25 | App commits scored on the three gold calls (10 commits, replay) | 3 calls x 10 | 2 to 4 min per call | transcript WER 11 to 13 % on CS008 across commits; report 94.0 local, 98.0 cloud | SCORES_2026-10-25.md |
| 10-25 | Local stability: the same call replayed five times on one commit | 5 x CS008 | 20 min | WER 11 to 13 % (Groq transcript); report identical all five times (94.0, same miss) | STABILITY_LOCAL_2026-10-25.md |
| 10-25 | Whisper hint ablation on the cloud model | 9 bursts, 4 variants | minutes | no prompt 28.2 %, radio hint 23.1 %, terms 23.6 %, hint plus names 28.9 % | HINT_ABLATION_2026-10-25.md |
| 10-25 to 26 | Demo batch: every call-set replayed live and recorded, cloud and local report writer | 22 x 2 recordings, 57 recordings in all | 12.5 h end to end | all 44 pass the sync check; cloud and local agree on type, level and ETA on 9 of 22 | DEMO_2026-10-26.md |
| 10-26 | Over the air: gold calls played through a speaker into the mic | 3 calls | 20 min | WER roughly doubles (12 to 15, 32 to 30, 17 to 45 %); no hallucinations; CS001 loses its vitals | AIR_2026-10-26.md |
| 10-26 | Over the air: practice scripts, Piper and XTTS voices, with and without the radio EQ | 11 runs | 30 min | STEMI 35 % plain to 8 % with EQ; trauma 5 to 6 %; every report had the right type, level and ETA | AIR_SCRIPTS_2026-10-26.md |
| 10-26 | Spectral survey of the corpus and the mic | 51 bursts | minutes | the feed is band-limited at 4.6 kHz; the dongle adds a shelf above 4.5 kHz | SPECTRAL_2026-10-26.md |
| 10-26 | n-gram agreement, app against off-the-shelf Whisper | 3 calls, 7 transcribers | minutes | app 0.744 avg F, hint-on cloud 0.749, no-prompt cloud 0.718, local medium 0.717, Vosk 0.216 | NGRAM_2026-10-26.md |
| 10-27 | Generated set built and rendered | 200 scripts, 1000 clips | 2 h | the set | SYNTH_V1_2026-10-27.md |
| 10-27 | Report model on the clean script text | 30 reports | 27 min | 96.4 weighted, 98.4 recall, 0.43 hallucinations per report (the prompt's own ceiling) | SYNTH_V1 |
| 10-27 | 16 transcriber runs over the 150-clip test split: medium (hint on, off, corrector, fallback, re-decode, hotwords), small, tiny.en, large-v3-turbo (plain, fallback), cloud turbo and large-v3 (hint on, off, corrector) | 150 clips each, 6285 s of audio | local 0.8x to 0.85x real time (about 85 to 90 min per run; small 34 min, tiny 11 min); cloud 7 to 9 min paced | the full table in SYSTEM_VS_BASE section 2; window collapse found; hotwords break medium; fallback is the wrong lever; re-decode fixes the one medium collapse | SYNTH_V1 |
| 10-27 | Report model over the medium transcripts, all SNRs (first 69; the rest on 10-28, next row) | 69 of 150 reports | 48 s per report | 95.1 to 95.9 weighted at every SNR on the partial set | reports/medium_hint |
| 10-28 | Offline run: CS008 fully local with every outbound call blocked, and the cloud control | 1 call x 2 | 3 min | local: 143 s, WER 18.4 %, weighted 88.0, no hallucination; cloud control fails to connect | reports_replay/CS008_offline |
| 10-28 | Re-decode on large-v3-turbo, then the four touched clips under the loop guard | 150 + 20 clips | 90 + 12 min | 13.4 % unguarded, 12.6 % guarded, against 12.8 % plain; two spans looped | SYSTEM_VS_BASE |
| 10-28 | The stacks on the real calls: base and ours, local and cloud, plus cloud large-v3 | 9 bursts x 5 | cloud 3 min, local 2 min | cloud base 27.6, local base 25.7, local ours 22.7, cloud ours 21.7, cloud large-v3 ours 16.2 % | REAL_STACK_2026-10-28.md |
| 10-28 | Report stage on the generated set: cloud writer over our cloud transcripts | 60 reports | 25 min paced (cloud) | 96.8 clean, 95.1 at 5 dB | reports/groq_hint_corr |
| 10-28 | Report stage, local writer over the base local transcripts and over our local transcripts (second Jetson) | 60 + 60 | 50 s per report | 95.7 / 94.1 and 93.7 / 93.7; the clean difference traced to report-model variation on near-identical input and one real collapse | SYSTEM_VS_BASE section 4 |
| 10-28 | Report stage with the grounding rules off (cohendynamics GPU) | 60 | 50 min | 88.0 / 88.3 against 93.7 / 93.7 with rules on; precision 89 against 96 %; hallucinations 0.7 against 0.2 per report; level right 67 against 100 %; +5.8 [+4.1, +7.5] and +5.4 [+3.8, +7.1] for the rules | SYSTEM_VS_BASE section 4 |
| 10-28 | Repeat of the bare-transcript report run, same device, same input (cohendynamics GPU) | 60 | 50 min | 60 of 60 weighted scores identical, 51 of 60 report JSONs byte-identical outside metadata: the writer is deterministic on one device, so the 2-point clean gap is caused by transcript formatting, not noise | SYSTEM_VS_BASE section 4 |
| 10-28 | medium_hint all-SNR report completion (vox-ai GPU; one report, SY099 at 10 dB, failed on malformed model JSON and was re-run on 10-29) | 81 | 70 min | the local report stage's noise curve, 150 of 150: weighted 94.7 / 94.7 / 92.9 / 94.2 / 93.9 at clean / 20 / 15 / 10 / 5 dB while the transcript WER goes 4.8 to 13.5 %; level right at every SNR; hallucinations 0.13 to 0.23 per report | SYSTEM_VS_BASE section 4 |
| 10-29 | Report stage, cloud writer over the bare cloud transcripts (Groq, paced 25 s) | 60 | 25 min | 95.2 / 94.4 weighted against 96.8 / 95.1 over our cloud stack; paired +1.5 [-0.5, +3.9] at clean, +0.5 [-1.6, +2.8] at 5 dB |
| 10-29 | Report stage, cloud writer with the rules off | 13 of 60 | 25 min, then HTTP 429 | 93.6 / 90.2 on the 13; the Groq daily token cap stopped the run, it resumes when the cap resets |
| 10-29 | Candidate transcribers on the test split (vox-ai CPU): distil-large-v3, distil-large-v3.5, turbo float32, turbo without VAD, medium beam 5, turbo beam 5 | 6 x 60 clips | 20 to 90 min each | none beats medium: 9.4 / 10.5, 8.8 / 12.6, 9.2 / 11.0, 12.8 / 10.7, 7.1 / 6.9 (1.2x), 6.2 / 8.1 % (0.81x) | SYSTEM_VS_BASE section 2 |
| 10-29 | Span re-decode rewritten (collapsed, sparse, repeating, trailing, looping and over-long windows; beam 5 on the span; 3 s lead-in with seam trim), seven decode variants measured on medium and turbo | 7 x 60 clips | 30 to 60 min each | medium 7.1 greedy to 5.0 / 7.0 % (shipped), turbo 12.8 / 10.8 to 6.0 / 8.5 % at greedy speed; 6 new tests, 245 pass | commits 0431bbb to 8cbfc31 |
| 10-29 | GPU whisper on cohendynamics: dustynv CUDA container, scipy added, app decode | bench 9 bursts; 3 models x 60 clips | 5 to 8 min per model | medium fp16 5.4 s per burst against 27.8 s CPU (5.1x); medium 4.6 / 7.3 % (5.3 s a clip), turbo 6.8 / 8.5 (4.1 s), large-v3 4.6 / 5.5 (8.4 s); with the corrector 4.5 / 7.1 and 4.6 / 5.5 | gpu_whisper on cohendynamics; evals/*_gpu_fp16_hint_rd5d |
| 10-29 | The real nine bursts on the GPU, every stack with the corrector, after two loop fixes found on them | 9 bursts x 2 models | 2 min | GPU large-v3 21.7 % (equals cloud turbo stack, beats CPU medium stack 22.7), GPU medium 25.8 % (one burst looped in fp16); first pass before the fixes was 31.5 % | SYSTEM_VS_BASE section 2, real-burst table |
| 10-29 | qwen3:8b report rows over the base and stack CPU transcripts (cohendynamics, 13 % of layers on CPU) | 56 + 50 of 60 | 171 s median, 270 s p90, 32 timeouts re-run | 96.2 / 94.2 and 95.6 / 94.8; against the 4B +0.2 [-1.0, +1.5] and +1.2 [-0.3, +2.9] at clean; offline-only, not a live writer | reports/*__qwen3-8b |
| 10-29 | 4B report rows over the GPU medium transcripts, with and without the corrector (vox-ai GPU) | 59 + 59 of 60 (SY117 at 5 dB: malformed model JSON both times) | 52 min each | 95.3 / 94.2 both (the corrector touched 3 transcripts, no score); +1.6 [+0.1, +3.4] over the CPU stack row at clean; -1.5 [-3.9, +0.9] against the cloud stack; level with the cloud base system | reports/medium_gpu_fp16_hint_rd5d*__qwen3-4b-instruct |
| 10-29 | Plain-decode GPU base rows (no hint, no re-decode, no corrector), medium and large-v3 fp16 on cohendynamics | 2 x 60 clips | 5 + 8 min | medium 7.5 / 8.3 % (stack -2.8 [-5.9, -0.9] at clean, -1.1 [-2.2, -0.1] at 5 dB); large-v3 4.5 / 6.3 % (stack -0.1 [-0.8, +0.6], -1.0 [-1.8, -0.2]); plain GPU large-v3 equals plain cloud large-v3 to the clip | evals/*_gpu_fp16_nohint |
| 10-29 | Whisper hint ablation on the cohendynamics GPU: hint trimmed to fit the 223-token cut, and sentence + radio phrases only; medium and large-v3 fp16 | 4 x 60 clips | 27 min | trimmed = current (medium +0.0 [-0.4, +0.5]); short worse (medium +1.0 [+0.2, +1.8] at clean); the terminology list is the hint's value, the opening sentence is worth nothing | evals/*_gpu_fp16_hint{fit,short}_rd5d |
| 10-29 | 4B report rows over the GPU large-v3 transcripts, with and without the corrector (vox-ai GPU) | 60 + 60 | 52 min each | 96.6 / 95.3 both; against the cloud stack -0.1 [-2.4, +2.0] at clean, +0.2 [-2.4, +2.7] at 5 dB: level; +1.4 / +0.9 over the GPU medium stack | reports/large-v3_gpu_fp16_hint_rd5d*__qwen3-4b-instruct |
| 10-30 | 4B report rows over the plain-decode GPU medium transcripts (vox-ai GPU) | 60 | 52 min | 94.1 / 94.1; the GPU stack is +1.2 [-0.7, +3.5] at clean and +0.1 [-1.4, +1.7] at 5 dB over it at the report stage; large-v3 base row running | reports/medium_gpu_fp16_nohint__qwen3-4b-instruct |
| 10-30 | 4B report rows over the plain-decode GPU large-v3 transcripts (vox-ai GPU) | 60 | 52 min | 96.0 / 95.4; the stack adds +0.7 [-0.7, +2.2] / -0.1 [-1.5, +1.2] over it; plain local large-v3 + 4B is +0.7 / +1.0 over plain cloud turbo + 120B (intervals through zero); the report matrix is complete bar the cloud rules-off remainder | reports/large-v3_gpu_fp16_nohint__qwen3-4b-instruct |
| 10-30 | Lexicon retrieval off, 4B writer over the large-v3 GPU stack transcripts (the RAG ablation) | 59 of 60 | 52 min | 95.0 / 94.7 against 96.6 / 95.1 with it on: +1.6 [-0.2, +3.3] at clean, +0.4 [-1.3, +2.0] at 5 dB, recall 94.0 to 96.1 % (P = 0.92); lever ranking: rules 5.8, lexicon 1.6, transcript stack 0.7, corrector 0, 8B writer within noise | reports/large-v3_gpu_fp16_hint_rd5d_corr__qwen3-4b-instruct_nolex |
| 10-30 | CS008 end to end, split pipeline (whisper server on the cohendynamics GPU, 4B on vox-ai) x 3, and one box the same morning | 4 runs | 6 min | split 70.4 / 71.0 / 74.5 s wall (transcript 15 to 21 s, writer 50 to 52 s); one box 134.6 s (67 + 65 s); the split pipeline halves the time to report; the writer is now 70 % of it | logs/split2_2026-10-30.log |
| 10-30 | large-v3 GPU noise curve at 20 / 15 / 10 dB through the whisper server (eval_set --provider remote), while vox-ai's writer ran | 90 clips | 14 min | 4.5 / 4.4 / 5.0 % against cloud large-v3's 3.8 / 3.6 / 4.0 (+0.7 to +0.9, intervals excluding or touching zero), term recall equal; full local curve 4.6 / 4.5 / 4.4 / 5.0 / 5.5 % | evals/large-v3_gpu_fp16_hint_rd5d_curve |
| 10-30 | Held-out check: large-v3 GPU, shipped decode, on the 30 VALIDATION scripts (never used while tuning), five levels | 150 clips | 20 min | 4.1 / 4.3 / 4.6 / 4.9 / 5.7 % against the test split's 4.6 / 4.5 / 4.4 / 5.0 / 5.5: within 0.5 points everywhere; term error 12.2 against 9.0 % at 5 dB; validation report row running | evals/large-v3_gpu_fp16_hint_rd5d_VAL |
| 10-30 | Compact-output writer row (omit empty fields) over the large-v3 GPU stack transcripts | 60 | 50 min | 96.0 / 95.4 against 96.6 / 95.3 (-0.6 [-1.4, +0.0] at clean); 748 against 806 output tokens, 48.5 against 51.6 s: 6 % faster for a possible half point, not shipped; the bulk of the output is the prose sections, not the nulls | reports/*_compact |
| 10-30 | Held-out report row: the local system (GPU large-v3 stack, 4B writer) on the 30 VALIDATION scripts | 59 of 60 | 52 min | 94.6 / 93.6 against the test split's 96.6 / 95.3: -2.1 [-5.0, +0.7], -1.7 [-4.8, +1.5] unpaired; hallucinations 0.30 / 0.34 against 0.17 / 0.20; the medical, arrest and sepsis families carry the loss; cloud on the validation split queued | reports/large-v3_gpu_fp16_hint_rd5d_VAL_corr__qwen3-4b-instruct |
| 10-30 | Cloud system on the 30 VALIDATION scripts (Groq turbo transcripts 5.2 / 6.5 % WER, 120B writer, paced) | 60 + 60 | 35 min | 95.8 / 93.8 against its 96.8 / 95.1 on the test split; local against cloud on the held-out scripts -1.3 [-3.6, +1.2], -0.5 [-2.6, +1.8]; the local transcript beats the cloud turbo transcript on unseen scripts (4.1 / 5.7 against 5.2 / 6.5 %) | evals/groq_hint_VAL*, reports/groq_hint_VAL_corr__openai/gpt-oss-120b |

Regression on every commit: 245 tests under `.venv/bin/python -m pytest` (2 min), including the pipeline on the four
gold samples with a fake model, the re-decode and its loop guard, the rules switch, and the transcript checks.

## 3. Bare Whisper out of the box against our stack

"Out of the box" is faster-whisper (or the Groq endpoint) with no prompt, no re-decode, no corrector, scored with the
same scorer on the same clips.

| | bare medium (local) | our local stack | bare cloud turbo | our cloud stack |
|---|---|---|---|---|
| speed, transcribe time over audio time | 0.80x | 0.85x (the hint and corrector add about 6 %; the re-decode fires on 1 clip in 150) | 0.08x (network bound, paced) | 0.07x |
| WER, generated set, clean / 5 dB | 5.8 / 8.1 % | 5.8 / 6.9 % | 5.3 / 7.5 % | 4.2 / 6.4 % |
| term recall, generated set, pooled over five SNRs | 87.1 % | 90.5 % (P = 99.2 %) | 87.8 % | 88.5 % (P = 68.6 %) |
| WER, real calls, 612 words | 25.7 % | 22.7 % | 27.6 % | 21.7 % |
| 4-gram recall, real calls | 0.53 | 0.55 | 0.54 | 0.62 |
| clips over 30 % WER, generated set | 2 of 150 | 1 of 150 | 0 | 0 |

So: we are not faster than bare Whisper on the same hardware; we are about 6 % slower and the same memory. We are
more accurate by 3 points of WER and 3.4 points of term recall on real and noisy audio, and the same at clean on
the generated set. The cloud is ten times faster than the Jetson CPU for the transcript and twenty times for the
report; that is the price of running without a network. The one speed lever is model size (small is 2.6 times
faster at 2 more points of WER), and the one big accuracy lever was choosing medium over the turbo default (8 points).

Revised 10-29. Two things changed the local column. The re-decode now catches every kind of collapsed window, so
medium on the CPU is 5.0 / 7.0 % (from 7.1 % greedy under the same hint) and turbo 6.0 / 8.5 % at greedy speed. And
the second device's GPU runs the same decode in fp16: medium 4.6 / 7.3 % at 0.13 times real time (6.6 times the CPU),
large-v3 4.6 / 5.5 % at 0.20 times, against the best cloud row's 3.9 / 4.9 % at 0.07 times. On the real bursts GPU
large-v3 with the stack is 21.7 %, equal to the cloud turbo stack. The speed is the hardware's, not the stack's: bare
medium on the GPU would be as fast. The accuracy gain over bare medium at 5 dB is now 0.8 to 1.1 points on the
generated set and 3 to 4 on the real calls; the local-against-cloud transcript gap is 0.6 to 0.7 points on the
generated set for the same model (large-v3) and zero on the real bursts for turbo-class cloud.

## 4. The report stage: bare model against ours, local against cloud

| | local qwen3:4b-instruct, bare prompt | local, our prompt and rules | cloud gpt-oss-120b, our prompt and rules |
|---|---|---|---|
| real corpus, 73 reports: claims not supported by the transcript | 98 (post-fix, before the rules) | 32 | 0 invented allergies or code status, before and after |
| real corpus: invented allergies / code status | 50 / 23 | 0 / 0 | 0 / 0 |
| four gold samples, weighted | 83.5 (baseline commit) | 91.3 | 98.0 on CS008 |
| generated set, 60 reports, weighted clean / 5 dB | 88.0 / 88.3 (rules off, same transcripts, same device) | 93.7 to 95.7 / 93.7 to 94.1 | 96.8 / 95.1 (95.2 / 94.4 over the bare cloud transcript) |
| hallucinations per report, generated set | 0.73 / 0.70 | 0.10 to 0.20 | 0.03 to 0.07 |
| generated set, 60 reports over the GPU transcripts with the 10-29 decode (local writer) | | medium 95.3 / 94.2 (-1.5 [-3.9, +0.9] against the cloud stack at clean); large-v3 96.6 / 95.3 (-0.1 [-2.4, +2.0], +0.2 [-2.4, +2.7]: level with the cloud) | |
| generated set, qwen3:8b writer over the CPU transcripts | | 96.2 / 94.2 bare, 95.6 / 94.8 stack (+0.2 to +1.2 over the 4B, intervals through zero) | |
| time per report | 50 s | 50 s (8B: 171 s median) | 3 s |
| repeatability | five replays of CS008 gave the identical report (10-25); a second run of 60 reports on the same device and input reproduced all 60 scores (10-28). The 11-clip difference between bare and stacked transcripts at clean is therefore caused by the transcripts' punctuation and case, to which the 4B writer is sensitive | | |

## 5. How long things take (for planning)

| job | time on the Orin Nano |
|---|---|
| one transcriber run over the 150-clip split, medium or turbo | about 90 min; small 34 min; tiny.en 11 min; cloud 7 to 9 min |
| one transcriber run over the 60-clip clean + 5 dB rows on the cohendynamics GPU (CUDA container) | medium 5 min, turbo 4 min, large-v3 8 min; model load 20 to 60 s; one container per model |
| the corrector pass over saved transcripts | seconds |
| one report-stage run, 60 reports, local | 50 min (50 s per report); cloud 25 min paced, 3 s per report |
| replay of one real call through the live server with a recording | 2 to 4 min |
| one real call end to end, headless: one box 135 s; split pipeline (GPU whisper on the other Jetson) 70 to 75 s | |
| the demo batch, 22 calls x 2 providers, recorded | 12.5 h |
| the full test suite | 2 min |
| rendering the generated set (Piper, five SNRs) | 2 h; XTTS clones 8 min per script on this CPU |

Whisper and Ollama cannot share this box's 7.4 GB; a second Orin Nano Super (cohendynamics) now runs the report
stage at 24 tokens a second, so transcriber and report runs go in parallel; since 10-29 it also runs faster-whisper on
its GPU through a CUDA container (the first device's JetPack is too old for it), with one model per container.

## 6. Where that leaves the abstract

The claim-by-claim table is section 7 of SYSTEM_VS_BASE_2026-10-28.md. In one paragraph: the recordings, the
configurations, the metrics, the statistic and the disconnection test can all be produced from what is on disk,
with their definitions changed to human ground truth. The recall claim has a counterpart (87.1 to 90.5 % term
recall, P = 99.2 %). The precision-from-filtering claim, the 1.5 to 2 times speedup and "both edge models
outperformed the cloud" do not; the defensible statements are "within a point of the cloud on the same model
class", "3 to 6 points better than the bare model on real audio", "runs disconnected", and "abstains rather than
invents". After 10-29 the local transcript on the second device's GPU is within a point of the same cloud model and
the local report over large-v3 GPU transcripts is level with the cloud stack on the test split (96.6 / 95.3 against 96.8 / 95.1, -0.1 [-2.4, +2.0]) and on the held-out validation split 94.6 / 93.6 against the cloud's 95.8 / 93.8 (-1.3 [-3.6, +1.2]; the held-out scripts are harder for both), at a fifth of real time; the
speed claim still belongs to the hardware, not to the stack, and an 8B writer bought nothing measurable. Twenty-two calls with three references and thirty synthetic scripts in one voice is a pilot, and every
number above should be read with that sample size in front of it.
