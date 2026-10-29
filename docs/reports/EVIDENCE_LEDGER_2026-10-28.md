# Evidence ledger: what we hold, what we have run, and where that leaves the abstract (2026-10-28)

One place for every dataset, every gold set, every evaluation run since the takeover on 2026-10-18, how long each
kind of run takes, and the bare-Whisper comparison. Companion to SYSTEM_VS_BASE_2026-10-28.md (the matrix and
the claim-by-claim table), CALCULATIONS_2026-10-28.md (the arithmetic) and POSTERIOR_2026-10-28.md (the statistics).

## 1. Gold data and references

| set | what it is | size | made by | used for |
|---|---|---|---|---|
| `tests/gold` | four sample handoffs (quick, STEMI, trauma, sepsis) with a gold JSON each | 4 cases | hand-written | the pasted-text path; regression on every commit |
| `~/labels/gold` | gold JSON for real call-sets CS001, CS008, CS014 | 3 calls | hand-labelled from the audio | report scoring on real calls |
| `~/labels/reference` | human reference transcripts for the bursts of those three calls | 9 bursts, 612 normalised words | typed and corrected by hand from candidate transcripts | WER and n-gram scoring on real audio |
| `~/datasets/synth_v1` | 200 generated radio scripts, each with text, spoken form and gold JSON; rendered in one Piper voice at clean, 20, 15, 10, 5 dB through the measured radio channel | 200 scripts, 1000 clips (plus 200 practice renders), split 140 / 30 / 30 by script | generated 2026-10-27 | transcriber curves by SNR, term error, report scoring; the only set large enough for intervals |
| `~/datasets/synth_v1/terms.txt` | the medical term list for MTER | 149 terms, 200 occurrences in the 30 test scripts | from the lexicon and the scripts | term recall |
| real call corpus | public radio feed, cut into call-sets with the corpus pipeline | 22 call-sets, 51 bursts, 1869 s of audio, 22 labelled rows | stages 2 to 6 of `~/labels/scripts` | replay, demo batch, hallucination review; 19 still lack a human reference |

What does not exist: references for the other 19 calls, a second annotator on anything, a term list for real audio,
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

Regression on every commit: 239 tests under `.venv/bin/python -m pytest` (2 min), including the pipeline on the four
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

## 4. The report stage: bare model against ours, local against cloud

| | local qwen3:4b-instruct, bare prompt | local, our prompt and rules | cloud gpt-oss-120b, our prompt and rules |
|---|---|---|---|
| real corpus, 73 reports: claims not supported by the transcript | 98 (post-fix, before the rules) | 32 | 0 invented allergies or code status, before and after |
| real corpus: invented allergies / code status | 50 / 23 | 0 / 0 | 0 / 0 |
| four gold samples, weighted | 83.5 (baseline commit) | 91.3 | 98.0 on CS008 |
| generated set, 60 reports, weighted clean / 5 dB | 88.0 / 88.3 (rules off, same transcripts, same device) | 93.7 to 95.7 / 93.7 to 94.1 | 96.8 / 95.1 |
| hallucinations per report, generated set | 0.73 / 0.70 | 0.10 to 0.20 | 0.03 to 0.07 |
| time per report | 50 s | 50 s | 3 s |
| repeatability | five replays of CS008 gave the identical report (10-25); a second run of 60 reports on the same device and input reproduced all 60 scores (10-28). The 11-clip difference between bare and stacked transcripts at clean is therefore caused by the transcripts' punctuation and case, to which the 4B writer is sensitive | | |

## 5. How long things take (for planning)

| job | time on the Orin Nano |
|---|---|
| one transcriber run over the 150-clip split, medium or turbo | about 90 min; small 34 min; tiny.en 11 min; cloud 7 to 9 min |
| the corrector pass over saved transcripts | seconds |
| one report-stage run, 60 reports, local | 50 min (50 s per report); cloud 25 min paced, 3 s per report |
| replay of one real call through the live server with a recording | 2 to 4 min |
| the demo batch, 22 calls x 2 providers, recorded | 12.5 h |
| the full test suite | 2 min |
| rendering the generated set (Piper, five SNRs) | 2 h; XTTS clones 8 min per script on this CPU |

Whisper and Ollama cannot share this box's 7.4 GB; a second Orin Nano Super (cohendynamics) now runs the report
stage at 24 tokens a second, so transcriber and report runs go in parallel.

## 6. Where that leaves the abstract

The claim-by-claim table is section 7 of SYSTEM_VS_BASE_2026-10-28.md. In one paragraph: the recordings, the
configurations, the metrics, the statistic and the disconnection test can all be produced from what is on disk,
with their definitions changed to human ground truth. The recall claim has a counterpart (87.1 to 90.5 % term
recall, P = 99.2 %). The precision-from-filtering claim, the 1.5 to 2 times speedup and "both edge models
outperformed the cloud" do not; the defensible statements are "within a point of the cloud on the same model
class", "3 to 6 points better than the bare model on real audio", "runs disconnected", and "abstains rather than
invents". Twenty-two calls with three references and thirty synthetic scripts in one voice is a pilot, and every
number above should be read with that sample size in front of it.
