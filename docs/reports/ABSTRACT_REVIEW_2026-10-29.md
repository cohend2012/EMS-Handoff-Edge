# Abstract review, 2026-10-29: "From Classroom to Cabin"

The draft abstract as sent on 2026-10-29, checked claim by claim against the reports in this folder. Every
number has a source row or it is marked. A corrected draft follows, with today's numbers where they have
moved.

## 1. Claim by claim

| draft says | evidence | verdict |
|---|---|---|
| "1,000 simulated radio clips across five noise levels" | the set has 1,000 clips (200 scripts x 5 SNRs); transcripts were *scored* on the 150-clip held-out test split, reports on 60 of them (clean and 5 dB) | say "a 1,000-clip generated set (150-clip held-out test split)" |
| "22 real emergency calls" | 22 call sets, 51 bursts, 1,869 s of audio; WER needs references and only 3 calls (9 bursts, 612 words) have them; the 22 were used for the report corpus (73 reports) | say "22 real calls, three with human reference transcripts" |
| "local system achieved a 5.8 % to 6.9 % error rate" | SYSTEM_VS_BASE section 2: the 10-28 CPU stack was 5.8 / 6.9 %; after the 10-29 decode fixes it is 5.0 / 7.0 % on the CPU, 4.6 / 7.3 % (medium) and 4.6 / 5.5 % (large-v3) on the second device's GPU | update to the current rows, say which hardware |
| "medical entity recall rose from 87.1 % to 90.5 % (99.2 % probability)" | POSTERIOR_2026-10-28: term recall pooled over five SNRs, bare medium against the CPU stack, P = 0.992 | correct; call it "medical-term recall", "P(stack > base) = 0.99" |
| "on real-world audio, reduced transcription errors from 25.7 % to 22.7 %" | REAL_STACK_2026-10-28: bare medium 25.7 %, CPU stack 22.7 % on 612 words; on the GPU, large-v3 with the stack is 21.7 % | correct; add the 612-word sample size |
| "grounding rules cut unsupported claims from 98 to 32" | ROUND2.md, 73 real-corpus reports | correct |
| "completely eliminated fabricated code-status entries (from 50 to zero)" | invented *allergies* went 50 to 0; invented *code status* went 23 to 0 | wrong pairing: "invented allergy entries from 50 to zero and code-status entries from 23 to zero" |
| "disabling these rules dropped clinical agreement from 100 % to 67 %" | section 4: with the rules off, the *alert level* (set by the stated-priority rule) was right 67 % against 100 %; weighted score fell 5.8 points [4.1, 7.5]; hallucinations 0.2 to 0.7 per report | name the measure: "alert-level agreement with gold"; add the weighted-score and hallucination numbers, which are the stronger result |
| "calls were processed in 143 seconds without cell service" | the offline run of CS008 (84.5 s of audio): whisper 66 s, writer 74 s, 143 s wall, every outbound connection blocked | correct for one call on one device; the average real call is 85 s of audio, so CS008 is representative; with whisper on the second device's GPU the same call measured 70 to 75 s over three runs on 10-30 |
| "cloud processing took 14 seconds" | not measured as one number: the cloud control in the offline test failed to connect by design; the cloud transcript runs at 0.07 x real time (about 6 s for CS008) and the cloud report at about 3 s, so 9 to 14 s is plausible but has no row | either measure it (one replay of CS008 with the cloud provider, timing logged) or say "about ten seconds" with the two component rates cited |
| "well within the standard debriefing window" | no source for the window's length | cite one, or drop |
| "safely abstain rather than invent clinical data" | 0 invented allergy / code status on 73 real reports; 0.10 to 0.24 hallucinations per report on the generated set, not zero | keep "abstain rather than invent" for the two rule-protected fields; do not say hallucination-free anywhere |
| "hallucination-free framework" (conclusion) | hallucinations per report are 0.03 to 0.24 in every row | overclaim; say "low-hallucination" or "grounded" |
| "processing latency can be further reduced via dedicated neural network acceleration engines" | the measured lever was the GPU already on the second Jetson (whisper 6.6x faster, 143 s to about 85 s); an NPU is a plan, not a result | say "has been reduced ... using the device's GPU; a dedicated accelerator is the next step" |
| "Bayesian framework" | Beta-Binomial posterior on per-term outcomes and a paired clip-level bootstrap | fine; the methods section should name both |
| "standard AI models and cloud baselines" | bare faster-whisper medium and large-v3-turbo; Groq whisper-large-v3-turbo, whisper-large-v3, openai/gpt-oss-120b | name them once |
| "compact micro-computer" | NVIDIA Jetson Orin Nano, 8 GB; the GPU rows used a second Orin Nano Super | name the device |

Two things the abstract does not say and could: the local report over the large-v3 GPU transcripts is level with the cloud stack (96.6 / 95.3 against 96.8 / 95.1,
-0.1 [-2.4, +2.0]), and the plain
local large-v3 on the Jetson GPU equals the plain cloud large-v3 clip for clip, so the remaining cloud lead
is in prompt handling, not the model.

## 2. Corrected draft (same structure, current numbers)

**Background.** EMS training relies on simulation. Video records the physical interventions, but instructors
transcribe and synthesise the radio audio by hand for the debrief, and in field training EMTs document from
memory. Cloud transcription services raise data-handling questions and the models behind them can state
clinical facts that were never said. We evaluated a local device that listens to the radio report and
produces a structured handoff without a network, so that after-action review can start from a transcript
and a grounded report rather than from memory.

**Methods.** We built an offline pipeline on an NVIDIA Jetson Orin Nano (8 GB): faster-whisper for the
transcript, with a radio-phrase prompt, a re-decode of collapsed decoder windows and a medical-lexicon
corrector, and a 4-billion-parameter instruction model for a 25-field handoff report under grounding rules
that abstain rather than invent. It was benchmarked against the bare models and against cloud baselines
(Groq whisper-large-v3-turbo, whisper-large-v3 and gpt-oss-120b) on a 1,000-clip generated radio set (200
scripted reports at five signal-to-noise ratios, 150-clip held-out test split) and on 22 real public-feed
calls, three of which have reference transcripts and gold reports (612 words; drafted from several transcriber outputs and pending human verification). We measured word error
rate, medical-term recall, a weighted report score against gold with recall and precision over stated
facts, and the count of fabricated facts, with a Beta-Binomial posterior on per-term outcomes and a paired
clip-level bootstrap for differences.

**Results.** On the generated set the local transcriber scored 5.0 % WER at clean and 7.0 % at 5 dB on the
device's CPU, and 4.6 % / 5.5 % with the largest model on a second Orin Nano's GPU, against 3.9 % / 4.9 %
for the best cloud model; medical-term recall rose from 87.1 % with the bare model to 90.5 % with the
additions (P = 0.99). On real audio the additions reduced WER from 25.7 % to 22.7 % (612 words), and the
GPU configuration equalled the cloud turbo model with the same additions at 21.7 %. The grounding rules
cut unsupported claims in 73 real-call reports from 98 to 32 and invented allergy and code-status entries
from 50 and 23 to zero; turning them off on the same transcripts cost 5.4 to 5.8 weighted points
[3.8, 7.5], raised fabricated facts from 0.2 to 0.7 per report and dropped alert-level agreement from
100 % to 67 %. The local report score over the largest local model's transcripts was level with the cloud pipeline on the test split
(96.6 / 95.3 against 96.8 / 95.1, difference -0.1 [-2.4, +2.0]) and 94.6 / 93.6 on 30 held-out validation scripts. An average real call (85 s of audio) was processed in 143 s with every
outbound connection blocked, and in 70 to 75 s with the transcript on the second device's GPU (three runs).

**Conclusion.** An edge device can produce a grounded handoff report from radio audio with no network,
within a point or two of a cloud pipeline on transcript and report accuracy, and with the two fields most
prone to fabrication protected by rules. It is slower than the cloud; the measured remedy is the device's
GPU, and a dedicated accelerator is the next step. These results are a feasibility study on a pilot sample
(three referenced calls, one synthetic voice) and the next phase adds references for the remaining calls,
a second annotator, and the state-aware pipeline that lets the device flag missing fields before the
transmission ends.

## 3. Revision of 2026-10-30 (after the held-out validation check)

The draft resubmitted on 10-30 is unchanged from 10-29. What changed is the evidence: the 30 validation scripts
(never used while tuning) were scored, and the report stage lost about two points there (94.6 / 93.6 against
96.6 / 95.3 on the test split), while the transcript stage held (4.1 / 5.7 against 4.6 / 5.5 % WER for large-v3).
Verdicts that change or sharpen:

| draft says | verdict now |
|---|---|
| "5.8 % to 6.9 % error rate" | stale and under-specified. Current: 5.0 / 7.0 % (CPU, medium) and 4.6 / 5.5 % (second Jetson's GPU, large-v3) on the test split; 4.1 / 5.7 % on the held-out validation split. Say which hardware, which model, which split. |
| "87.1 % to 90.5 % (99.2 % probability)" | correct for the test split, CPU medium, pooled over five noise levels; the probability is a Beta-Binomial posterior that treats term occurrences as independent and overstates certainty; quote the clip-level interval too. The validation split's term error was 3 points higher at 5 dB, so say "on the test split". |
| "25.7 % to 22.7 % on real-world audio" | correct, 612 reference words over 9 bursts of 3 calls; add the sample size. The GPU configuration reaches 21.7 %. |
| "98 to 32", "50 to zero" | 98 to 32 correct (73 reports); "code-status entries from 50 to zero" is the wrong pairing: allergy entries 50 to 0, code-status entries 23 to 0. |
| "100 % to 67 %" | correct for alert-level agreement with the stated priority; name it, and add the stronger measured effects: 5.8 / 5.4 weighted points, hallucinations 0.2 to 0.7 per report. |
| "143 seconds without cell service" | correct, one call (CS008, 84.5 s of audio, the average length), 10-28; the same call is 70 to 75 s with whisper on the second Jetson's GPU (10-30, three runs). |
| "cloud processing took 14 seconds" | no measurement; drop or say "about ten seconds" from component rates, or run it. |
| "well within the standard debriefing window" | no source; cite one or drop. |
| report accuracy (not stated in the draft) | the draft tracks "report accuracy" in Methods but reports none; the number is 96.6 / 95.3 (test) and 94.6 / 93.6 (validation) weighted against gold; cloud 96.8 / 95.1 on the test split only. Say "within about two points of the cloud pipeline" and that the held-out set is two points lower. |
| "hallucination-free framework" | overclaim: 0.17 to 0.34 fabricated facts per report on the generated set. "Abstains rather than invents" holds for the two rule-protected fields (allergies, code status) on 73 real reports. |
| "1,000 simulated radio clips", "22 real emergency calls" | the set is 1,000 clips; scoring used the 150-clip test split and, from 10-30, the 150-clip validation split; reports 60 + 60. Real WER rests on 3 referenced calls of the 22; the 22 supplied the 73-report fabrication counts and the 10-30 state replay. |
| "acceleration engines" | the measured lever is the GPU already on the second Jetson (whisper 6.6x, call time halved); an NPU is a plan. |
| OODA-loop auditing | now partly built: a timed, sourced patient state with published prehospital rules, a documentation checklist per alert type, and an ask rule that fired on 5 of 22 real calls (4 naming items the crew did not state). Pilot-size; can be mentioned as implemented, not as evaluated. |

### Corrected draft, 10-30

**Background.** EMS training relies on simulation. Video records the physical interventions, but instructors transcribe
and synthesise the radio audio by hand for the debrief, and in field training EMTs document from memory. Cloud
transcription services raise data-handling questions, and the models behind them can state clinical facts that were
never said. We evaluated a local device that listens to the radio report and produces a structured, sourced handoff
without a network, so that after-action review starts from a transcript, a report and a timed record of what was
stated and what was missing, rather than from memory.

**Methods.** We built an offline pipeline on NVIDIA Jetson Orin Nano devices (8 GB): faster-whisper for the transcript,
with a radio-phrase prompt, a re-decode of collapsed decoder windows and a medical-lexicon corrector, and a
4-billion-parameter instruction model for a 25-field handoff report under grounding rules that abstain rather than
invent, followed by a patient-state layer that applies published prehospital rules (shock index, field-triage
vital-sign criteria, stroke and STEMI documentation elements) and lists what the call has not stated. It was benchmarked
against the bare models and against cloud baselines (Groq whisper-large-v3-turbo, whisper-large-v3 and gpt-oss-120b) on
a 1,000-clip generated radio set (200 scripted reports in one synthetic voice at five signal-to-noise ratios; a 150-clip
test split used during development and a 150-clip validation split held out until the end) and on 22 real public-feed
calls, three of which have reference transcripts and gold reports (612 words; drafted from several transcriber outputs and pending human verification). We measured word error rate,
medical-term recall, a weighted report score against gold with recall and precision over stated facts, and the count of
fabricated facts, with paired clip-level bootstrap intervals and a Beta-Binomial posterior on per-term outcomes.

**Results.** On the test split the local transcriber scored 5.0 % WER at clean and 7.0 % at 5 dB on the device's CPU,
and 4.6 % / 5.5 % with the largest model on a second device's GPU, against 3.9 % / 4.9 % for the best cloud model; on the
held-out validation split the GPU configuration scored 4.1 % / 5.7 %. Medical-term recall rose from 87.1 % with the bare
model to 90.5 % with the additions (P = 0.99, test split). On real audio the additions reduced WER from 25.7 % to 22.7 %
(612 words), and the GPU configuration reached 21.7 %. The grounding rules cut unsupported claims in 73 real-call reports
from 98 to 32 and invented allergy and code-status entries from 50 and 23 to zero; turning them off on the same
transcripts cost 5.4 to 5.8 weighted points [3.8, 7.5], raised fabricated facts from 0.2 to 0.7 per report and dropped
alert-level agreement from 100 % to 67 %. The local report scored 96.6 / 95.3 (weighted, clean / 5 dB) on the test split,
level with the cloud pipeline's 96.8 / 95.1 (difference -0.1 [-2.4, +2.0]), and 94.6 / 93.6 on the held-out split, with
0.3 fabricated facts per report. An average real call (85 s of audio) was processed in 143 s with every outbound
connection blocked, and in 70 to 75 s with the transcript on the second device's GPU. On the 22 real calls the state
layer flagged a published threshold on 8 and, at the close of 5, named documentation the crew had not stated (4 of 5
confirmed against the transcript).

**Conclusions.** An edge device can produce a grounded, sourced handoff report from radio audio with no network, within
about two points of a cloud pipeline on transcript and report accuracy, with the two fields most prone to fabrication
protected by rules, and can tell the receiver what a closing call has not stated. It is slower than the cloud; the
measured remedy is the device's GPU, and a dedicated accelerator is the next step. These are feasibility results on a
pilot sample (three referenced calls whose references await human verification, one synthetic voice), and the report stage lost two points on
held-out scripts, so the next phase verifies the references by listening, adds references for the remaining calls, a second annotator and a second voice.

## 4. Submission version, 2026-10-30 (due today): sent / now / future, and the text

What the sent draft says, what is true right now, and what belongs to future work. Only the middle column goes in.

| sent | true now | future |
|---|---|---|
| 5.8 to 6.9 % WER | 5.0 / 7.0 % (CPU medium) and 4.6 / 5.5 % (GPU large-v3) on the 30 test scripts; 4.1 / 5.7 % on 30 held-out scripts; exact references (the scripts) | another voice, a second channel |
| recall 87.1 to 90.5 %, P 99.2 % | true on the test split, bare medium against the CPU stack; quote P = 0.99 and say test split | held-out term recall (3 points worse at 5 dB) |
| 25.7 to 22.7 % on real audio | true against assistant-drafted references (9 bursts, 612 words), not human-verified; must say so or leave the number out | human-verified references for all 51 bursts |
| 98 to 32, code status 50 to 0 | 98 to 32 true (73 reports, statements with no supporting words in the transcript, counted automatically); allergies 50 to 0 and code status 23 to 0 | |
| 100 to 67 % | true for alert-level agreement; add 5.8 / 5.4 points and 0.2 to 0.7 fabricated facts | |
| 143 s, cloud 14 s | 143 s true (one call); 70 to 75 s with the second device's GPU; the cloud figure was never measured: drop | measure it |
| report accuracy (promised, not given) | 96.6 / 95.3 test, 94.6 / 93.6 held-out, cloud 96.8 / 95.1 test only | cloud on the held-out split (running) |
| hallucination-free | 0.2 to 0.3 fabricated facts per report; "abstains on allergies and code status" is the true claim | |
| OODA-loop auditing | a sourced patient state and a missing-documentation prompt exist and fired on 5 of 22 real calls; not evaluated by clinicians | clinician review |
| 1,000 clips, 22 calls | 1,000 clips generated, 300 scored; 22 calls: fabrication counts on all, WER on 3 against unverified references | |

### Text (about 370 words)

**Background.** EMS training relies on simulation. Video records the physical interventions, but instructors
transcribe and synthesise the radio audio by hand for the debrief, and in field training EMTs document from
memory. Cloud transcription raises data-handling questions, and the models behind it can state clinical facts
that were never said. We evaluated a local device that turns the radio report into a structured, sourced
handoff without a network, so that after-action review starts from a transcript and a report rather than
from memory.

**Methods.** We built an offline pipeline on NVIDIA Jetson Orin Nano devices (8 GB): faster-whisper for the
transcript with a radio-phrase prompt, a re-decode of collapsed decoder windows and a medical-lexicon
corrector; a 4-billion-parameter model for a 25-field handoff report under grounding rules that abstain
rather than invent; and a patient-state layer that applies published prehospital rules and lists what the
call has not stated. We benchmarked it against the bare models and cloud baselines (Groq
whisper-large-v3-turbo, whisper-large-v3, gpt-oss-120b) on a generated radio set (200 scripted reports in one
synthetic voice at five signal-to-noise ratios, 1,000 clips; a 150-clip test split used during development
and a 150-clip validation split held out until the end) and on 22 real public-feed calls (51 bursts). Metrics:
word error rate, medical-term recall, a weighted report score against gold with recall and precision over
stated facts, and the count of fabricated facts; paired clip-level bootstrap intervals.

**Results.** On the test split the local transcriber scored 5.0 % / 7.0 % WER (clean / 5 dB) on the device's
CPU and 4.6 % / 5.5 % with the largest model on a second device's GPU, against 3.9 % / 4.9 % for the best cloud
model; on the held-out split, 4.1 % / 5.7 %. Term recall rose from 87.1 % (bare model) to 90.5 % (P = 0.99).
On the real calls the grounding rules cut report statements unsupported by the transcript from 98 to 32 over
73 reports and invented allergy and code-status entries from 50 and 23 to zero; disabling them cost 5.4 to 5.8
weighted points [3.8, 7.5], raised fabricated facts from 0.2 to 0.7 per report and dropped alert-level
agreement from 100 % to 67 %. The local report scored 96.6 / 95.3 on the test split, level with the cloud
pipeline's 96.8 / 95.1 (difference -0.1 [-2.4, +2.0]), and 94.6 / 93.6 on the held-out split. An average call
(85 s of audio) was processed in 143 s with every outbound connection blocked, and in 70 to 75 s with the
transcript on the second device's GPU. Human-verified transcripts of the real calls are pending, so real-audio
word error rates are not reported.

**Conclusions.** An edge device can produce a grounded handoff report from radio audio with no network, within
about two points of a cloud pipeline on a generated benchmark, with the two fields most prone to fabrication
protected by rules. It is slower than the cloud; the device's GPU halves the gap, and a dedicated accelerator
is the next step. These are feasibility results on a pilot sample: one synthetic voice, and real-call accuracy
awaiting human-verified references.

### 300-word version for the STACS reviewers (11:25, after the cloud's held-out row)

The conference accepted the abstract (Best Practice Competition, STACS 2026, 19 November) and asks for a revision of at most 300 words by 31 October noon, with firmer methodology, conclusions aligned with findings, and less promotional language. This is that version (299 words).

Background: EMS training relies on simulation. Instructors transcribe radio audio by hand for debriefing, and field trainees document from memory. Cloud transcription raises data-handling questions, and its models can state facts never said. We evaluated a local device that converts the radio report into a structured handoff without a network.

Methods: We built an offline pipeline on 8 GB NVIDIA Jetson devices: faster-whisper transcription with a radio-phrase prompt, re-decoding of collapsed decoder windows and a medical-lexicon corrector; a 4-billion-parameter language model producing a 25-field handoff report under rules that leave unstated facts null; and a patient-state layer applying published prehospital criteria. We compared it with bare-model and cloud baselines (Groq Whisper, gpt-oss-120b) on 1,000 generated radio clips (200 scripted reports, one synthetic voice, five noise levels; development and held-out splits) and 22 real public-feed calls (51 bursts). Outcomes: word error rate, medical-term recall, weighted report score against gold, fabricated facts per report; paired bootstrap intervals.

Results: Local word error rate was 4.6%/5.5% (clean/5 dB) on a second device's GPU versus 3.9%/4.9% for the best cloud model; held-out 4.1%/5.7% versus cloud turbo 5.2%/6.5%. Term recall rose from 87.1% to 90.5% (P=0.99). On real calls, rules reduced unsupported statements from 98 to 32 (73 reports) and invented allergy and code-status entries from 50 and 23 to zero; disabling them cost 5.4 to 5.8 points [3.8, 7.5]. Local report score was 96.6/95.3 versus cloud 96.8/95.1 (-0.1 [-2.4, +2.0]); held-out 94.6/93.6 versus 95.8/93.8 (-1.3 [-3.6, +1.2]). An average call (85 s) took 143 s offline, 70 to 75 s with the GPU. Human-verified real-call transcripts are pending.

Conclusions: An offline edge device produced handoff reports within about one point of a cloud pipeline on a generated benchmark, with fabrication-prone fields protected by rules. Pilot limits: one synthetic voice, real-call accuracy awaiting verified references.
