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
| "calls were processed in 143 seconds without cell service" | the offline run of CS008 (84.5 s of audio): whisper 66 s, writer 74 s, 143 s wall, every outbound connection blocked | correct for one call on one device; the average real call is 85 s of audio, so CS008 is representative; with whisper on the second device's GPU the same call is about 85 to 90 s |
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
calls, three of which have human reference transcripts (612 words) and gold reports. We measured word error
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
100 % to 67 %. The local report score over the largest local model's transcripts was level with the cloud pipeline (96.6 / 95.3
against 96.8 / 95.1, difference -0.1 [-2.4, +2.0]). An average real call (85 s of audio) was processed in 143 s with every
outbound connection blocked, and in about 85 s with the transcript on the second device's GPU.

**Conclusion.** An edge device can produce a grounded handoff report from radio audio with no network,
within a point or two of a cloud pipeline on transcript and report accuracy, and with the two fields most
prone to fabrication protected by rules. It is slower than the cloud; the measured remedy is the device's
GPU, and a dedicated accelerator is the next step. These results are a feasibility study on a pilot sample
(three referenced calls, one synthetic voice) and the next phase adds references for the remaining calls,
a second annotator, and the state-aware pipeline that lets the device flag missing fields before the
transmission ends.
