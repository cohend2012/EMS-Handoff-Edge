# What we compare and how every score is computed

Written 2026-10-30 from the scorer code (`scripts/benchmark.py`, `scripts/corpus/score_callsets.py`,
`scripts/corpus/eval_set.py`, `scripts/corpus/posterior.py`). Every number in the reports is one of the
quantities defined here, computed on one of the data sets defined here. Nothing else.

## 1. Two stages, scored separately

The pipeline has two stages and each is scored on its own output.

| stage | input | output | scored against |
|---|---|---|---|
| transcript | audio of one radio burst or clip | text | a reference text |
| report | the text from a named transcript run | a 25-field JSON handoff report | a gold list of facts |

A "system" row (local or cloud) is a transcript run followed by a report run over that run's transcripts.

## 2. The data sets

**Generated set, test split.** 30 scripted radio reports (written by `gen_scripts.py` from templates with
known facts), read by one synthetic voice (Piper), passed through a filter that matches the real feed's
band limit, and mixed with noise at five levels: clean (no added noise), 20, 15, 10 and 5 dB signal to noise.
That is 30 clips per level, 150 clips, 1,257 reference words per level. The reference text is the script
itself, so it is exact. Transcript rows are reported per level; report rows were run at clean and 5 dB only
(60 reports per configuration).

**Real calls.** 22 call sets recorded from a public county EMS feed, 51 bursts, 1,869 s of audio. Three call
sets (CS001, CS008, CS014), nine bursts, 612 words, have a reference transcript drafted by the assistant from several transcriber outputs read side by side (Groq turbo with two hints, Groq large-v3, local medium, Vosk), with every judgement call marked "verify" in corrections.md, not from a person listening to the audio; those nine bursts are the only real audio with a transcript score. Those three call sets also have a
gold report; the other 19 do not, so real-call report rows (the 73-report corpus runs, the 22-call state
replay) measure invented facts and documentation gaps against the transcript, not field accuracy against gold.

## 3. Transcript scores

**Normalisation first.** Both texts are lower-cased, punctuation is stripped, hyphens split words, "%" becomes
"percent", "/" becomes "over", number words become digits, spoken units become their written forms ("324
milligrams" = "324 mg"), and spelled abbreviations are joined ("E T A" = "eta"). So a transcript is not
penalised for writing "BP 120/80" where the script said "BP 120 over 80".

**WER (word error rate).** Levenshtein distance between the two normalised word sequences (one insertion,
deletion or substitution = one edit) divided by the number of reference words. A row's WER is pooled: all
edits over all reference words at that level. "4.6 % at clean" means 58 edits in 1,257 words. Lower is better.

**Term recall and MTER.** `terms.txt` lists the medical terms we care about (drugs, vitals, scales,
findings). Every occurrence of a listed term in the reference is one trial; the transcript either contains it
(after normalisation) or lacks it. MTER = lacking / total; term recall = 1 − MTER. There are 200 term
occurrences per level in the test split. "90.5 % term recall" means 181 of 200 term occurrences were present.

**n-gram F.** The harmonic mean of 1- to 4-gram precision and recall between the two texts; a secondary
column, reported but not used for decisions.

## 4. Report scores

**Gold.** For each script, a list of 23 fields. Each field has a path in the report ("vital_signs.heart_rate"),
a kind, an expected value and a weight (3 for alert type, alert level and chief complaint; 2 for ETA, age,
sex, vitals; 1 for the rest). For a fact the script did not state, the gold field has kind "unknown": the
right answer is null. For the real gold cases the fields were written by hand from the reference transcript.

**Five buckets.** Each gold field lands in exactly one:

| bucket | meaning | counts as |
|---|---|---|
| hit | the gold states a fact and the report has it (enum: an accepted value; int: equal; contains: one of the expected phrases is in the field, or in the fallback fields the gold names; list: every expected item present) | good |
| omission | the gold states a fact and the report left the field null or "not stated" | bad |
| wrong | the gold states a fact and the report has a different value | bad |
| abstained | the gold says the fact was never stated and the report left it null | good |
| hallucination | the gold says the fact was never stated and the report filled it; or a history list carries an item the gold does not; or a forbidden phrase appears | bad |

**weighted** = 100 × (sum of weights of hit and abstained fields) / (sum of all weights). "96.6" means 96.6 %
of the weight landed in hit or abstained. Abstaining on an unstated fact scores as well as getting a stated
fact right, by design: the product's safety claim is that it does not invent.

**recall** = hit / (hit + omission + wrong): of the facts the gold states, the share the report got right.

**precision** = hit / (hit + wrong + hallucination): of the facts the report asserts, the share that are right.

**hallucinations per report** = the mean count of hallucination buckets per report.

**alert type right** = alert_type equals the gold's expected type or one it accepts (a chest-pain script
expects MEDICAL and accepts CARDIAC). **level right** = alert_level matches the priority the script states (1
RED, 2 YELLOW, 3 GREEN). **ETA right** = eta_minutes equals the stated ETA.

All report rows are the mean over the 30 scripts at a level (or 29 when one report failed).

## 5. The head-to-head that the headline quotes

"Local 96.6 / 95.3 against cloud 96.8 / 95.1" is the mean weighted report score at clean / 5 dB over the same
30 scripts for:

| | transcript | writer |
|---|---|---|
| local | whisper large-v3, fp16, on the cohendynamics GPU, with the radio hint, the span re-decode and the lexicon corrector | qwen3:4b-instruct (Ollama, vox-ai GPU), grounding rules on, lexicon retrieval on |
| cloud | Groq whisper-large-v3-turbo with the radio hint and the lexicon corrector | openai/gpt-oss-120b (Groq), the same prompt, rules on, lexicon retrieval on |

Two things to state plainly. First, the cloud side's transcriber in that row is **turbo**, not large-v3: the
cloud large-v3 transcripts (WER 3.9 / 4.9 %) were never run through the 120B writer, so the cloud report row
may be a little below the best the cloud could do. Second, the writers differ in size (4B against 120B) by
design; that is the comparison the project is about.

## 6. What the intervals and probabilities mean

**Paired bootstrap interval.** For each of the 30 scripts, the difference in score between the two
configurations on that same script (B minus A). Resample the 30 differences with replacement 10,000 times and
take the mean each time; the interval is the 2.5th to 97.5th percentile of those means. "−0.1 [−2.4, +2.0]"
means: on average the local report scored 0.1 points below the cloud report on the same script, and with 30
scripts the data are consistent with anything from 2.4 below to 2.0 above. An interval that includes zero
means we cannot say which is better at this sample size; it does not mean they are equal.

**P(B > A).** A Beta-Binomial posterior on per-trial counts (term occurrences, or stated facts), flat prior.
It treats trials as independent, which they are not (one collapsed window loses many terms at once), so it
overstates certainty; the bootstrap is the conservative reading and is the one to quote.

## 7. What these numbers are not

- Not clinical accuracy judged by a clinician. Gold for the generated set is the script's own facts; gold for
  three real calls were drafted by the assistant from several transcriber outputs, not by a person listening; human verification is pending.
- Not a measure on the 22 real calls' field accuracy: 19 of them have no gold report.
- Not independent of the transcriber: a report row's number belongs to the transcript run it read.
- Not comparable across hint versions: rows before 2026-10-27 used an older whisper hint.
- The "4 of 5 asks right" in the state replay was judged by reading the transcripts, not by a clinician.
- The 143 s and 70 to 75 s latency figures are one call (CS008, 84.5 s of audio, the average length) through
  the headless path; the live app adds the quiet-period wait (25 s offline) before the writer starts.
