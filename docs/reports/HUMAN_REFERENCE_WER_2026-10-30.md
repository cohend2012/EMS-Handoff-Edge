# Real-audio WER against human references (Lois Sayrs live reads, 2026-10-30)

First real-call accuracy measured against a human listener. Lois Sayrs listened to 10 call sets and wrote what she
heard per burst. 22 bursts, 1,489 reference words after normalisation, are scored here. The references stay on the
dev Jetson in ~/labels/human (one <CS>/<segment>.human.txt per burst, README.txt with the mapping and exclusions);
they are not committed to the repos. Her CS001 review confirmed the three existing CS001 references at 100 %.

Mapping notes: the unlabelled CS004 read is S038_01 and the CS009 read is S022_01. Her CS006 "S031_01" text is the
audio of S032_01 (she skipped the ETA confirmation burst), so it is scored as S032_01. CS006 S030_01 is excluded: her
read leaves out the words she could not hear, so it is partial. Her bracketed notes are removed from the references.

Scoring: scripts/corpus/score_callsets.py `wer` (Levenshtein edits over normalised words), pooled over bursts.
Interval: 95 % bootstrap over bursts.

| transcript (saved per burst) | bursts | ref words | edits | WER | 95 % interval |
|---|---|---|---|---|---|
| cloud_large_ours (Groq large-v3 + our stack) | 22 | 1489 | 219 | 14.7 % | 11.5 to 18.9 |
| Groq large-v3 raw | 22 | 1489 | 223 | 15.0 % | 12.3 to 18.4 |
| assistant consensus drafts | 18 | 1135 | 170 | 15.0 % | 11.9 to 19.3 |
| Groq turbo, radio prompt | 22 | 1489 | 255 | 17.1 % | 13.8 to 21.7 |
| cloud_base (Groq turbo, the whisper-flow start) | 22 | 1489 | 270 | 18.1 % | 12.1 to 28.7 |
| local_ours (local medium + our stack) | 22 | 1489 | 308 | 20.7 % | 17.6 to 25.1 |
| cloud_ours (Groq turbo + our stack) | 22 | 1489 | 309 | 20.8 % | 14.8 to 30.1 |
| local_base (local medium, no stack) | 22 | 1489 | 313 | 21.0 % | 17.8 to 26.0 |
| Groq turbo raw | 22 | 1489 | 334 | 22.4 % | 16.0 to 32.7 |
| Vosk small (live captions) | 22 | 1489 | 1016 | 68.2 % | 65.6 to 71.4 |

Paired differences (B minus A, points of WER, burst bootstrap):

| comparison | difference | 95 % interval | P(B better) |
|---|---|---|---|
| local_ours minus local_base (our stack on local medium) | -0.3 | -2.3 to +1.5 | 0.63 |
| cloud_ours minus cloud_base (our stack on turbo) | +2.6 | -7.0 to +13.0 | 0.29 |
| cloud_large_ours minus Groq large-v3 raw | -0.3 | -1.8 to +1.3 | 0.60 |
| local_ours minus cloud_large_ours | +6.0 | +2.3 to +10.5 | 0.00 |
| local_ours minus cloud_base | +2.6 | -5.4 to +9.5 | 0.25 |

What this changes:

1. Real radio WER is 15 to 21 %, about four times the synthetic-set level (4.6 % on the test split). The synthetic
   number must not be presented as real-call accuracy.
2. The earlier real-audio claim "the stack reduced WER from 25.7 % to 22.7 %" (assistant-drafted references, 612 words)
   does not hold against human references. The stack's effect on real audio is -0.3 points, interval spanning zero.
3. The best real-audio transcript is large-v3. The local CPU medium path is 6.0 points worse. The GPU large-v3 path on
   cohendynamics has no saved real-call transcripts yet; it is the next row to run on these 22 bursts.
4. The assistant consensus drafts were 15 % wrong. They could never have stood in for a listener.

Lois's observations from listening (her words, summarised): openings are often cut off because the medic talks before
the channel opens; accents and tired, slurred speech defeat every model; base-hospital names and unit numbers are
misheard most; shorthand such as "duoneb" is transcribed literally; speaker labels (base vs medic) would help review.

Limits: one listener, no second-listener agreement yet; base-hospital and unit names she could not make out are
written as she heard them; 12 call sets and 2 bursts inside these 10 calls are not read yet.
