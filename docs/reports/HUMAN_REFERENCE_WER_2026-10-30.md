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
| gpu_large_v3_ours (cohendynamics GPU, large-v3 fp16, our stack, run 2026-10-30 19:10) | 22 | 1489 | 240 | 16.1 % | not bootstrapped |
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
| gpu_large_v3_ours minus cloud_base (local GPU against the whisper-flow start) | -2.0 | -8.2 to +2.2 | 0.77 |
| gpu_large_v3_ours minus cloud_large_ours | +1.4 | +0.1 to +2.7 | 0.01 |
| gpu_large_v3_ours minus local_ours (GPU large-v3 against CPU medium) | -4.6 | -8.5 to -1.5 | 1.00 |

What this changes:

1. Real radio WER is 15 to 21 %, about four times the synthetic-set level (4.6 % on the test split). The synthetic
   number must not be presented as real-call accuracy.
2. The earlier real-audio claim "the stack reduced WER from 25.7 % to 22.7 %" (assistant-drafted references, 612 words)
   does not hold against human references. The stack's effect on real audio is -0.3 points, interval spanning zero.
3. The best real-audio transcript is large-v3. Cloud large-v3 is 14.7 %. Our local GPU large-v3 path is 16.1 %, 1.4
   points behind cloud large-v3, 2.0 points ahead of the turbo starting point (interval spans zero) and 4.6 points ahead
   of the CPU medium path. On real audio the GPU Jetson is what makes the local system competitive.
4. The assistant consensus drafts were 15 % wrong. They could never have stood in for a listener.

Lois's observations from listening (her words, summarised): openings are often cut off because the medic talks before
the channel opens; accents and tired, slurred speech defeat every model; base-hospital names and unit numbers are
misheard most; shorthand such as "duoneb" is transcribed literally; speaker labels (base vs medic) would help review.

Limits: one listener, no second-listener agreement yet; base-hospital and unit names she could not make out are
written as she heard them; 12 call sets and 2 bursts inside these 10 calls are not read yet.

## 2026-10-31 01:00: live decode settings and a same-model vote (22 bursts)

Constraint: everything must run live. The cohendynamics GPU holds one large-v3 (about 600 MB left free), so no second
model; vox-ai's 4.7 GB "free" exists only while Ollama has unloaded the writer. The server now takes per-request decode
options on the one loaded model (scripts/whisper_server.py: ?hint=0, ?beam=N, ?vad=0, ?pad=S; commit ee80222), and each
of Lois's 22 bursts was decoded nine ways through it. Script and raw output in the session scratchpad (sweep.py,
sweep.json); scoring as above (pooled WER, paired burst bootstrap against the live decode).

| setting | WER | vs live | 95 % interval | GPU s / burst |
|---|---|---|---|---|
| live (hint, greedy, VAD) | 16.6 % | | | 6.4 |
| no hint | 16.0 % | -0.6 | -1.8 to +0.7 | 6.0 |
| beam 5 | 15.4 % | -1.2 | -2.7 to +0.6 | 7.8 |
| beam 5, no hint | 16.7 % | +0.1 | -1.8 to +2.0 | 7.0 |
| VAD off | 22.6 % | +6.2 | -0.9 to +20.4 | 7.5 |
| 0.5 s padding | 16.4 % | -0.2 | -1.8 to +1.1 | 6.5 |
| VAD off + padding | 16.0 % | -0.6 | -2.2 to +0.7 | 6.4 |
| band-pass 250-3800 Hz + normalise | 19.3 % | +2.8 | +1.2 to +5.0 | 6.4 |
| band-pass + beam 5 | 17.7 % | +1.2 | -0.2 to +2.9 | 7.9 |

ROVER vote of three decodes (alignment to the live decode, majority per slot):
- named before the sweep, live + no hint + beam 5: **15.6 %, -1.0 [-1.6, -0.4]**, 20 s of GPU per burst (an average burst
  is 36.6 s, so it runs in real time);
- best of the 36 trios, live + beam-5-no-hint + VAD-off-padded: 15.0 %, -1.6 [-2.9, -0.6]; picked from 36 on the same
  bursts, so it is not quoted as a result until it holds on unread bursts.

The live decode scored 16.6 % here against 16.1 % in the 2026-10-30 19:10 run of the same server on the same bursts; the
cause is not yet found (all rows in this table come from the same run). For comparison, a vote over all nine saved
transcripts including the cloud ones reaches 13.5 %, and a local vote with Whisper medium is worse than large-v3 alone
(+0.9). Band-pass filtering hurts and is dropped.
