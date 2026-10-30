# What the system does against the base models, locally and in the cloud (2026-10-28)

Written to answer an abstract draft ("From Classroom to Cabin", 279 words) whose methods and numbers do not
come from this repository: twenty recordings, four configurations, cosine similarity against the cloud output,
RTF and TTFT, a Beta-Binomial posterior, a disconnection test, recall 88.0 to 92.0 %, precision 83.68 to 90.64 %,
and "both edge models outperformed the cloud control". None of those figures appear in the repo, its history, or
the Jetson's folders. The interns' tree (last commit 2026-09-09) holds no dataset, no references, no scorer and
no evaluation harness; its only comparison prints a local and a cloud transcript side by side with wall time.
Everything below was measured after 2026-10-18, with one scorer, on clips and calls that exist on disk.

Every local row is in. The cloud writer over the base cloud transcripts (60 of 60) landed on 2026-10-29; the cloud
rules-off row reached 13 of 60 before the Groq daily token cap stopped it and resumes when the cap resets (its
partial numbers are marked as such). Logs: `~/datasets/synth_v1/logs/`.

## 1. How we score

- **WER**, word error rate: Levenshtein edits between the transcript and a human reference (real calls) or the
  script (generated set), divided by reference words. A "point" is one percentage point, one wrong word per
  hundred spoken. `scripts/corpus/score_callsets.py`, with the spoken-unit normalisation (ETA, IV, BP, SpO2 ...).
- **MTER**, medical term error rate: the share of the 200 occurrences of the 149 terms of `terms.txt` in the 30
  test scripts that the transcript lacks. Generated set only (real calls have no term list yet).
- **n-gram F**: for n = 1 to 4, precision and recall of the transcript's n-word runs against the reference; the
  table shows the mean F over n and the 4-gram recall (whole phrases intact, what the report model reads).
  There is no 5-gram. `scripts/corpus/ngram_scores.py`.
- **Weighted field score**: the report against gold JSON over 25 clinical fields with weights; a correct value,
  or a correct null when nothing was said, earns the weight; an omission, a wrong value or a hallucination earns
  nothing; out of 100. Recall = hits over stated facts; precision = hits over hits + wrong + hallucinated. These
  are defined against human gold, never against the cloud output. `scripts/benchmark.py`.

"Base" below is the model alone: no prompt, no re-decode, no corrector, and (for the report stage) the grounding
rules off. "Ours" is the app's stack: the radio-phrase hint, the gap re-decode (local only), the transcript
corrector, the extraction prompt with lexicon context, and the grounding rules after the model.

## 2. Transcriber, generated set (150 clips, 30 scripts x 5 SNRs, Piper voice through the radio channel)

From SYNTH_V1_2026-10-27.md, one scorer over every row. Local = faster-whisper int8 on the Jetson CPU.

| transcriber | stack | WER clean | WER 5 dB | MTER clean | MTER 5 dB | n-gram F clean | n-gram F 5 dB | speed |
|---|---|---|---|---|---|---|---|---|
| cloud whisper-large-v3 | base (no prompt) | 4.6 % | 6.5 % | 7.5 % | 12.5 % | 0.924 | 0.880 | |
| cloud whisper-large-v3 | hint | 3.9 % | 4.9 % | 6.5 % | 8.5 % | 0.935 | 0.912 | |
| cloud large-v3-turbo | base (no prompt) | 5.3 % | 7.5 % | 9.0 % | 13.5 % | 0.914 | 0.866 | |
| cloud large-v3-turbo | hint | 4.3 % | 6.4 % | 10.0 % | 14.0 % | 0.927 | 0.891 | |
| cloud large-v3-turbo | hint + corrector (ours, cloud) | 4.2 % | 6.4 % | 8.5 % | 13.0 % | 0.929 | 0.892 | |
| local medium | base (no prompt) | 5.8 % | 8.1 % | 10.5 % | 14.0 % | 0.902 | 0.850 | 0.8x |
| local medium | hint | 4.8 % | 7.0 % | 10.0 % | 13.5 % | 0.915 | 0.876 | 0.85x |
| local medium | hint + corrector | 4.8 % | 6.9 % | 9.0 % | 13.0 % | 0.916 | 0.876 | 0.85x |
| local medium | hint + gap re-decode * | 5.9 % | 7.0 % | 10.5 % | 13.5 % | 0.912 | 0.876 | 0.85x |
| local medium | hint + gap re-decode + corrector (ours, local) * | 5.8 % | 6.9 % | 9.5 % | 13.0 % | 0.913 | 0.876 | 0.85x |
| local small | hint | 6.2 % | 9.6 % | 17.5 % | 17.0 % | 0.885 | 0.841 | 0.33x |
| local distil-large-v3 (10-29) | hint | 9.4 % | 10.5 % | 17.5 % | 20.0 % | 0.850 | 0.834 | 0.72x |
| local large-v3-turbo, float32 instead of int8 (10-29) | hint | 9.2 % | 11.0 % | 15.5 % | 16.0 % | 0.885 | 0.839 | 1.65x |
| local large-v3-turbo, int8, beam 5 instead of greedy (10-29) | hint | 6.2 % | 8.1 % | 13.0 % | 15.0 % | 0.908 | 0.859 | 0.81x |
| local distil-large-v3.5 (10-29) | hint | 8.8 % | 12.6 % | 15.5 % | 21.0 % | 0.859 | 0.806 | 0.71x |
| local large-v3-turbo, int8, greedy, VAD filter off (10-29) | hint | 12.8 % | 10.7 % | 21.5 % | 16.0 % | 0.881 | 0.849 | 0.75x |
| local medium, int8, beam 5 instead of greedy (10-29) | hint | 7.1 % | 6.9 % | 13.0 % | 14.0 % | 0.908 | 0.885 | 1.2x |
| local large-v3-turbo, int8, greedy + gap re-decode with beam 5 on the span (10-29) | hint | 10.8 % | 10.8 % | 18.0 % | 16.0 % | 0.889 | 0.840 | 0.77x |
| local medium, int8, greedy + gap re-decode with beam 5 on the span (10-29) | hint | 5.9 % | 7.0 % | 10.5 % | 13.5 % | 0.912 | 0.876 | 0.85x |
| local large-v3-turbo, int8, greedy + re-decode, word-rate trigger, no lead-in (10-29) | hint | 9.1 % | 8.5 % | 18.0 % | 16.0 % | 0.893 | 0.847 | 0.85x |
| local medium, int8, greedy + re-decode, word-rate trigger, 3 s lead-in (10-29) | hint | 5.3 % | 7.0 % | 10.0 % | 13.5 % | 0.914 | 0.876 | 0.85x |
| local large-v3-turbo, int8, greedy + re-decode, word-rate trigger, 3 s lead-in (10-29) | hint | 6.7 % | 8.5 % | 11.5 % | 16.0 % | 0.907 | 0.847 | 0.83x |
| local large-v3-turbo, int8, greedy + re-decode, every 10-29 fix (trigger, trailing speech, lead-in, seam skip) | hint | 6.0 % | 8.5 % | 11.5 % | 16.0 % | 0.910 | 0.847 | 0.79x |
| local medium, int8, greedy + re-decode, every 10-29 fix (the shipped default) | hint | 5.0 % | 7.0 % | 10.0 % | 13.5 % | 0.915 | 0.876 | 0.88x |
| local medium, float16 on the cohendynamics GPU (container), greedy + re-decode, every 10-29 fix | hint | 4.6 % | 7.3 % | 9.0 % | 13.5 % | 0.925 | 0.876 | 0.13x (5.3 s per clip) |
| local large-v3-turbo, float16 on the cohendynamics GPU (container), greedy + re-decode, every 10-29 fix | hint | 6.8 % | 8.5 % | 11.5 % | 16.0 % | 0.897 | 0.855 | 0.10x (4.1 s per clip) |
| local medium, float16 on the cohendynamics GPU, greedy + re-decode, every 10-29 fix + corrector (ours, local, GPU) | hint + corrector | 4.5 % | 7.1 % | 8.0 % | 12.0 % | 0.926 | 0.877 | 0.13x |
| local large-v3, float16 on the cohendynamics GPU (container), greedy + re-decode, every 10-29 fix | hint | 4.6 % | 5.5 % | 9.0 % | 9.0 % | 0.925 | 0.902 | 0.20x (8.4 s per clip) |
| local large-v3, float16 on the cohendynamics GPU, every 10-29 fix + corrector (ours, local, GPU, largest) | hint + corrector | 4.6 % | 5.5 % | 8.5 % | 8.5 % | 0.925 | 0.903 | 0.20x |
| local medium, float16 on the cohendynamics GPU, plain decode (base: no hint, no re-decode, no corrector) | base (no prompt) | 7.5 % | 8.3 % | 13.0 % | 14.5 % | 0.892 | 0.856 | 0.12x (4.9 s per clip) |
| local large-v3, float16 on the cohendynamics GPU, plain decode (base) | base (no prompt) | 4.5 % | 6.3 % | 7.5 % | 12.5 % | 0.924 | 0.881 | 0.20x (8.3 s per clip) |
| local large-v3, GPU fp16, every 10-29 fix, hint trimmed to 220 whisper tokens (opening sentence kept) | hint (fit) | 4.7 % | 5.5 % | 8.5 % | 9.0 % | 0.925 | 0.903 | 0.20x |
| local large-v3, GPU fp16, every 10-29 fix, hint = opening sentence + radio phrases only (94 tokens) | hint (short) | 4.8 % | 6.1 % | 7.5 % | 10.0 % | 0.920 | 0.890 | 0.20x |
| local medium, GPU fp16, every 10-29 fix, hint trimmed to 220 tokens | hint (fit) | 4.6 % | 7.0 % | 9.0 % | 13.5 % | 0.925 | 0.881 | 0.12x |
| local medium, GPU fp16, every 10-29 fix, hint = sentence + radio phrases only | hint (short) | 5.5 % | 7.9 % | 9.5 % | 13.0 % | 0.910 | 0.864 | 0.12x |
| local large-v3-turbo (the app default until today) | hint | 12.8 % | 10.8 % | 21.5 % | 16.0 % | 0.881 | 0.840 | 0.8x |
| local large-v3-turbo | hint + gap re-decode, before the loop guard | 13.4 % | 10.8 % | 21.5 % | 16.0 % | 0.875 | 0.840 | 0.8x |
| local large-v3-turbo | hint + gap re-decode, with the loop guard (shipped) | 12.6 % | 10.8 % | 21.5 % | 16.0 % | 0.881 | 0.840 | 0.8x |
| local tiny.en | hint | 13.3 % | 33.3 % | 26.0 % | 42.5 % | 0.823 | 0.624 | 0.1x |

GPU base rows (10-29, late): the GPU stack now has its own base, the same fp16 decode with no hint, no re-decode and
no corrector. For medium the stack is worth -2.8 points of WER at clean [-5.9, -0.9] and -1.1 at 5 dB [-2.2, -0.1],
and term recall rises from 86.2 to 88.8 % pooled over both rows (P(stack > base) = 0.857). For large-v3 the stack is
worth nothing at clean (-0.1 [-0.8, +0.6]) and -1.0 at 5 dB [-1.8, -0.2], term recall 90.0 to 91.0 % (P = 0.683):
the largest model needs the additions least, and what it gains it gains under noise. Two cross-device pairs: the
plain fp16 large-v3 on the Jetson GPU equals the plain cloud whisper-large-v3 to the clip (+0.0 [-0.2, +0.3] at
clean, -0.2 [-0.6, +0.1] at 5 dB, term recall identical at 92.5 and 87.5 %), so the cloud's remaining lead
(3.9 / 4.9 against our 4.6 / 5.5 %) is in how the endpoint uses the prompt, not in the model or the hardware; and
plain fp16 medium on the GPU is 1.4 points worse at clean than plain int8 medium on the CPU [-0.2, +4.2] and 0.6
better at 5 dB [-2.3, +0.6], the fp16 and int8 decoders of one model being different decoders, as the real bursts
also showed.

Hint ablation on the GPU (10-29, late): the app's hint is 780 characters, 243 whisper tokens, and faster-whisper keeps
only the last 223 tokens of a prompt, so the opening sentence ("The following is a paramedic radio report.
Terminology: B") never reaches the local decoder; the Groq endpoint may keep all of it. Two variants, same decode
otherwise: the hint trimmed to 220 tokens so the sentence survives, and the sentence plus the radio phrases alone
with no terminology list. The trimmed hint changes nothing (medium +0.0 [-0.4, +0.5] at clean and -0.3 [-0.9, +0.3]
at 5 dB; large-v3 4.7 / 5.5 against 4.6 / 5.5 %), so the sentence is worth nothing and the 20 terms it displaces are
worth nothing either. The short hint is worse: medium +1.0 point of WER at clean [+0.2, +1.8] and +0.8 at 5 dB
[-0.1, +1.6], large-v3 +0.4 [-0.4, +1.2] and +0.6 [-0.2, +1.3], term recall unchanged within a point. So the
terminology list is what the hint is for, and the cloud's lead over the same model (3.9 / 4.9 against 4.6 / 5.5 %)
is not explained by the token cut. What remains untested is the Groq endpoint's own prompt handling (whether it
keeps the whole 243 tokens, and whether it applies the prompt to every 30 s window or only the first), and a
family-specific term list after the first burst, which is the dynamic-hint step of the stateful plan.

Candidate rows added on 2026-10-29 (`HORSEPOWER_GOALS.md` lever C): distil-large-v3 with the hint is uniformly
worse than medium (median clip WER 9.8 % at clean, no tail drops or loops) and only 15 % faster on the CPU
(30 s against 35 s per 42 s clip), so it is out. Turbo in float32 rescues two of the four tail-drop clips
(SY074 46 to 8 %, SY078 40 to 5 %) but not the other two (SY055 44 %, SY188 68 %) and runs 2.3 times slower
(69 s per clip), so quantization is part of the turbo failure, not all of it. Beam search (beam 5, still int8)
fixes every one of the tail-drop clips (SY055 44 to 6 %, SY074 46 to 7 %, SY078 40 to 6 %, SY184 40 to 7 % at clean;
SY188 68 to 12 % at 5 dB) at the same speed as greedy (34 s against 33 s per clip), so the turbo failure was greedy
decoding on a quantized model, not the model: turbo with beam 5 scores 6.2 / 8.1 %, median clip WER 5.4 % at
clean, no tail drops. That is still behind medium with the hint (4.8 / 7.0 %), and the two differ by 1.4 and 1.1
points on 30 clips each, inside the noise of this set, so beam 5 makes turbo usable, not better. distil-large-v3.5
(8.8 / 12.6 %, 0.71x) is out like distil-large-v3. Turning the VAD filter off changes nothing
(12.8 / 10.7 %, the same four tail drops), so the filter is not the cause either. Medium with beam 5 scores 7.1 / 6.9 %
at 1.4 times the time (50 s against 35 s per clip). The 4.8 % greedy row above is the 10-27 run with the old
hint; with today's hint greedy medium scores 7.1 % clean (the 10-27 fallback run, where the fallback changed
nothing), and it loses the 28 to 58 s window of SY078 exactly as beam 5 does (both 68 words of 139, decoded
again on 10-29 to check). So beam 5 on medium is a wash against the same-hint greedy row, and the SY078
collapse is the hint's doing, not the beam's. Decision: medium stays greedy as the default.

What the failing clips look like, decoded again on 10-29 with the segment times printed: on SY055, SY074 and
SY184 greedy turbo returns the first 30 s window whole (62 to 64 words) and the second window as one segment
spanning 30 to 60 s with two words ("Thank you.", "The End"); with beam 5 the second window comes back as
four or five segments and 46 to 49 words. That is the collapsed-window shape the gap re-decode already
detects (a segment over 12 s with two words or fewer), and the turbo "hint + gap re-decode" rows above did not
rescue them because the span decoded alone, greedily, looped and the loop guard discarded it. The medium SY078
collapse is the other shape the re-decode detects: a 30 s wordless gap at speech level. So the fix is not a
beam on the whole clip but a beam on the re-decoded span: `LOCAL_WHISPER_REDECODE_BEAM` (default 5) added
10-29, the gap re-decode now decodes the span with beam 5 while the clip stays greedy. The first
measurement (`large-v3-turbo_hint_rd5`) gives turbo 10.8 / 10.8 %: it rescued SY074 (46 to 6 %) and nothing
else. The two misses are instructive. SY055 and SY184 collapsed again, but this time to five and eight words for
30 s ("We'll see you next time."), above the detector's two-word limit; the same clips gave two words in the
10-29 diagnostic run, so the int8 CPU decode is not repeatable word for word and a count threshold is fragile.
SY078 and SY188 are the opposite shape: a 30 s window that repeats itself ("blood pressure 93 over 48, rate of
127 ..." three times, 108 words; "13." sixty times, 122 words), which no gap rule sees. Both are word-rate
anomalies on a long segment, so the trigger is now a rate band on any segment of 12 s or more: under 0.5 words
a second is a collapsed window, over 3.2 is a repeating one (the reports run 2.0 to 2.5), and a repeating span's
re-decode is spliced in only if it comes back shorter. The rows `large-v3-turbo_hint_rd5b` and
`medium_hint_rd5b` (queued) measure that; `medium_hint_rd5` measures the beam 5 span decode alone on medium: 5.9 / 7.0 %, the same as the
greedy-span re-decode of 10-27 (`medium_hint_rd`) clip for clip, at the same time per clip (36 s). Its one
failure, the 28 to 58 s gap of SY078, re-decodes to the same 109 words either way (45 to 21 % WER): the span
decoded alone still drops its first 12 s ("temp of" runs straight into "lactate is 2.8", losing the glucose,
GCS and lung sounds), so on medium the beam changes nothing and the remaining loss is the span's start, not
its decode. Decoding that span seven ways on 10-29 (scratch `span_diag.py`) found the cause: alone with the
hint it comes back as 42 words starting at "lactate is 2.8" (the first 20 s gone), with the temperature
fallback the same, and the first half of the span alone gives three words; without the hint it comes back
whole (66 words), and with 3 s of lead-in and the hint it comes back whole (73 words, with the previous
segment's last words repeated). A span that starts mid-sentence and the hint together make medium skip to
the first phrase that matches the hint's vocabulary. So the re-decode now starts 3 s before the gap and
trims the words that repeat the previous segment's tail (`LEAD_IN_SECONDS`, `_trim_seam`); `medium_hint_rd5b`
and `large-v3-turbo_hint_rd5c` (queued) measure the word-rate trigger with the lead-in; `large-v3-turbo_hint_rd5b`
is the trigger without it.

Results of those two (18:05): the word-rate trigger on turbo, without the lead-in, gives 9.1 / 8.5 %: both
repeating windows are fixed (SY078 40 to 6 %, SY188 68 to 12 %), SY074 stays fixed, and SY055 and SY184 are
still collapsed (44 and 38 %), which the per-clip diagnostic below explains. Medium with the trigger and the
lead-in gives 5.3 / 7.0 %, the best local medium row with today's hint (greedy 7.1, old re-decode 5.9): SY078
goes 45 to 11 % (148 words for 139, the seam repeats a few words), and no clip got worse.

Two more fixes from those rows (18:10). The seam: the lead-in re-said "24% on 4 liters, nasal cannula, temp
of" against "94% on 4 liters nasal cannula temp of", one word off at the start, so the exact-prefix trim did
nothing; the trim now lets the matched run start up to three words in. The two turbo clips: decoded again with
the lead-in code both come back whole (118 and 120 words, correct endings), and the eval's miss came from the
first pass returning a short segment with the rest of the clip left wordless, which the rules skipped as
trailing silence; speech after the last worded segment (6 s or more at speech level) is now a gap like any
other. `large-v3-turbo_hint_rd5c` (lead-in without these two) gives turbo 6.7 / 8.5 % at greedy speed (35 s against
33 s per clip): every one of the five failing clips is rescued (SY055 44 to 11 %, SY074 46 to 9 %, SY078 40 to
6 %, SY184 40 to 11 %; SY188 68 to 12 % at 5 dB), no clip is worse by more than two points, and the row sits
half a point behind whole-clip beam 5 (6.2 / 8.1 %) because the seam still repeats a few words on SY055 and
SY184 (124 and 129 words for 117). With every fix (`large-v3-turbo_hint_rd5d`) turbo is 6.0 / 8.5 % at 34 s per
clip: on the CPU, greedy decoding plus the targeted re-decode now matches whole-clip beam 5 (6.2 / 8.1 %) at
greedy speed. Medium with every fix (`medium_hint_rd5d`, the shipped default) is 5.0 / 7.0 % at 37 s per clip:
the seam skip takes SY078 from 11 to 4 %, nothing else moves by more than two points, and with today's hint the
CPU medium path has gone from 7.1 % greedy to 5.0 % in one day of decode work, without touching the model.

The same decode on the cohendynamics GPU (the `dustynv/faster-whisper` image plus scipy, float16, the app's own
`eval_set.py` inside the container, 19:11 to 19:22): medium 4.6 / 7.3 % at 5.3 s per clip (160 s for 1257 s of
audio, RTF 0.127) and turbo 6.8 / 8.5 % at 4.1 s per clip. Medium in float16 on the GPU is therefore both the
most accurate local transcriber measured with today's hint (int8 on the CPU with the same fixes: 5.3 % clean on
the lead-in row) and 6.6 times faster than the CPU. Against the cloud: cloud turbo with our stack is 4.2 / 6.4 %
and cloud large-v3 with our stack 3.9 / 4.9 %, so the local GPU medium is within half a point at clean and a
point at 5 dB before the corrector, at 5 s a burst against the cloud's 2 to 3. With the corrector (the full local
transcript stack, `medium_gpu_fp16_hint_rd5d_corr`) it is 4.5 / 7.1 %: 0.3 behind cloud turbo with our stack at
clean and 0.7 behind at 5 dB, on 30 clips a side. large-v3 in float16 on the GPU (lever D, `large-v3_gpu_fp16_hint_rd5d`, 3.1 GB of weights, 1.5 GB free
beside it): 4.6 / 5.5 % at 8.4 s per clip, the same as medium at clean (11 clips better, 12 worse, 7 tied) and
1.8 points better at 5 dB (15 better, 5 worse, 10 tied; SY071 20 to 0 %, SY117 14 to 4 %), with half medium's
dropped tail. With the corrector it is 4.6 / 5.5 %, against cloud large-v3 with our stack at 3.9 / 4.9 %: the
largest local transcriber is now within a point of the largest cloud one on both rows, at 8 s a burst. The writer
rows over the GPU medium and GPU large-v3 transcripts are queued on the vox-ai GPU.

On the nine real bursts with a human reference (section 3), the same GPU medium decode, before the corrector,
first came out at 31.5 % against 22.7 % for the CPU medium stack and 16.2 % for cloud large-v3 with our stack,
at 4.0 s a burst against 27.4. All of the difference is one burst: on CS001 S044 the float16 decode looped
("Priority 1, protocol." fourteen times for the first 30 s, 57 % WER against the CPU's 22 %), and at 1.4 words a
second the loop was under the dense trigger. The other eight bursts match the CPU stack within a few points
(CS008 17 against 18 %, CS014 S015 28 against 21 % without the corrector). So the re-decode now also triggers on
the repeat collapser's verdict for any segment of 6 s or more, whatever its word rate, and when the span loops
again on its own the first pass is kept with its repeats collapsed rather than as the loop. Two more rounds were
needed: the GPU loop came as fourteen 2 s segments of the same three words, so a run of three or more identical
consecutive segments is now one loop span; a loop span's re-decode is judged by the loop check alone (its real
content, 63 words, is longer than the 42-word loop); and a clip whose first pass has more words than its length
allows (large-v3 gave "How old? 30." fourteen times for a 3.7 s burst) is decoded again whole. With all of that,
on the nine real bursts, every stack with the corrector (10-29, 19:50):

| set | bursts | CPU medium stack (vox-ai, int8) | cloud turbo + stack | cloud large-v3 + stack | GPU medium fp16 + stack | GPU large-v3 fp16 + stack |
|---|---|---|---|---|---|---|
| CS001 | 3 | 20.9 % | 16.6 % | 20.9 % | 27.0 % | 22.7 % |
| CS008 | 1 | 18.4 % | 11.2 % | 10.2 % | 16.8 % | 14.3 % |
| CS014 | 5 | 27.3 % | 33.2 % | 17.8 % | 32.0 % | 26.9 % |
| all | 9 (597 words) | 22.7 % (27.4 s per burst) | 21.7 % (0.9 s) | 16.2 % (1.1 s) | 25.8 % (4.4 s) | 21.7 % (7.0 s) |

On real radio audio the local GPU large-v3 now equals cloud turbo with our stack and beats the CPU medium stack by
a point, at 7 s a burst against 27; cloud large-v3 with our stack is still 5.5 points ahead, on 597 reference
words. GPU medium loses to the CPU int8 medium by 3 points on these nine (float16 and int8 make different
mistakes on the same audio; CS001 S044 is still 8 points worse after the loop fix), so on real calls large-v3 is
the GPU model to run. These are nine bursts; the term list and more references (section 8) are what would make
the real-call row decisive.

\* The medium re-decode rows were run with the day's revised hint, which by itself moved the clean row from 4.8 to
7.1 % (two clips collapsed the other way; SYNTH_V1_2026-10-27.md); against that same-hint run the re-decode is a
1.2-point gain made of one clip (SY078, 44.6 % to 20.9 %) with 149 of 150 clips identical. Read the local "ours"
row against "hint" at 5 dB (6.9 against 7.0 %, MTER 13.0 against 13.5 %), where the hint change does not interfere.

On large-v3-turbo the re-decode fired on four clips and made two of them worse: the span decoded alone looped
("ETA ten minutes" 25 times on SY074 at clean; "the end of the day" 37 times on SY051 at 20 dB, 250 words for a
28 s span). The re-decode now discards a span that comes back as a sentence loop or at over five words a second
and keeps the first pass (`LocalTranscriber.MAX_WORDS_PER_SECOND`, tested); the four clips were re-run under the
guard for the row above. The default local model is `medium` from today: 4.8 % against 12.8 % at clean, same CPU
speed, no collapse the re-decode cannot handle.

Levers measured and rejected, with numbers in SYNTH_V1_2026-10-27.md: the temperature fallback (recovers two
of four collapses on turbo, none on medium), hotword boosting with the term list (breaks medium, 32 % WER at
clean on clips that score 5 % plain).

## 3. Transcriber, real call-sets with a human reference (CS001, CS008, CS014; 9 bursts, 597 words)

`scripts/corpus/real_stack.py`, same scorer, pooled over 612 reference words (the earlier 597 counted differently).

Full tables in REAL_STACK_2026-10-28.md. Over all bursts:

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3 + hint + corrector | 9 | 612 | 16.2 % (99/612) | 0.769 | 0.67 |
| cloud large-v3-turbo + hint + corrector | 9 | 612 | 21.7 % (133/612) | 0.729 | 0.62 |
| local medium + hint + gap re-decode + corrector | 9 | 612 | 22.7 % (139/612) | 0.681 | 0.55 |
| local medium, base (no prompt) | 9 | 612 | 25.7 % (157/612) | 0.671 | 0.53 |
| cloud large-v3-turbo, base (no prompt) | 9 | 612 | 27.6 % (169/612) | 0.665 | 0.54 |

### CS001

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3-turbo + hint + corrector | 3 | 163 | 16.6 % (27/163) | 0.755 | 0.64 |
| cloud large-v3 + hint + corrector | 3 | 163 | 20.9 % (34/163) | 0.709 | 0.60 |
| local medium + hint + gap re-decode + corrector | 3 | 163 | 20.9 % (34/163) | 0.698 | 0.56 |
| local medium, base (no prompt) | 3 | 163 | 25.8 % (42/163) | 0.678 | 0.51 |
| cloud large-v3-turbo, base (no prompt) | 3 | 163 | 46.6 % (76/163) | 0.550 | 0.42 |

### CS008

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3 + hint + corrector | 1 | 196 | 10.2 % (20/196) | 0.862 | 0.79 |
| cloud large-v3-turbo + hint + corrector | 1 | 196 | 11.2 % (22/196) | 0.849 | 0.78 |
| cloud large-v3-turbo, base (no prompt) | 1 | 196 | 13.3 % (26/196) | 0.796 | 0.70 |
| local medium, base (no prompt) | 1 | 196 | 18.4 % (36/196) | 0.747 | 0.62 |
| local medium + hint + gap re-decode + corrector | 1 | 196 | 18.4 % (36/196) | 0.734 | 0.62 |

### CS014

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3 + hint + corrector | 5 | 253 | 17.8 % (45/253) | 0.735 | 0.62 |
| cloud large-v3-turbo, base (no prompt) | 5 | 253 | 26.5 % (67/253) | 0.638 | 0.50 |
| local medium + hint + gap re-decode + corrector | 5 | 253 | 27.3 % (69/253) | 0.630 | 0.50 |
| local medium, base (no prompt) | 5 | 253 | 31.2 % (79/253) | 0.608 | 0.48 |
| cloud large-v3-turbo + hint + corrector | 5 | 253 | 33.2 % (84/253) | 0.619 | 0.50 |

Reading: on real radio audio the stack is worth 3 to 6 points on either side (cloud turbo 27.6 to 21.7 %, local
medium 25.7 to 22.7 %), and the local stack lands within a point of the cloud stack on the same model class.
The base local model beats the base cloud turbo here (25.7 against 27.6 %), which is the one direction in which
"edge beat cloud" is true in our data, and it is the base-against-base row, not the stacks. The best row is the
full cloud model with the stack (16.2 %, 4-gram recall 0.67). CS014 is the hard call (five short bursts, two
voices); the cloud turbo stack is worst there (33.2 %) and large-v3 best (17.8 %), so model order is not stable
across calls at this sample size.

## 4. Report stage, generated set (30 scripts at clean and at 5 dB)

The same transcripts through the app's report pipeline, scored against the gold JSON. Local = qwen3:4b-instruct
(Q4_K_M, Ollama 0.33.0) on the GPU of the named Jetson: vox-ai (the development Orin Nano) or cohendynamics (the
second Orin Nano Super, same model blob, used so report runs could proceed while vox-ai's CPU ran whisper); cloud =
openai/gpt-oss-120b on Groq. All transcripts came from the vox-ai CPU (local rows) or Groq (cloud rows). Report runs
are compared at the one-point level only within one device.

| transcript in | report model | stack | reports | weighted clean | weighted 5 dB | recall clean / 5 dB | precision clean / 5 dB | halluc. per report | alert type right | level right | ETA right | extract s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| local medium, base | local qwen3:4b-instruct, cohendynamics GPU | rules on | 60 | 95.7 | 94.1 | 94.6 / 93.6 % | 96.4 / 96.2 % | 0.10 / 0.13 | 90 % | 100 % | 83 / 78 % | 50 |
| local medium, our stack | local qwen3:4b-instruct, cohendynamics GPU | rules on (ours, local) | 60 | 93.7 | 93.7 | 93.1 / 93.6 % | 95.7 / 95.9 % | 0.17 / 0.20 | 90 % | 100 % | 78 / 83 % | 51 |
| local medium, our stack | local qwen3:4b-instruct, cohendynamics GPU | rules off | 60 | 88.0 | 88.3 | 90.7 / 91.2 % | 87.4 / 88.4 % | 0.73 / 0.70 | 90 % | 67 % | 78 / 83 % | 51 |
| local medium, base | local qwen3:4b-instruct, cohendynamics GPU | rules on, second run of the same input | 60 | 95.7 | 94.1 | 94.6 / 93.6 % | 96.4 / 96.2 % | 0.10 / 0.13 | 90 % | 100 % | 83 / 78 % | 50 |
| cloud turbo, base | cloud gpt-oss-120b, Groq | rules on | 60 | 95.2 | 94.4 | 94.5 / 92.6 % | 97.4 / 95.3 % | 0.13 / 0.20 | 100 / 97 % | 100 % | 65 / 74 % | 3 |
| cloud turbo, our stack | cloud gpt-oss-120b, Groq | rules on (ours, cloud) | 60 | 96.8 | 95.1 | 96.4 / 93.2 % | 98.6 / 97.0 % | 0.03 / 0.07 | 100 / 97 % | 100 % | 91 / 87 % | 3 |
| cloud turbo, our stack | cloud gpt-oss-120b, Groq | rules off, partial: 13 of 60 before the daily cap | 13 | 93.6 | 90.2 | 93.3 / 90.5 % | 95.5 / 93.9 % | 0.14 / 0.33 | 100 % | 80 % | 100 % | 3 |
| script text (no audio) | local qwen3:4b-instruct, vox-ai GPU | rules on | 30 | 96.4 | | 98.4 % | 96.3 % | 0.43 | 93 % | 100 % | 100 % | 55 |
| local medium, base | local qwen3:8b (Q4_K_M), cohendynamics GPU + CPU (10-29) | rules on; 56 of 60 finished, 4 timed out at 420 s | 56 | 96.2 | 94.2 | 95.7 / 93.8 % | 96.3 / 95.8 % | 0.22 / 0.28 | 96 / 93 % | 100 % | 76 / 82 % | 232 / 186 |
| local medium, our stack | local qwen3:8b (Q4_K_M), cohendynamics GPU + CPU (10-29) | rules on; 50 of 60 finished, 10 timed out at 420 s | 50 | 95.6 | 94.8 | 95.4 / 94.2 % | 97.4 / 96.5 % | 0.20 / 0.20 | 100 / 96 % | 100 % | 90 / 81 % | 285 / 206 |
| GPU medium fp16, our stack (10-29, every decode fix, cohendynamics GPU transcripts) | local qwen3:4b-instruct, vox-ai GPU | rules on (ours, local, GPU whisper); 59 of 60, SY117 at 5 dB returned malformed JSON twice | 59 | 95.3 | 94.2 | 94.6 / 94.2 % | 97.1 / 95.4 % | 0.10 / 0.24 | 90 % | 100 % | 83 / 82 % | 52 |
| GPU medium fp16, hint + re-decode, no corrector (10-29, cohendynamics GPU transcripts) | local qwen3:4b-instruct, vox-ai GPU | rules on; 59 of 60, same SY117 failure | 59 | 95.3 | 94.2 | 94.6 / 94.2 % | 97.1 / 95.4 % | 0.10 / 0.24 | 90 % | 100 % | 83 / 82 % | 51 |
| GPU large-v3 fp16, our stack (10-29, every decode fix + corrector, cohendynamics GPU transcripts; the best local system) | local qwen3:4b-instruct, vox-ai GPU | rules on (ours, local, GPU whisper, largest model) | 60 | 96.6 | 95.3 | 96.6 / 94.9 % | 97.1 / 96.9 % | 0.17 / 0.20 | 90 % | 100 % | 83 / 87 % | 53 |
| GPU large-v3 fp16, hint + re-decode, no corrector (10-29) | local qwen3:4b-instruct, vox-ai GPU | rules on; identical to the corrector row | 60 | 96.6 | 95.3 | 96.6 / 94.9 % | 97.1 / 96.9 % | 0.17 / 0.20 | 90 % | 100 % | 83 / 87 % | 52 |

The GPU local stack row (10-29): the first full local system number with the day's decode fixes and the GPU transcripts (medium fp16 on cohendynamics, hint, span re-decode, corrector, then the 4B writer with rules on vox-ai) is 95.3 weighted at clean and 94.2 at 5 dB. Paired against the CPU local stack on the same clips it is +1.6 points at clean [+0.1, +3.4] and +0.6 at 5 dB [-0.3, +1.7]; against the cloud stack's 96.8 / 95.1 it is -1.5 at clean [-3.9, +0.9] and -1.1 at 5 dB [-3.9, +1.6] (30 and 28 pairs). So the GPU transcripts close the local-against-cloud gap of section 5 (which was -3.0 [-5.6, -0.6] at clean) to a difference whose interval now includes zero. The same transcripts without the corrector give the identical row: the corrector changed three of the sixty GPU transcripts (SY028 at 5 dB, SY079 clean, and SY117 at 5 dB, the clip whose report failed either way) and none of the changes moved a report score, so on this set the corrector's value is in the transcript metric (4.6 to 4.5 % WER), not in the report. Against the CPU base writer row (95.7 / 93.9, medium without a hint) the GPU hint + re-decode row is -0.4 [-1.8, +1.4] at clean and +0.3 [-0.8, +1.4] at 5 dB, and against the cloud base row (95.2 / 94.2) +0.1 [-2.3, +2.6] and -0.0 [-2.2, +2.2]: the local writer over GPU transcripts is level with the cloud base system. There is no GPU row with a plain base decode (no hint, no re-decode) yet. The rows over the large-v3 GPU transcripts are running.

The large-v3 GPU rows (10-29, night): the full local system with the largest model that fits the second device's GPU
(large-v3 fp16, every decode fix, corrector, then the 4B writer with rules) scores 96.6 weighted at clean and 95.3 at
5 dB, 60 of 60 reports. Paired against the cloud stack (96.8 / 95.1) it is -0.1 at clean [-2.4, +2.0] and +0.2 at 5 dB
[-2.4, +2.7]: the local system and the cloud system are level at the report stage. Against the GPU medium stack it is
+1.4 [-0.2, +3.0] and +0.9 [-0.3, +2.3]; the corrector again changed no report score. Precision is a point under the
cloud's (96.8 against 98.3 % at clean, P(local > cloud) = 0.10) and recall a point over (96.1 against 95.7 %), so the
remaining difference is in what the 4B states beyond the transcript, which is the slot-state writer's job.

The 8B rows (10-29): the 8B against the 4B on the same transcripts, paired over the reports the 8B finished, is +0.2
weighted points at clean [-1.0, +1.5] and +0.3 at 5 dB [-1.3, +2.0] over the base transcripts (27 and 29 reports),
and +1.2 [-0.3, +2.9] and +0.7 [-1.6, +3.0] over our stack (25 and 25). Every interval includes zero. The cost: the 8B
is 6.0 GB in a 7.5 GB box with no swap, Ollama placed 13 % of its layers on the CPU, 166 MB was left for everything
else, and a report took 171 s at the median, 270 s at the 90th percentile and 1401 s at worst (page-cache thrash),
with 32 timeouts in 120 attempts at a 420 s limit, against 50 s for the 4B on the same GPU. The 8B is not a live
writer on an 8 GB Orin (STATEFUL_PIPELINE.md section 1); its one use is offline, as a judge or a teacher.

Paired, per clip (`posterior.py --reports`): the local report over our transcripts against over the base transcripts is
-2.0 weighted points at clean [-3.4, -0.7] and -0.4 at 5 dB [-1.6, +0.7]; the cloud system against the local system
is +3.0 at clean [+0.6, +5.6] and +1.4 at 5 dB [-1.4, +4.2]. Recall and precision posteriors are in POSTERIOR_2026-10-28.md.

The clean-row drop is not what the transcript numbers predict: the two transcript sets have the same WER at clean
(5.8 %), and of the eleven clips that lost points, several did so with an unchanged or better transcript (SY079
WER 8.5 to 6.1 %, score 100 to 89; SY014 at 5 dB, WER 0 in both, score 100 to 93 with a new hallucination). One clip
is real: SY078 collapses under the revised hint and the re-decode recovers it only in part (WER 5.8 to 20.9 %, score
94 to 87). The two SY014 transcripts differ only in punctuation and case ("62 year old" against "62-year-old", "Lams" against
"LAMS"); from the punctuated one the model wrote "Applied cervical collar", reasoning in its own notes that "clear"
meant c-collar, and the claim check let it through because the check nulls allergies, code status and ETA but does
not yet drop an intervention the transcript never names. That is a rule gap to close (an intervention is a claim
too), recorded here rather than fixed mid-matrix so the rows stay comparable.

The repeat run settles what the rest of the gap is. The bare-transcript reports were run a second time on the same
device with the same input: all 60 weighted scores were identical to the first run, and 51 of 60 report JSONs were
byte-identical outside their metadata (the other nine differ in free-text wording that the scorer does not read).
The writer is deterministic on one device at temperature 0. So the 2.0-point drop at clean is not run-to-run noise;
it is caused by the transcripts themselves. The stack's transcripts are punctuated and cased (the hint and the
corrector produce "62-year-old", "LAMS", sentence breaks) where the bare model's are flatter, and on eleven clips
that surface form led the 4B model to a different and, on average, slightly worse extraction, SY014's invented
collar among them. On 5 dB audio the effect is within the interval. This is a sensitivity of the report model to
transcript formatting, worth a normalisation step before extraction, and it is the honest cost of the local stack on
clean audio: better words, a touch worse report, until the extraction stage is made indifferent to punctuation.

With that known, the reading of the report stage on the generated set is: local 94 to 96, cloud 95 to 97, the cloud
ahead by 3 points at clean with an interval that excludes zero and by 1.4 at 5 dB with one that does not;
hallucinations at or under 0.2 per report with the rules on, 0.7 with them off; and the level never wrong with the
rules on. Operational note: Ollama's model server stopped and was restarted by the client's retry loop 28 times in
the log of 240 reports on cohendynamics and 5 times in 81 on vox-ai; no report was lost, but the restarts are part
of the 50 s per report and an item for the deployment list.

The local report stage across the noise curve, over the hint-only medium transcripts (`medium_hint`, the stack
without the corrector and re-decode), 150 reports on the vox-ai GPU, 69 on 10-27 and 81 on 10-28 and 10-29 (one
report, SY099 at 10 dB, failed on malformed model JSON and was re-run; it then scored 90.9. The pipeline retries
a stopped model server but not a malformed reply, a gap for the deployment list):

| SNR | transcript WER | reports | weighted | recall | precision | halluc. per report | alert type | level | ETA | extract s |
|---|---|---|---|---|---|---|---|---|---|---|
| clean | 4.8 % | 30 | 94.7 | 94.4 % | 95.7 % | 0.20 | 90 % | 100 % | 78 % | 52 |
| 20 dB | | 30 | 94.7 | 93.9 % | 96.1 % | 0.13 | 90 % | 100 % | 78 % | 50 |
| 15 dB | 7.0 % | 30 | 92.9 | 92.2 % | 94.7 % | 0.17 | 87 % | 100 % | 74 % | 49 |
| 10 dB | 10.0 % | 30 | 94.2 | 93.8 % | 95.7 % | 0.23 | 90 % | 100 % | 78 % | 50 |
| 5 dB | 13.5 % | 30 | 93.9 | 94.1 % | 95.4 % | 0.23 | 90 % | 100 % | 83 % | 51 |

The WER nearly triples from clean to 5 dB and the weighted report score moves by under a point between the two ends
(94.7 to 93.9); the lowest point is 15 dB at 92.9, 1.8 under clean, with no paired interval computed for it. The
level is right at every SNR and hallucinations stay at or under 0.23 per report. The report stage is, on this set,
nearly indifferent to the transcriber's noise response in the range tested: the terms the writer needs survive
the channel at 5 dB, and the errors that remain fall mostly on words the scorer does not weight. The transcript WER
column repeats section 2's hint row (clean, 15, 10, 5 dB; no 20 dB column there).

Real corpus, from ROUND2.md (73 reports, same transcripts, local 4B): the grounding rules took unsupported claims
from 98 to 32, invented allergies and code status from 50 and 23 to 0 and 0, crashes from 4 to 0; on the four
gold samples the weighted score went 83.5 (baseline) to 88.6 (post-fix) to 91.3 (round 2). Fields filled per
report fell from 15.9 to 10.3: the model abstains instead of guessing, which is the intended trade.

## 4b. The abstract's statistic, done on our rows

`scripts/corpus/posterior.py`: a Beta-Binomial posterior P(B > A) per trial outcome with a flat prior, and a
paired bootstrap over clips for the per-clip score difference (the conservative companion: terms inside one
clip fail together, so the posterior overstates certainty). Full tables in POSTERIOR_2026-10-28.md.

| comparison (B against A) | term recall A | term recall B | P(B > A), pooled | WER difference at 5 dB, 95 % interval |
|---|---|---|---|---|
| local medium: our stack against the base model | 87.1 % | 90.5 % | 99.2 % | -1.8 points [-3.5, -0.6] |
| cloud turbo: our stack against the base model | 87.8 % | 88.5 % | 68.6 % | -1.4 points [-2.5, -0.4] |
| cloud stack against local stack | 90.5 % (local) | 88.5 % (cloud) | 7.3 % | -1.0 point [-3.0, +0.5] |
| medium against large-v3-turbo, hint on | 81.6 % | 88.7 % | 100.0 % | -3.7 points [-8.5, -0.4] |

Read with care: the local stack's term-recall gain over its base is real (99.2 %, and the WER interval excludes
zero at four of five SNRs); the cloud stack's gain shows in WER at four SNRs but not in term recall; and on term
recall the local stack is ahead of the cloud turbo stack (the cloud is ahead on WER only at clean). The abstract's
"88.0 to 92.0 %, P = 94.76 %" has the same shape as the first row and no provenance; the row above is the number
with one.

Report stage, paired on clips (`posterior.py --reports`):

| comparison (B against A) | recall A / B | P(B > A) | precision A / B | P(B > A) | weighted A / B | mean diff, 95 % interval |
|---|---|---|---|---|---|---|
| rules on against rules off, clean (same transcripts, same device) | 90.3 / 93.0 % | 92.2 % | 89.2 / 96.0 % | 100.0 % | 88.0 / 93.7 | +5.8 [+4.1, +7.5] |
| rules on against rules off, 5 dB | 90.3 / 93.0 % | 91.9 % | 90.1 / 96.0 % | 100.0 % | 88.3 / 93.7 | +5.4 [+3.8, +7.1] |
| our transcripts against base transcripts, local writer, clean | 94.9 / 93.0 % | 12.4 % | 96.8 / 96.0 % | 27.7 % | 95.7 / 93.7 | -2.0 [-3.4, -0.7] |
| our transcripts against base transcripts, local writer, 5 dB | 93.3 / 93.0 % | 44.6 % | 96.5 / 96.0 % | 35.9 % | 94.1 / 93.7 | -0.4 [-1.6, +0.7] |
| our transcripts against base transcripts, cloud writer, clean | 94.7 / 95.7 % | 73.9 % | 97.8 / 98.3 % | 69.2 % | 95.2 / 96.8 | +1.5 [-0.5, +3.9] |
| our transcripts against base transcripts, cloud writer, 5 dB (29 pairs) | 94.5 / 93.7 % | 33.2 % | 96.9 / 97.9 % | 80.0 % | 94.6 / 95.1 | +0.5 [-1.6, +2.8] |
| cloud system against local system, clean | 93.0 / 95.7 % | 95.0 % | 96.0 / 98.3 % | 97.0 % | 93.7 / 96.8 | +3.0 [+0.6, +5.6] |
| cloud system against local system, 5 dB | 93.0 / 93.7 % | 66.3 % | 95.9 / 97.9 % | 94.6 % | 93.7 / 95.1 | +1.4 [-1.4, +4.2] |

The grounding rules are the largest single effect in the report stage: 5.4 to 5.8 weighted points with intervals
far from zero, precision from 89 to 96 % (P = 1.000 both rows), hallucinations from 0.7 to 0.2 per report, and
the alert level from 67 % right to 100 % (the stated-priority rule). That is the "barrier rules" claim of the
architecture, measured.

## 5. The fully local path with the network cut

CS008 (84.5 s of audio) through `python -m ems_handoff.cli --wav` with every outbound HTTP call sent to a dead
proxy (`https_proxy=http://127.0.0.1:9`, `NO_PROXY=localhost` so Ollama is reachable, `HF_HUB_OFFLINE=1`).
Artifacts: `~/reports_replay/CS008_offline_2026-10-28/`.

| path | result |
|---|---|
| control: `HANDOFF_TRANSCRIBER=groq` | fails, `APIConnectionError: Connection error` |
| `HANDOFF_TRANSCRIBER=local`, medium, `qwen3:4b-instruct` | 143 s wall: transcribe 66.2 s, extract 74.4 s; WER 18.4 % (36/196); weighted 88.0 (16 hit, 0 omission, 3 wrong, 8 abstained, 0 hallucination); CARDIAC YELLOW ETA 5 (gold YELLOW, 5) |

This is the first recorded disconnected run. It shows the local path works with no network; it says nothing
about HIPAA, which is a matter of handling and policy, not of a proxy setting. The app's default live pipeline
still uses Groq for the transcript and for caption repair when a key is set.

## 6. Latency and footprint on the Orin Nano (8 GB)

| stage | local | cloud |
|---|---|---|
| transcript, speed as transcribe time over audio time | medium 0.85x, small 0.33x, large-v3-turbo 0.8x, tiny.en 0.1x (CPU int8) | paced at 20 clips a minute; about 1 to 2 s per burst |
| transcript on the GPU (10-29): medium on cohendynamics in the `dustynv/faster-whisper` container, 25 real bursts (983 s of audio), beam 5, no hint, no VAD | float16: 5.4 s per burst, RTF 0.137, model load 3.1 s; int8_float16: 6.2 s per burst, RTF 0.158. The same 25 bursts on the vox-ai CPU (int8, greedy, hint): 27.8 s per burst, RTF 0.706. Texts differ from the CPU pass by 18 to 20 % WER (different precision and decode settings); on the 5 bursts with a human reference the GPU float16 pass is 33.2 % WER against the CPU's 31.2 % (253 words, too few to rank them) | |
| report, extract time per clip | 50 s mean over 150 synthetic reports (vox-ai GPU), 50 to 51 s over 240 on cohendynamics; 52.7 s mean over the real corpus baseline | about 3 s |
| memory | llama-server 3.7 GB resident with qwen3:4b; faster-whisper large-v3-turbo 1.35 GB; medium about 1.3 GB; the two must not run at once on this box | |

The GPU row answers the speed question in HORSEPOWER_GOALS.md lever A: whisper medium on the Orin GPU is 5.1 times
faster than on the CPU at beam 5 (greedy would be faster still), which puts the local transcriber at 5 s per burst
against the cloud's 2 to 3, and leaves the GPU room (the container holds medium fp16 in 1.5 GB) for large-v3 fp16
(lever D) once the writer runs on the other box. Accuracy on the GPU is to be measured with the app's own decode
(hint, greedy plus re-decode) on the test split, not with the bench's plain beam 5; that run is next.

Time to first token was never measured; the app does not stream the report.

## 7. The abstract's claims against the data

| the abstract says | what the data says | source |
|---|---|---|
| Twenty pre-hospital recordings | 22 real calls on disk, 3 with human references (612 words); 1000 generated clips | DATA_LAYOUT.md, section 3 |
| Four configurations | five model sizes and the stack components measured one at a time; the closest honest mapping is cloud large-v3 (control), medium base (baseline edge), medium + stack (precision edge), small + stack (speed edge) | section 2 |
| Cosine similarity against the cloud control | not used; scoring against the control makes the control unbeatable. WER, term error and n-gram recall against human references and scripts | section 1 |
| Semantic precision and recall | recall and precision over stated facts against gold JSON; term recall over 200 occurrences | sections 2, 4, 4b |
| RTF and TTFT | real-time factor measured (medium 0.85x on the CPU); TTFT not measured, the report is not streamed; report time 50 s local, 3 s cloud | section 6 |
| Noise injection at varying SNRs | five SNRs with the measured radio floor on the generated set | section 2 |
| Beta-Binomial P(Opt > Base) | computed with a flat prior on term occurrences and stated facts, with a paired bootstrap beside it | section 4b |
| Network disconnection verified HIPAA compliance | one disconnected run completed; that shows the local path runs offline and nothing about HIPAA | section 5 |
| Both edge models outperformed the cloud control | the local stack is within a point of the cloud turbo stack on real audio (22.7 against 21.7 % WER) and ahead on term recall (90.5 against 88.5 %, P = 92.7 %); the full cloud model with the stack is best (16.2 %); base against base on real audio the local model wins (25.7 against 27.6 %) | sections 2, 3, 4b |
| Recall 88.0 to 92.0 %, P = 94.76 % | term recall, local medium, stack against base: 87.1 to 90.5 %, P = 99.2 %; WER interval excludes zero at four of five SNRs | section 4b, CALCULATIONS_2026-10-28.md |
| Precision 83.68 to 90.64 % from acoustic filtering, P = 99.44 % | nothing measured corresponds; the only filters are a 150 Hz mic high-pass and the radio EQ for synthesis; report-stage precision is 96 to 98 % on every row | sections 4, 6 |
| 1.5 to 2.0 times faster | no variant is faster at equal accuracy; small is 2.6 times faster than medium at 2 more points of WER and 7 more of term error | section 6 |
| Layered RAG isolates clean transcriptions | the lexicon retrieval feeds the extraction prompt; textbook retrieval has no store built and is off in every run; the clean transcript comes from the hint, the re-decode, the corrector and the checks | code, section 1 |
| Real-time resuscitation support | the local report takes about 50 s after the call ends on the Orin Nano; the cloud, about 3 s | section 6 |

## 8. Still owed

- Cloud rules-off row: 13 of 60 reports before the Groq 200k tokens-per-day cap (`logs/matrix_2026-10-29_cloud_base.log`);
  `report_set.py` resumes it when the cap resets. The cloud base row is in (60 of 60, 10-29).
- Term list for the real references, so MTER exists on real audio.
- Paired bootstrap intervals over clips for every before-and-after difference (FINETUNE_PLAN.md section 4).
