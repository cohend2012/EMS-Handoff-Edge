# Worked calculations behind SYSTEM_VS_BASE_2026-10-28.md

## 1. Word error rate (CS008, the offline run)

Reference after normalisation: 196 words. Levenshtein edits (substitutions + deletions + insertions) from the transcript to the reference: 36.
WER = 36 / 196 = 0.1840 = 18.4 %.

Normalisation folds case, punctuation and spoken units ("S-P-O-2" and "SP02" to spo2, "E.T.A." to eta) so a spelling of the same word is not an error.

## 2. Term recall and the Beta-Binomial posterior (local medium: our stack B against the base model A)

Trials: the 149 terms of terms.txt occur 200 times in the 30 test scripts; over 5 SNRs that is 1000 trials per run.
A: 871 terms present, 129 missing. Recall = 871/1000 = 87.1 %.
B: 905 present, 95 missing. Recall = 905/1000 = 90.5 %.

Flat prior Beta(1, 1). The posterior for a Binomial with h successes and m failures is Beta(1 + h, 1 + m).
A ~ Beta(872, 130): mean 0.8703, sd 0.0106
B ~ Beta(906, 96): mean 0.9042, sd 0.0093
P(B > A): 200 000 Monte Carlo draws give 0.9920; numerical integration of the two Beta densities gives 0.9920; the normal approximation (difference of means 0.0339, sd 0.0141, z = 2.41) gives 0.9919.

Caveat: trials inside one clip are not independent (a collapsed window loses its ten terms together), so the effective trial count is smaller and this probability is optimistic. The bootstrap is the check.

## 3. Paired bootstrap on WER at 5 dB (same pair)

30 clips. Per-clip WER difference (B minus A), in points: -1.0, +0.0, +0.9, -0.8, +0.0, +0.0, -21.4, -6.2, +0.0, +0.8, -0.7, +0.0, -2.2, -2.6, -0.6, +0.0, -0.9, -2.2, -2.9, +1.9, -1.5, -5.7, -1.0, +1.0, -2.8, +0.0, -2.6, +0.0, -1.1, -1.8.
Mean difference = -1.78 points. Clips where B is better: 18, worse: 4, equal: 8.
Resample the 30 differences with replacement 10 000 times and take each mean: the 2.5th and 97.5th percentiles are [-3.4, -0.6] points. Zero is outside, so the 5 dB gain holds at the clip level.

## 4. Weighted field score (CS008 offline report against its gold)

| field | gold | report | bucket | weight | earned |
|---|---|---|---|---|---|
| alert_type | enum | CARDIAC | wrong | 3 | 0 |
| alert_level | enum | YELLOW | hit | 3 | 3 |
| eta_minutes | int | 5 | hit | 2 | 2 |
| patient.age | int | 62 | hit | 2 | 2 |
| patient.age_unit | enum | years | hit | 1 | 1 |
| patient.sex | enum | Female | hit | 2 | 2 |
| chief_complaint | contains | substernal chest pain radiatin | hit | 3 | 3 |
| chief_complaint | contains | substernal chest pain radiatin | hit | 1 | 1 |
| chief_complaint | contains | substernal chest pain radiatin | hit | 1 | 1 |
| mechanism_or_onset | contains | began while sitting on couch a | hit | 1 | 1 |
| vital_signs.blood_pressure | contains | 188/111 | hit | 2 | 2 |
| vital_signs.heart_rate | int | 95 | wrong | 2 | 0 |
| vital_signs.heart_rate | contains | {"blood_pressure": "188/111",  | wrong | 1 | 0 |
| vital_signs.spo2 | contains | 95% on room air | hit | 2 | 2 |
| vital_signs.spo2 | contains | 95% on room air | hit | 1 | 1 |
| vital_signs.respiratory_rate | unknown | null | abstained | 1 | 1 |
| vital_signs.temperature | unknown | null | abstained | 1 | 1 |
| vital_signs.blood_glucose | unknown | null | abstained | 1 | 1 |
| vital_signs.gcs | unknown | null | abstained | 1 | 1 |
| ems_interventions | list | 2 items | hit | 3 | 3 |
| ems_interventions | contains | [{"intervention": "aspirin adm | hit | 2 | 2 |
| ems_interventions | contains | [{"intervention": "aspirin adm | hit | 2 | 2 |
| ems_interventions | contains | [{"intervention": "aspirin adm | hit | 1 | 1 |
| history.past_medical | unknown | null | abstained | 2 | 2 |
| history.medications | unknown | null | abstained | 3 | 3 |
| history.allergies | unknown | null | abstained | 3 | 3 |
| history.code_status | unknown | null | abstained | 3 | 3 |

Earned weight 44 of 50: weighted = 100 x 44/50 = 88.0. Recall = hit / stated = 16/19 = 84.2 %. Precision = hit / (hit + wrong + hallucinated) = 16/19 = 84.2 %. An abstention (gold says nothing, report says null) earns its weight; a hallucination earns nothing and counts against precision.

## 5. n-gram precision and recall (CS014 burst S013_01, cloud stack)

Reference: "Medic 729, Shady Grove. You have my doctor yet?"  
Transcript: "Like, 729C group, you in my doctor yet?"  
Normalised: "medic 729 shady grove you have my doctor yet" / "like 729 c group you in my doctor yet"

| n | reference n-grams | transcript n-grams | shared | precision = shared / transcript | recall = shared / reference | F |
|---|---|---|---|---|---|---|
| 1 | 9 | 9 | 5 | 0.56 | 0.56 | 0.56 |
| 2 | 8 | 8 | 2 | 0.25 | 0.25 | 0.25 |
| 3 | 7 | 7 | 1 | 0.14 | 0.14 | 0.14 |
| 4 | 6 | 6 | 0 | 0.00 | 0.00 | 0.00 |

avg n-gram F = mean of the four F values = 0.237. 4-gram recall = 0.00 is the share of four-word runs that survived intact.

## 6. Speed as real-time factor

- medium_hint_rd: 5351 s of transcribing for 6285 s of audio = 0.85x real time
- small_hint: 2049 s of transcribing for 6285 s of audio = 0.33x real time
- large-v3-turbo_hint: 4997 s of transcribing for 6285 s of audio = 0.80x real time
- tiny.en_hint: 642 s of transcribing for 6285 s of audio = 0.10x real time
- small against medium: 0.33 / 0.85 = 2.6 times faster, at 6.2 % against 4.8 % WER and 17.5 % against 10.0 % terms missed at clean. That is the only speed-for-accuracy trade in the data; no variant is faster at equal accuracy.

## 7. Pooled WER on the real calls (why 612 words)

Edits and reference words are summed over the nine bursts, then divided, so long bursts weigh more than short ones (a mean of per-burst WERs would let a two-word burst count as much as a 196-word report):

- cloud_large_ours: 99 / 612 = 16.2 %
- cloud_ours: 133 / 612 = 21.7 %
- local_ours: 139 / 612 = 22.7 %
- local_base: 157 / 612 = 25.7 %
- cloud_base: 169 / 612 = 27.6 %
