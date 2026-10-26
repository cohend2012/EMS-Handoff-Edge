# Whisper hint ablation (Groq whisper-large-v3-turbo, temperature 0, WER against ~/labels/reference)

## CS008

| variant | S025_01 | all |
|---|---|---|
| none | 13% | 13% |
| terms | 18%±5 | 18% |
| radio | 11%±0 | 11% |
| radio+names | 16%±4 | 16% |

## CS014

| variant | S054_01 | S055_01 | S013_01 | S014_01 | S015_01 | all |
|---|---|---|---|---|---|---|
| none | 40% | 67% | 56% | 22% | 22% | 28% |
| terms | 42%±3 | 0% | 56% | 11% | 17% | 23% |
| radio | 85% | 0% | 56% | 11% | 23% | 36% |
| radio+names | 69%±35 | 33% | 33% | 59%±11 | 29%±6 | 39% |

## CS001

| variant | S044_01 | S045_01 | S046_01 | all |
|---|---|---|---|---|
| none | 52% | 33% | 27% | 47% |
| terms | 38% | 15% | 13% | 32% |
| radio | 21% | 4% | 13% | 17% |
| radio+names | 36%±11 | 4% | 13% | 29% |

## All call-sets

| variant | WER over all words |
|---|---|
| none | 28.2% (516/1833) |
| terms | 23.6% (433/1833) |
| radio | 23.1% (424/1833) |
| radio+names | 28.9% (529/1833) |
