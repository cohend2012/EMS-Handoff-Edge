# The transcriber stacks on the real call-sets with a reference (CS001, CS008, CS014)

Base = the model alone, no prompt, no re-decode, no corrector. Ours = the app's radio-phrase hint, the gap
re-decode (local only) and the transcript corrector. WER pooled over all reference words; n-gram scores
word-weighted. Cached transcripts: ~/labels/reference/<CS>/<seg>.stack.json (scripts/corpus/real_stack.py).

## All bursts

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3 + hint + corrector | 9 | 612 | 16.2 % (99/612) | 0.769 | 0.67 |
| cloud large-v3-turbo + hint + corrector | 9 | 612 | 21.7 % (133/612) | 0.729 | 0.62 |
| local medium + hint + gap re-decode + corrector | 9 | 612 | 22.7 % (139/612) | 0.681 | 0.55 |
| local medium, base (no prompt) | 9 | 612 | 25.7 % (157/612) | 0.671 | 0.53 |
| cloud large-v3-turbo, base (no prompt) | 9 | 612 | 27.6 % (169/612) | 0.665 | 0.54 |

## CS001

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3-turbo + hint + corrector | 3 | 163 | 16.6 % (27/163) | 0.755 | 0.64 |
| cloud large-v3 + hint + corrector | 3 | 163 | 20.9 % (34/163) | 0.709 | 0.60 |
| local medium + hint + gap re-decode + corrector | 3 | 163 | 20.9 % (34/163) | 0.698 | 0.56 |
| local medium, base (no prompt) | 3 | 163 | 25.8 % (42/163) | 0.678 | 0.51 |
| cloud large-v3-turbo, base (no prompt) | 3 | 163 | 46.6 % (76/163) | 0.550 | 0.42 |

## CS008

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3 + hint + corrector | 1 | 196 | 10.2 % (20/196) | 0.862 | 0.79 |
| cloud large-v3-turbo + hint + corrector | 1 | 196 | 11.2 % (22/196) | 0.849 | 0.78 |
| cloud large-v3-turbo, base (no prompt) | 1 | 196 | 13.3 % (26/196) | 0.796 | 0.70 |
| local medium, base (no prompt) | 1 | 196 | 18.4 % (36/196) | 0.747 | 0.62 |
| local medium + hint + gap re-decode + corrector | 1 | 196 | 18.4 % (36/196) | 0.734 | 0.62 |

## CS014

| stack | bursts | words | WER | avg n-gram F | 4-gram recall |
|---|---|---|---|---|---|
| cloud large-v3 + hint + corrector | 5 | 253 | 17.8 % (45/253) | 0.735 | 0.62 |
| cloud large-v3-turbo, base (no prompt) | 5 | 253 | 26.5 % (67/253) | 0.638 | 0.50 |
| local medium + hint + gap re-decode + corrector | 5 | 253 | 27.3 % (69/253) | 0.630 | 0.50 |
| local medium, base (no prompt) | 5 | 253 | 31.2 % (79/253) | 0.608 | 0.48 |
| cloud large-v3-turbo + hint + corrector | 5 | 253 | 33.2 % (84/253) | 0.619 | 0.50 |
