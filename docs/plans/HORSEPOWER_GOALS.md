# Goal: faster and more accurate than the cloud, on our own hardware

Set 2026-10-29. The standing goal for the two Jetsons (and the NPU to come): a local system that is
faster and more accurate than the cloud path, or as close as the hardware allows, with proof of the kind
in `docs/reports/SYSTEM_VS_BASE_2026-10-28.md`: same test split, same scorer, same seeds, every row
naming its device, paired comparisons with a posterior. Fine-tuning and shrinking models are allowed.

## 1. The numbers to beat (the cloud path, measured 10-28 and 10-29)

| stage | cloud (Groq) | local today (vox-ai, CPU whisper, GPU writer) | gap |
|---|---|---|---|
| transcript WER, synthetic test split, clean / 5 dB | turbo + our stack 4.2 / 6.4 %; large-v3 + our stack 3.9 / 4.9 % | medium + our stack 5.8 / 6.9 % (medium + hint 4.8 / 7.0 %) | about 1 to 2 points |
| transcript WER, real calls (bursts with a reference) | turbo + stack 11 to 33 % by set; large-v3 + stack 10 to 21 % | medium + stack 18 to 27 % | 2 to 8 points, set by set |
| transcriber speed | about 2 to 3 s per burst (network included) | medium int8 on the CPU: 28 s per burst (RTF 0.7 to 0.85) | 10x |
| report quality, weighted, clean / 5 dB | gpt-oss-120b, rules on: 96.8 / 95.1 (halluc. 0.03 to 0.07 per report) | qwen3:4b-instruct, rules on: 93.7 to 95.7 / 93.7 to 94.1 (halluc. 0.10 to 0.20) | 1 to 3 points |
| report speed | 3 s | 50 s (4B on one Orin GPU) | 17x |

Beating the cloud therefore means: whisper on a GPU (speed), a larger or adapted acoustic model made
affordable by that GPU (accuracy), and a writer that is as grounded as the 120B at a fraction of its size.

## 2. What the measurements already tell us

- **The local large-v3-turbo is not a worse model, it fails catastrophically on a few clips.** On the clean
  test split its median WER is 6.1 % (medium 4.8 %), but 4 of 30 clips lose most of their tail
  (SY074, SY055, SY184 with 64 to 79 % of the tail missing; SY078 repeats), which drags the mean to 12.8 %.
  The temperature fallback and the gap re-decode did not rescue them. The local decode runs int8, beam 1,
  with the VAD filter on; the cloud runs the same weights in fp16 with a wider beam and no VAD. Fixing the
  decode settings is cheaper than changing the model and is the first experiment.
- **The grounding rules are worth 5.8 weighted points** (precision 89 to 96 %, hallucinations 0.7 to 0.2
  per report), so the writer's size is not the only lever; the 8B measurement running today says how much
  size buys on top.
- **The GPU writer is deterministic on one device** (a second run of 60 reports reproduced every number).

## 3. Levers, in the order we will pull them, each with its proof

| # | lever | runs on | experiment (proof) | status |
|---|---|---|---|---|
| A | whisper on the GPU | cohendynamics, dustynv faster-whisper container (docker group granted 10-29) | 25 real bursts, medium fp16 and int8, seconds per burst and RTF against the 28 s CPU baseline; same texts as the CPU run | queued behind the 8B run |
| B | turbo decode fix | vox-ai CPU now, GPU after A | test split clean + 5 dB: turbo float32, beam 5, VAD off, each alone; the four tail-drop clips must come back | done 10-29: beam 5 fixes all four (turbo 6.2 / 8.1 %, same speed); float32 fixes two; VAD off none. Medium beam 5 is a wash against same-hint greedy (7.1 % both), so medium stays greedy. The failures are collapsed 30 s windows, which the gap re-decode detects but its greedy span decode looped; now the span is decoded with beam 5 (LOCAL_WHISPER_REDECODE_BEAM), rows _rd5 running |
| C | smaller models that keep accuracy | vox-ai CPU, then GPU | distil-large-v3 and distil-large-v3.5 with the hint on the same split; WER, term recall, tail, seconds | done 10-29: both out (9.4 / 10.5 % and 8.8 / 12.6 % against medium 4.8 / 7.0 %, only 15 to 30 % faster) |
| D | the largest model that fits | cohendynamics GPU with the writer moved to the other box | large-v3 fp16 (3.1 GB) on the GPU, our stack; against cloud large-v3 + stack (3.9 / 4.9 %) | after A |
| E | domain fine-tune | needs CUDA torch (container on cohendynamics if A works; else a rented GPU for the training hours only, deployment stays local) | LoRA on the train split (140 scripts x 5 SNR = 700 clips) plus real bursts held out by call set; score on the test split and the 51 real bursts; convert to ctranslate2 int8 and fp16 | after A |
| F | writer size and grounding | cohendynamics GPU | 8B against 4B on the same 60 transcripts (running); then rules on/off on the 8B | running |
| G | split the live pipeline | both boxes | whisper on one GPU, writer on the other; CS008 replay end to end, time to report | after A |

Each row lands in the matrix report with its device named, its posterior against the row it replaces, and
its per-clip rows kept under `~/datasets/synth_v1/evals/<label>/rows.tsv`.

## 4. Decision rules

- A model change ships only if it is at least as good on the test split **and** on the real bursts, and
  no clip regresses by a tail drop or a loop.
- Speed counts only with accuracy held: report RTF and seconds per burst next to WER, never alone.
- The writer changes only if weighted score, hallucinations per report and level-right all hold or improve.

## 5. The NPU

Do not buy until A and D are measured: they tell us whether the bottleneck is the encoder (NPU-friendly)
or the decoder and the writer (memory-bound, GPU-friendly). When buying, the test is simple: the vendor
must ship a whisper encoder or a ctranslate2-class runtime for the part, with a published example on
Jetson; we then rerun rows A to D on it and add a column. Until then the second Jetson's GPU is the
measurement platform.

## 6. What needs the user

- One sudo on cohendynamics for the CUDA toolkit (`nvidia-jetpack`), only if the JetPack 6 container
  fails on this R39.2 host (lever A tells us).
- A decision on training compute for lever E if the container route cannot run torch: rent GPU hours,
  or train slowly on the Jetson with the small model.
