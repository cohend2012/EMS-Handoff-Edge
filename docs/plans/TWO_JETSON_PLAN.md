# Two Jetsons: where the horsepower is and the plan to use it (2026-10-29)

Two Orin Nano boxes, 8 GB each: vox-ai (development, the app's screen and microphone) and cohendynamics
(Orin Nano Super, connected 2026-10-28). Each box alone cannot hold the report model (3.7 GB resident for
qwen3:4b) and whisper medium (1.3 GB) comfortably at once, and the transcriber runs on the CPU on both,
because ctranslate2 ships CPU-only for aarch64 (0 CUDA devices on either box). That is the whole shape of the
problem: one box has to serialise the two heavy stages, and the heaviest one never touches the GPU.

## Where we are (measured)

| stage | today, one box | with the second box |
|---|---|---|
| transcribe, medium int8 on CPU | 0.85x real time (a 60 s burst takes 50 s) | same per box; two boxes do two bursts at once |
| report, qwen3:4b on the GPU | 50 s per report; blocks whisper on the same box | runs on cohendynamics while vox-ai transcribes |
| report, qwen3:8b on the GPU | does not fit beside whisper | fits alone on cohendynamics: 9.9 tok/s generation (4B: 17.7), 68 s cold load |
| the matrix runs (150 clips, 60 reports a row) | 85 to 90 min per transcriber row, 50 min per report row, serial | 2026-10-28: report rows on cohendynamics while vox-ai transcribed; 2026-10-29: the 51 real bursts split 22/29 across the boxes |

The second box is already paying for itself in evaluation throughput. It is not yet part of the live app.

## The plan, in the order worth doing

1. **Split the live pipeline across the boxes (this week, no new software).** vox-ai keeps the microphone,
   the screen and the transcriber (medium + hint + gap re-decode + corrector); cohendynamics serves the report
   writer over the wired link (OLLAMA_HOST=0.0.0.0 on cohendynamics, OLLAMA_URL=http://10.42.0.1:11434/v1 in
   vox-ai's .env). Nothing in the app changes; the config already takes a remote Ollama URL. Measure with the
   CS008 replay the way the offline run was measured on 2026-10-28 (143 s fully local on one box): the clock
   from last word to report on one box against two. Expected: the report no longer waits for whisper to release
   memory and the two stages overlap on multi-burst calls; the per-report time stays about 50 s with the 4B.

2. **A bigger writer, measured before it is trusted (running today).** The report stage with qwen3:8b on
   cohendynamics over the same 60 transcripts as the 4B rows (bare transcript and our stack, clean and 5 dB),
   scored by the same scorer; log `reports_8b_2026-10-29.log` on that box. The 4B rows are 93.7 to 95.7
   weighted with the rules on. If the 8B gains two points or more on precision or hallucinations at about
   100 s per report, it becomes the writer in the two-box configuration; if not, the 4B stays and the second
   box's GPU is spare for step 3.

3. **Put whisper on a GPU (the largest single lever, needs Daniel's sudo once).** A CUDA build of ctranslate2
   for the Orin runs medium several times faster than real time instead of 0.85x; that turns a 60 s burst from
   50 s of transcription into a few seconds and lets large-v3 run locally. Two routes, both need root on
   cohendynamics: the jetson-containers faster-whisper image under Docker with the NVIDIA runtime (Docker is
   installed there; the runtime and the user's docker-group membership need sudo), or a ctranslate2 wheel built
   with CUDA on the box (needs the CUDA toolkit from JetPack installed). Try the container first: one afternoon
   to know whether it works on R39. Then the transcriber moves to cohendynamics as a small HTTP service, and
   vox-ai becomes the screen, the microphone and the report writer, or the roles swap.

4. **Keep splitting the evaluation runs (already the practice).** transcribe_callsets.py --ids and the
   report_set queues split any run by call-set or by row; half the wall time for every matrix from here.

5. **What not to do.** Do not run whisper and the 8B together on one box (5.2 + 1.3 GB plus the system does
   not fit in 8 GB without swap thrash). Do not try to make the two boxes share one whisper job; bursts are
   independent and splitting by burst is the parallelism. Do not buy the "GPU whisper" claim without the
   measurement in step 3.

## What to compare against

The base flow is whisper medium alone and the writer with no rules; the stack's gains over it are in
SYSTEM_VS_BASE_2026-10-28.md. The two-box work does not change accuracy by itself; it changes time and what
fits. The accuracy question it opens is the 8B writer (step 2), and the time question is step 1 and step 3.
Report each with the same clip set, the same scorer and the device named on every row.
