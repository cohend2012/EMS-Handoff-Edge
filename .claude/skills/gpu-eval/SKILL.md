---
name: gpu-eval
description: Run a whisper transcriber evaluation on the second Jetson's GPU (cohendynamics, Orin Nano Super) for the EMS-Handoff project. Use whenever the user wants a model, a decode setting, a hint variant or a noise curve measured "on the GPU", "on cohendynamics", "on the other Jetson", or faster than the vox-ai CPU allows; covers the CUDA container, the whisper server route, label names, syncing results back, and the memory limits that bite.
---

# GPU eval on cohendynamics

vox-ai's JetPack is too old for CUDA faster-whisper, so GPU decoding happens on cohendynamics
(`cohendynamics@100.77.24.78`, key SSH, 7.5 GB RAM, no swap) inside the image `ems-whisper-gpu:scipy`. Two
routes; pick by what else is running.

## Route A: eval_set inside a container (a model that is not already loaded)

The repo there is an rsync copy at `~/EMS-Handoff-app` (not git). After any code change:

```bash
rsync -a --exclude .venv --exclude .git --exclude output --exclude __pycache__ ems_handoff scripts tests cohendynamics@100.77.24.78:EMS-Handoff-app/
```

Run through a script on cohendynamics (pattern in `~/gpu_whisper/run_gpu_eval.sh`):

```bash
docker run --rm --runtime nvidia -e LD_LIBRARY_PATH=/opt/nvidia/l4t-gpu-libs/nvgpu:/usr/local/cuda/lib64 \
  -e LOCAL_WHISPER_DEVICE=cuda -e LOCAL_WHISPER_COMPUTE=float16 -e HOME=/root -e PYTHONUNBUFFERED=1 \
  -v /home/cohendynamics/EMS-Handoff-app:/app -v /home/cohendynamics/datasets/synth_v1:/data \
  -v /home/cohendynamics/.cache/huggingface/hub:/root/.cache/huggingface/hub \
  ems-whisper-gpu:scipy python3 /app/scripts/corpus/eval_set.py --set /data --split test --snr inf 5 \
  --model large-v3 --redecode --label large-v3_gpu_fp16_hint_rd5d
```

`--runtime nvidia` and the `LD_LIBRARY_PATH` are both required or CUDA reports zero devices. One container
per model: loading two models in one process died silently. `--redecode` turns the span re-decode on (evals
leave it off by default); `--no-hint` gives the plain base; `--hint-file /data/hints/<v>.txt` a hint variant.
Launch with `setsid nohup ... &` and wait with an anchored pgrep over ssh (see below). Timing: medium 5 min,
turbo 4 min, large-v3 8 min for the 60-clip clean + 5 dB rows.

Then sync the label back and pair it (the `matrix-row` skill):

```bash
rsync -a cohendynamics@100.77.24.78:datasets/synth_v1/evals/<label> ~/datasets/synth_v1/evals/
```

## Route B: the whisper server (large-v3 already loaded)

The container `ems-whisper-server` (restart unless-stopped, port 8765) runs `scripts/whisper_server.py` with
large-v3 fp16, hint and re-decode. From vox-ai, eval_set can score through it with no second model load:

```bash
HANDOFF_TRANSCRIBER=remote HANDOFF_WHISPER_URL=http://100.77.24.78:8765 .venv/bin/python scripts/corpus/eval_set.py \
  --split test --snr 20 15 10 --provider remote --model large-v3 --label large-v3_gpu_fp16_hint_rd5d_curve
```

About 8 to 9 s a clip over Tailscale. Check `curl http://100.77.24.78:8765/health` first; after a code change,
rsync and `docker restart ems-whisper-server` (50 s to reload). Use this route while vox-ai's writer is busy:
the two GPUs then work at once.

## Memory

Two large-v3 instances (server plus a container) do not fit beside each other in 7.5 GB with no swap. Run a
second model in a container only when the server is stopped, or use medium (1.5 GB). Ollama's 8B there needs
CPU offload and is offline-only; never run it beside whisper.

## Labels and results

`<model>_gpu_fp16_<stack>` where stack is `hint_rd5d` (shipped decode), `hint_rd5d_corr` (plus corrector,
made on vox-ai with `eval_set --from <label> --corrector`), `nohint` (plain base), `hintfit`/`hintshort`
(ablations). Real bursts go through `~/gpu_whisper/real_gpu.py` (the `real-bursts` skill). Stack caches under
`~/labels/reference/<CS>/<seg>.stack.json` are merged key-wise, never rsynced whole: a plain rsync overwrote
the other box's rows once.

## Waiting

Foreground sleeps are blocked; anchor the pattern to the job's argv so the waiter does not match itself:

```bash
while ssh -o BatchMode=yes cohendynamics@100.77.24.78 'pgrep -f "^/bin/bash /home/cohendynamics/gpu_whisper/run_x.sh" >/dev/null'; do sleep 300; done
```
