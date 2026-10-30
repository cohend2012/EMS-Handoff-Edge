---
name: split-pipeline
description: Run the EMS-Handoff app or its headless CLI with whisper on the cohendynamics GPU and the report writer on vox-ai (the two-Jetson split), start or restart the whisper server, and time a real call end to end. Use whenever the user mentions the split pipeline, the remote transcriber, "both GPUs", "whisper on the other Jetson", time to report, or wants a latency number for a call.
---

# Split pipeline: whisper on cohendynamics, writer on vox-ai

## The server (cohendynamics)

`scripts/whisper_server.py` is the app's `LocalTranscriber` (hint, span re-decode, model choice) behind one
HTTP endpoint. It runs in the CUDA container as `ems-whisper-server`, restart unless-stopped, port 8765,
large-v3 fp16 (3.5 GB; medium fits beside the 4B if both must share that box, large-v3 does not).

```bash
# after a code change: rsync the repo copy, then
ssh cohendynamics@100.77.24.78 'docker restart ems-whisper-server'      # 50 s to reload
# from scratch
ssh cohendynamics@100.77.24.78 'docker run -d --name ems-whisper-server --restart unless-stopped --runtime nvidia -p 8765:8765 \
  -e LD_LIBRARY_PATH=/opt/nvidia/l4t-gpu-libs/nvgpu:/usr/local/cuda/lib64 -e LOCAL_WHISPER_DEVICE=cuda -e LOCAL_WHISPER_COMPUTE=float16 \
  -e LOCAL_WHISPER_MODEL=large-v3 -e HANDOFF_TRANSCRIBER=local -e HOME=/root -e PYTHONUNBUFFERED=1 \
  -v /home/cohendynamics/EMS-Handoff-app:/app -v /home/cohendynamics/.cache/huggingface/hub:/root/.cache/huggingface/hub -w /app \
  ems-whisper-gpu:scipy python3 scripts/whisper_server.py --port 8765'
curl -s http://100.77.24.78:8765/health      # {"ok": true, "model": "large-v3", "device": "cuda", ...}
```

## The client (vox-ai)

Two settings switch the app, the CLI and `eval_set --provider remote` to the server:

```bash
HANDOFF_TRANSCRIBER=remote HANDOFF_WHISPER_URL=http://100.77.24.78:8765
```

`RemoteTranscriber` posts the burst as WAV and reads `{"text", "seconds", "notes"}`; the hint and decode are
the server's. Put the two lines in `.env` to make the live app use it; leave them out for the one-box path.

## Timing a real call end to end

`/usr/bin/time` is not installed on vox-ai; use bash's `time` keyword:

```bash
cd /home/voxai/EMS-Handoff-app; export OLLAMA_MODEL=qwen3:4b-instruct TIMEFORMAT='wall %R s'
{ time env HANDOFF_TRANSCRIBER=remote HANDOFF_WHISPER_URL=http://100.77.24.78:8765 .venv/bin/python -m ems_handoff.cli \
    --wav /home/voxai/callsets/CS008/S025_01.wav --provider ollama --json 2>&1 | grep -E "^Transcript \(|transcribe_s|extract_s" ; } 2>&1
```

Run three times, then the same with `HANDOFF_TRANSCRIBER=local` for the same-day one-box pair; report
transcript, writer and wall seconds for each. Ollama must be free (no writer row running), or the writer time
is not the writer's. CS008 (84.5 s of audio, the average real call) is the reference call: 10-30 gave 70.4 /
71.0 / 74.5 s split against 134.6 s one box (transcript 15 to 21 s against 66.6 s, writer 50 to 52 s).

## What the numbers mean

The writer's 50 s is now 70 % of the wait and is mostly output tokens (about 820 at 16 tokens a second on
vox-ai). The next latency levers are on the writer side (shorter output, slot-state delta writes:
`docs/plans/STATEFUL_PIPELINE.md`), not on whisper. Write results into the matrix section 6, the ledger, the
study's 5.5 and abstract, and goals lever G (the `matrix-row` skill).
