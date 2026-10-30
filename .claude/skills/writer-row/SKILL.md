---
name: writer-row
description: Run the report writer (qwen3:4b-instruct through Ollama on the vox-ai GPU) over a set of transcripts and score the reports against gold for the EMS-Handoff matrix. Use whenever a transcript label needs its report-stage row, whenever the user asks what the writer scores on some transcripts, for ablations of the writer's prompt, rules or lexicon retrieval, and for queueing several such runs one after another with waiters.
---

# Writer row (report stage)

```bash
cd /home/voxai/EMS-Handoff-app
.venv/bin/python scripts/corpus/report_set.py --split test --from <eval label> --snr inf 5 --provider ollama --model qwen3:4b-instruct [--label <name>]
```

Reads `~/datasets/synth_v1/evals/<eval label>/transcripts/`, writes one JSON per clip and `rows.tsv` /
`summary.md` under `~/datasets/synth_v1/reports/<eval label>__qwen3-4b-instruct/` (or `--label`). Resumable:
rerunning skips finished clips. About 50 s a report, 52 minutes for the 60 clean + 5 dB reports.

Switches for ablations (each wants its own `--label` so the default folder is not overwritten):

| what | how |
|---|---|
| grounding rules off | `--no-rules` (label gets `_norules`) |
| lexicon retrieval off | `HANDOFF_USE_LEXICON=0 ... --label <eval>__qwen3-4b-instruct_nolex` |
| another writer | `--model qwen3:8b` (only on cohendynamics, offline; 171 s median, it does not fit live) |
| cloud writer | `--provider groq` (paced 25 s; the 200k tokens/day cap fits about 60 reports) |

## Memory rule

Whisper medium and the Ollama 4B cannot share vox-ai's 7.4 GB. Report runs go on vox-ai's GPU while whisper
work goes to cohendynamics (the `gpu-eval` skill); never run a local eval_set and a report_set on vox-ai at
once. Pin the model with `OLLAMA_MODEL=qwen3:4b-instruct` (the `qwen3:4b` tag is thinking-only).

## Queueing

Write a bash script in `~/datasets/synth_v1/logs/queue_<name>_<date>.sh` that waits on the previous queue
with an anchored pgrep and then runs the report_set lines, launch it with `setsid nohup ... > log 2>&1 < /dev/null &`,
and arm a background waiter on the queue's own argv:

```bash
while pgrep -f "^/bin/bash /home/voxai/datasets/synth_v1/logs/queue_<name>.sh" >/dev/null; do sleep 300; done
```

Never `pkill -f`: it kills the Bash call itself. Progress is the JSON count in the reports folder (python
stdout through grep is block-buffered).

## Failures

"PipelineError: The AI returned malformed JSON" on one clip is the model, not the run; the row is reported as
59 of 60 with the clip named, and `report_set --ids <id>` can retry it. "needs ~3.9 GB but only 3.3 GB is
free" is a warning; the run continues. Ollama's server restarting mid-run is retried by the client.

## Then

Pair the row with `posterior.py --reports A B` (B minus A) against the best local system
`large-v3_gpu_fp16_hint_rd5d_corr__qwen3-4b-instruct` (96.6 / 95.3) and the cloud stack
`groq_hint_corr__openai/gpt-oss-120b` (96.8 / 95.1), and write it up with the `matrix-row` skill.
