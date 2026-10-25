# Baseline run — HANDOFF.AI pipeline over the MoCo call-set corpus

| | |
|---|---|
| start time | 2026-10-24T12:38:20-07:00 |
| app repo | ~/EMS-Handoff-app, branch handoff-app |
| commit | abff90003cfa68246283b9ecb9612fa925ded4aa (abff900) — "Ollama: prefer qwen3:4b-instruct; count a loaded model's memory as free" |
| working tree | 0 modified tracked files; untracked: scripts/benchmark.py tests/gold/  |
| report LLM provider | ollama (local, Jetson Orin Nano 8 GB) |
| model | qwen3:4b-instruct (chosen by the app's memory rule; qwen3:8b does not fit) |
| transcription (secondary path) | Groq whisper-large-v3-turbo via the app's auto transcriber |
| prompt / pipeline code | unchanged for the whole batch (frozen at the commit above) |
| inputs | 23 call-sets (~/callsets/CS*/combined_transcript.txt), 26 bursts >= 40 words (~/segments/*.wav) |
| runner | ~/labels/scripts/stage5_reports.py --all ; log ~/labels/stage5_log.csv |

CS001_combined was run alone first at 12:19:35 (60 s wall, 57.4 s LLM) and is part of this baseline.

## Incident during the run

At 12:42:34 the app picked `llama3.2:3b` for CS006_combined: it re-picks the Ollama model per
report from current free memory, and with qwen3:4b-instruct loaded the Jetson has ~4.8 GB usable
against the ~3.9 GB the app requires, so a dip hands the report to a smaller model. The batch was
stopped at CS007, the CS006 output set aside in ~/reports/_rejected/, and the runner now pins
OLLAMA_MODEL=qwen3:4b-instruct and rejects any report whose `_meta.model` differs (retry up to 3x).
Pipeline code and prompt were not changed. Restarted 2026-10-24T12:44:49-07:00 from CS006.


## Results (2026-10-24 13:28)

```
runs logged: 56; ok 45 (primary 20, secondary 25); not ok 11
  rejected_wrong_model   primary   CS006_combined   llama3.2:3b        
  rejected_wrong_model   primary   CS011_combined   llama3.2:3b        app picked llama3.2:3b; attempt 1
  failed                 primary   CS015_combined                      Report failed: The report is too short to process. Please say or paste more detail.
  rejected_wrong_model   primary   CS016_combined   llama3.2:3b        app picked llama3.2:3b; attempt 1
  failed                 primary   CS018_combined                      Report failed: The report is too short to process. Please say or paste more detail.
  failed                 primary   CS022_combined                      AttributeError: 'NoneType' object has no attribute 'extend'
  rejected_wrong_model   secondary S040_01          llama3.2:3b        app picked llama3.2:3b; attempt 1
  rejected_wrong_model   secondary S022_01          llama3.2:3b        app picked llama3.2:3b; attempt 1
  rejected_wrong_model   secondary S054_01          llama3.2:3b        app picked llama3.2:3b; attempt 1
  failed                 secondary S007_01                             AttributeError: 'NoneType' object has no attribute 'extend'
  rejected_wrong_model   secondary S001_01          llama3.2:3b        app picked llama3.2:3b; attempt 1
models in accepted reports: {'qwen3:4b-instruct': 45}
total wall time, all attempts incl. rejected: 51.8 min (3108 s); accepted runs only: 46.5 min
batch clock span (first to last log entry after CS001): 48.6 min, including the stop/restart
primary   LLM time: mean 52.7 s, median 51.5 s, worst 68.1 s (CS011_combined), best 38.5 s; n=20
secondary LLM time: mean 50.5 s, median 48.8 s, worst 89.8 s (S001_01), best 26.7 s; n=25
secondary wall (incl. Groq transcription): mean 63 s, worst 155 s
data_quality.confidence over 20 combined reports: {'MEDIUM': 19, 'HIGH': 1}
alert_type: {'STROKE': 7, 'TRAUMA': 5, 'CARDIAC': 4, 'ENDOCRINE': 3, 'PEDIATRIC': 1}
alert_level: {'RED': 16, 'YELLOW': 4}
per set: CS001=MEDIUM/STROKE/RED, CS002=MEDIUM/STROKE/YELLOW, CS003=MEDIUM/STROKE/RED, CS004=MEDIUM/TRAUMA/RED, CS005=MEDIUM/TRAUMA/RED, CS006=MEDIUM/STROKE/RED, CS007=MEDIUM/CARDIAC/RED, CS008=MEDIUM/CARDIAC/RED, CS009=MEDIUM/TRAUMA/RED, CS010=HIGH/STROKE/RED, CS011=MEDIUM/TRAUMA/YELLOW, CS012=MEDIUM/ENDOCRINE/YELLOW, CS013=MEDIUM/ENDOCRINE/RED, CS014=MEDIUM/PEDIATRIC/RED, CS015=MISSING//, CS016=MEDIUM/TRAUMA/YELLOW, CS017=MEDIUM/CARDIAC/RED, CS018=MISSING//, CS019=MEDIUM/STROKE/RED, CS020=MEDIUM/CARDIAC/RED, CS021=MEDIUM/ENDOCRINE/RED, CS022=MISSING//, CS023=MEDIUM/STROKE/RED
```

## After the baseline (2026-10-24, later the same day)

The baseline above is frozen; the notes here explain why the corpus no longer matches its input list.

- **CS015 retired, its burst joined into CS014.** S055_01 ("How old? Sorry.", feed 4430, 07:41:12 UTC) is the
  hospital asking the child's age 10 s into the Medics 729 pediatric consult on feed 4390. Stage 4 had recorded the
  relation but kept it apart under the never-join-across-feeds rule. On review it was joined (callset_relations.json
  `"join": true`); CS014 is now 5 bursts, 234 words, in time order. Set IDs are stable: CS016–CS023 keep their numbers,
  CS015 is retired (folder in ~/callsets/_retired/). The baseline CS014_combined report was made from the 4-burst text.
- **Pipeline no longer refuses short transcripts** (the CS015 and CS018 failures above). Every transcript is processed;
  an empty one returns a complete UNCLEAR report. Null lists from the model (the CS022 and S007_01 failures) are now
  filled in. Post-fix check reports live in ~/reports/_postfix_check/, never here.
- **Stage 5 runner no longer skips bursts under 40 words.** The post-fix run covers all 51 bursts.
