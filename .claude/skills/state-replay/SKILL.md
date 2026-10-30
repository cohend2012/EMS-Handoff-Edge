---
name: state-replay
description: Replay the EMS-Handoff real call sets through the incident manager and the patient-state module, and judge the flags and asks against the transcripts. Use whenever the state module (ems_handoff/state.py), the ask rule, the required-slot checklists or the incident triggers change, whenever the user asks how often flags or asks fire on real calls, or whether an ask was right; produces docs/reports/STATE_REPLAY_*.md.
---

# State replay over the real calls

Two replays answer two questions. Run the cheap one on every change to `state.py`; run the real one when the
incident manager, the writer or the checklists change.

## 1. Cheap: the state over saved reports (seconds, no model)

Script pattern (session scratchpad `state_replay.py`, re-create if missing): for each `~/callsets/CS*`, take
the per-burst reports already on disk (`~/reports_round2/<segment>.json`, written per burst on 10-24) in
burst order, call `ems_handoff.state.build_state([("burst i", None, report), ...])`, and tabulate flags,
crossings, missing items, conflicts. Caveat to carry into the write-up: those reports were written from each
burst alone, so a later two-word reply yields a near-empty report; the state tolerates that (empty values never
overwrite, UNCLEAR is not a type change, the checklist runs over merged slots) but conflicts counted there are
mostly artefacts.

## 2. Real: the incident manager as the live app runs it (about 70 s a call)

```bash
OLLAMA_MODEL=qwen3:4b-instruct HANDOFF_PROVIDER=ollama .venv/bin/python scripts/corpus/state_replay.py --out ~/reports_state
```

Bursts at their real epochs from `callset_info.json`, the 25 s offline quiet period, closing cues, cumulative
reports, the 4B writer with rules, then the state and the ask rule. Output `~/reports_state/<CS>.json`
(bursts, reports, events, final state, ask) and `summary.md`. Resumable per call set. Needs vox-ai's Ollama,
so queue it like a writer row (the `writer-row` skill).

## Judging the asks

For every ask, read the call's saved transcripts (`~/callsets/<CS>/S*_01.txt`) and decide:

- **right**: the item is not stated anywhere in the call (a stroke call with no last-known-well, a trauma with
  no respiratory rate);
- **wrong upstream**: the writer's alert type is wrong, so the right checklist was applied to the wrong call
  (CS002: "paramedic CVA on board" was a misheard unit name, not a stroke); say so, the rule is not at fault
  but the line's "for STROKE:" prefix is what lets the receiver discard it;
- **noise**: the call is a fragment with nothing to ask about.

Report asks per call, right / wrong-upstream / noise, and which items lead the missing list. On 10-30 the
cumulative replay gave 8 closing cues, 5 asks, 4 right, 1 wrong upstream, with anticoagulant status, last
known well, glucose and a stroke scale leading on stroke calls and respiratory rate on trauma.

## Write-up

Append a dated section to `docs/reports/STATE_REPLAY_2026-10-29.md` (or a new file for a new month): the
command, totals in a code block, bullets for what changed and why, the ask table, the per-call table from
`summary.md`. Update the status cells in `docs/plans/LIVE_STATE_PLAN.md` section 6 and the goals doc, commit
with the numbers in the message, push both remotes.

## Why

The ask rule is the first feature that speaks to the crew's omissions, so its precision must be measured on
real calls before it is trusted, and the thresholds in `state.py` cite published rules that the clinical
co-authors still have to confirm (the adult SBP line in the 2021 field triage guideline is marked for that).
