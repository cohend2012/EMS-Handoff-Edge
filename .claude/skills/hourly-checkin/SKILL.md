---
name: hourly-checkin
description: The EMS-Handoff hourly check-in. Use whenever a cron or the user asks to "check in", "push what's done", "send the hourly email", or report status on the two Jetsons (vox-ai and cohendynamics): gathers job state in one or two Bash calls, commits and pushes work in progress to both remotes, and sends one plain-text status email to cohend2012@gmail.com under 250 words. Also use for any end-of-hour or end-of-day status report on this project, even when the word "hourly" is not said.
---

# Hourly check-in

One pass, three steps, one email. The user reads only the email, so its numbers must be the ones that matter
and its hashes must be real. Never start new jobs from a check-in: the main session owns job completions.

## 1. Gather state (one or two Bash calls)

On vox-ai:

```bash
pgrep -af "report_set|eval_set|state_replay|queue_|ems_handoff.cli" | grep -v pgrep
```

For every running or recently finished job, the count out of total. Report rows are JSON files under
`~/datasets/synth_v1/reports/<label>__qwen3-4b-instruct/` (60 = complete); eval rows are
`~/datasets/synth_v1/evals/<label>/summary.md` ("| inf |" and "| 5 |" lines); replay and queue logs live in
`~/datasets/synth_v1/logs/`. Grep the logs for `\[error\]` and `Traceback`.

On cohendynamics (key SSH, `ssh -o BatchMode=yes cohendynamics@100.77.24.78`): `docker ps` (the whisper server
container `ems-whisper-server` should be Up), `pgrep -af "run_gpu|eval_set|report_set"`.

The Groq cloud rules-off row is still partial (`~/datasets/synth_v1/logs/matrix_2026-10-29_cloud_base.log`);
mention it under BLOCKED while it stays at 13 of 60.

`git status --short` and `git log --oneline -5` in `/home/voxai/EMS-Handoff-app`.

## 2. Commit and push

Commit uncommitted work under `docs/`, `scripts/`, `ems_handoff/`, `tests/`, `.claude/` with a message that
says what changed and why (the project's commit messages carry the result and its interval, not "update docs").
Never commit `.env`, keys, recordings, audio, or anything under `output/`. End every message with:

```
Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01CfqT4XngfcAP3UhSRZy3Rh
```

Then `git push origin main` and `git push handoff main:two-jetson-horsepower-2026-10-29`. The handoff remote
is the user's own repo and takes a dated feature branch, never its main. Never force-push, rebase, or resolve
a conflict by discarding work; on a push error, report the error text and leave it.

## 3. The email

`mcp__claude_ai_Gmail__send_message` to cohend2012@gmail.com. Subject `EMS-Handoff hourly HH:MM: <three-word
gist>`. Plain text, no markdown, under 250 words, four sections in this order:

```
DONE SINCE LAST HOUR
<what finished, with the numbers that matter (score, interval, seconds); what was committed and pushed, hashes and branch>

RUNNING NOW
<each job, progress count out of total, where it runs; or "Nothing. Both Jetsons are idle.">

BLOCKED OR FAILED
<the error line, or "nothing">

NEXT
<what the session will do next>
```

If nothing changed since the last email, say so in one line under DONE and still send. Write numbers as the
reports do: "96.6 / 95.3 weighted", "+1.6 [-0.2, +3.3]", "70 to 75 s against 135 s".

## Why these rules

The user follows the project from a phone between other work. One email an hour with real hashes lets them
trust the repo state without opening it; a check-in that starts jobs would race the main session's waiters,
and a check-in that pushes to the handoff repo's main would overwrite the user's own branch work.
