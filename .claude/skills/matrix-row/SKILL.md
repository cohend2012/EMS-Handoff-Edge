---
name: matrix-row
description: Add a measured result to the EMS-Handoff evidence reports. Use whenever an eval or report run finishes and its numbers need to go into docs/reports/SYSTEM_VS_BASE_2026-10-28.md (the matrix), EVIDENCE_LEDGER_2026-10-28.md, STUDY_2026-10-28.md or the plans, or whenever the user asks "how does X compare to Y", "is that better than the cloud", "add that row", or wants a paired interval. Covers the posterior.py conventions, which labels pair with which, and how a row is worded.
---

# Matrix row with a paired interval

Every number in the reports has a row, a device, and an interval against the row it replaces. A result is
not "in" until it has all three; the user's co-authors read these files directly.

## Where results live

| thing | path |
|---|---|
| transcript evals | `~/datasets/synth_v1/evals/<label>/{rows.tsv,summary.md,transcripts/}` |
| report (writer) runs | `~/datasets/synth_v1/reports/<eval label>__<model>/{rows.tsv,summary.md,*.json}`; Groq labels are paths: `groq_hint_corr__openai/gpt-oss-120b` |
| the matrix | `docs/reports/SYSTEM_VS_BASE_2026-10-28.md`: section 2 transcripts, section 4 reports, section 6 latency |
| the ledger | `docs/reports/EVIDENCE_LEDGER_2026-10-28.md`: section 2 is every run in order (add a row), 3 to 5 are the summaries |
| the study | `docs/reports/STUDY_2026-10-28.md`: 5.1 transcripts, 5.2 real bursts, 5.3 reports, 5.5 latency, abstract |

## Pairing

```bash
.venv/bin/python scripts/corpus/posterior.py --evals   A B      # transcript rows: WER, term recall
.venv/bin/python scripts/corpus/posterior.py --reports A B      # writer rows: weighted, recall, precision
```

The header prints "`B` against `A`"; **mean diff = B − A**. Put the thing being tested as B and the row it
replaces as A, so a positive difference means B is better on weighted score and worse on WER. Report the
clip-level bootstrap interval, never the posterior alone (terms in a clip fail together, the bootstrap is the
conservative reading). Pair on the same clips: the tool intersects ids, so a 59-of-60 run pairs on 59.

Standard comparisons:

- a new local transcriber against the shipped default `medium_hint_rd5d` (CPU) or `medium_gpu_fp16_hint_rd5d` /
  `large-v3_gpu_fp16_hint_rd5d` (GPU), and against its own plain-decode base `*_nohint`;
- a new writer row against `large-v3_gpu_fp16_hint_rd5d_corr__qwen3-4b-instruct` (the best local system,
  96.6 / 95.3) and the cloud stack `groq_hint_corr__openai/gpt-oss-120b` (96.8 / 95.1).

## Writing the row

Section tables take one line per configuration; copy a neighbouring row's columns. Name the device in the
row ("vox-ai CPU int8", "cohendynamics GPU fp16", "Groq"), the stack ("hint + re-decode + corrector"), and the
count when it is not 60 ("59 of 60, SY117 at 5 dB malformed JSON"). Then one paragraph after the table, in
the house style: the number, the pair, the interval, what it means, and what remains untested. Example:

> The lexicon-retrieval row (10-30): switching it off on the same transcripts gives 95.0 / 94.7 against
> 96.6 / 95.1: the retrieval is worth +1.6 weighted points at clean [-0.2, +3.3] and +0.4 at 5 dB [-1.3, +2.0].

Hint rows from before 10-27 (the 4.8 % medium row) used an older hint and are not comparable with later rows;
say so when they are mentioned together.

## Then

Add the ledger row (date, run, size, time, headline with interval, where), update the study if the headline
changes a sentence there (abstract included), commit with the result in the message, and push to both remotes
as the `hourly-checkin` skill describes. Note the result in the matrix-runs memory file.
