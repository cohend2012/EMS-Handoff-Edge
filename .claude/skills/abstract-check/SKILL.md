---
name: abstract-check
description: Check a draft abstract, slide, or summary of the EMS-Handoff work claim by claim against the evidence reports, and produce a corrected draft. Use whenever the user pastes an abstract or a co-author's text, asks "is this number right", "can we say X", "what can we report", or wants numbers for a paper, poster or email; every number must trace to a row in docs/reports or be marked as having none.
---

# Abstract check

Co-authors write from memory and from earlier drafts; numbers drift, pairings swap (allergies 50 to 0 and
code status 23 to 0 became "code status 50 to 0"), and claims outrun rows ("hallucination-free", "14 s in the
cloud"). The check is mechanical: one table, one verdict per claim, then a corrected draft.

## Sources, in the order to look

1. `docs/reports/SYSTEM_VS_BASE_2026-10-28.md`: sections 2 (transcripts), 4 (reports), 6 (latency), 7 (claim table).
2. `docs/reports/EVIDENCE_LEDGER_2026-10-28.md`: every run with size, time and headline; section 6 is the
   one-paragraph state of the evidence.
3. `docs/reports/STUDY_2026-10-28.md`: the current abstract and results wording.
4. Older per-topic reports in `docs/reports/` (BASELINE, POSTFIX, ROUND2, REAL_STACK, HINT_ABLATION, NGRAM,
   STATE_REPLAY, AIR, DEMO) for the real-corpus numbers.
5. `docs/reports/ABSTRACT_REVIEW_2026-10-29.md`: the last review, so corrections are not re-derived.

## The table

```
| draft says | evidence | verdict |
```

One row per numeric or categorical claim. "Evidence" names the report and the row (with device and sample
size); "verdict" is one of: correct; correct, add the sample size / device; update to the current row (give
it); wrong pairing (give the right one); no row (say what would produce one); overclaim (give the defensible
wording). Watch for these recurring ones:

- sample sizes: the 1,000-clip set was *scored* on the 150-clip test split and 60 reports; real WER rests on
  3 referenced calls of the 22;
- "hallucination-free" or "zero fabrications": rows show 0.03 to 0.24 per report; "abstains rather than
  invents" holds for the two rule-protected fields only;
- cloud latency: no measured row; component rates give about 10 s;
- speed claims: the stack is not faster than the bare model on the same hardware; the GPU is the speed lever;
- the local-against-cloud comparison: name the model class (large-v3 against large-v3) and the device.

## The corrected draft

Keep the author's structure and voice; replace numbers with current rows; add the interval where a difference
is claimed; name the devices once. Write it after the table in the same file, dated. Then tell the user in a
short list which corrections matter most and which numbers moved since their draft.

## Current headline numbers (10-30)

Local transcriber: 5.0 / 7.0 % WER on the CPU, 4.6 / 5.5 % large-v3 on the second Jetson's GPU; cloud best
3.9 / 4.9 %. Term recall 87.1 to 90.5 % (P = 0.99). Real audio 25.7 to 22.7 % (CPU stack), GPU large-v3 stack
21.7 %. Reports: local 96.6 / 95.3 against cloud 96.8 / 95.1 (-0.1 [-2.4, +2.0]); rules worth 5.8 / 5.4;
lexicon 1.6 / 0.4. Latency: 143 s one box offline, 70 to 75 s split pipeline. Check the matrix before quoting
these; they move.
