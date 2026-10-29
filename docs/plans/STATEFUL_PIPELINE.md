# Plan: a stateful pipeline, dynamic hints and prompts (and why not LangChain)

Written 2026-10-29, in answer to: what is wrong with the 8B on the Jetson, what is our best system today,
and would a LangChain or state-machine design with dynamic RAG and prompting make it better.

## 1. The 8B on cohendynamics: it does not fit

Measured 16:53 on 2026-10-29 while the 8B report run was at 87 of 120:

| fact | value |
|---|---|
| box memory | 7.5 GB total, no swap, 166 MB available, largest free block 4 MB |
| qwen3:8b in Ollama | 6.0 GB, split 13 % CPU / 87 % GPU (Ollama could not place every layer on the GPU) |
| llama-server | 200 % CPU, GPU 72 to 99 % busy |
| other load | none: no container, the whisper bench waits, the CPU eval runs on the other box |
| seconds per report (87 done) | min 116, median 171, p90 270, max 1401; 26 timeouts at 420 s |
| the 4B on vox-ai | about 3 s per report, GPU only |

So it is not "too much running". The model plus its working memory is bigger than the box: an eighth of
the layers run on the CPU, the rest of the system lives in the 166 MB that is left, and when the page cache
is squeezed the weights are re-read from disk, which is the 1401 s report. The first 24 paired reports put
the 8B at 96.1 against 95.5 for the 4B (5 better, 4 worse, 15 ties), a point at best for 50 times the
time. Decision: the 8B is not a live writer on an 8 GB Orin. The run finishes (33 reports left, about
1.5 hours) because the paired comparison is nearly complete and nothing else needs that box until then;
then the GPU whisper bench starts. One more cheap test is worth it afterwards: `qwen3:8b` at a 3-bit quant
(about 4 GB) fits the GPU whole; if it keeps the 8B's precision it is a candidate, if not the 8B is an
offline judge and teacher only (section 4, distillation).

## 2. The best system we have today, and why

Every number from `docs/reports/SYSTEM_VS_BASE_2026-10-28.md` (same split, same scorer, device named).

| stage | best local today | the cloud | note |
|---|---|---|---|
| transcript, synthetic test, clean / 5 dB | medium int8 + hint + gap re-decode (beam 5 span) + corrector: 5.8 / 6.9 % | turbo + stack 4.2 / 6.4; large-v3 + stack 3.9 / 4.9 | with the SAME hint, greedy medium is 7.1 %; the 4.8 % row is the old hint |
| transcript, real calls | medium + stack 18 to 27 % by set | 10 to 33 % | the real-call gap is the one that matters |
| transcriber speed | 28 s per burst on the CPU (RTF 0.8) | 2 to 3 s | the GPU container on cohendynamics is the fix, bench queued |
| report, weighted | qwen3:4b-instruct + grounding rules: 93.7 to 95.7 | gpt-oss-120b + rules: 96.8 / 95.1 | the rules are worth 5.8 points; the 8B adds about 0.6 |
| report speed | 3 s on the Orin GPU (4B) | 3 s | even |

Why this stack: medium is the largest whisper that decodes on the CPU in under real time; the hint buys
terms; the gap re-decode catches the collapsed-window failure that any decode setting can trigger on some
clip (SYSTEM_VS_BASE, 10-29 rows); the corrector fixes the lexicon; the 4B is the largest writer that
fits the GPU with whisper beside it, and the grounding rules make it nearly as grounded as the 120B.

Two things learned today that change where the effort goes:

- **The hint is a lever, both ways.** A six-character change to the whisper prompt moved medium from
  4.8 to 7.1 % clean by collapsing one window and un-collapsing another (SYNTH_V1_2026-10-27.md). The
  hint is the cheapest thing we have never optimized.
- **The failure is the 30 s window, not the model.** turbo with beam 5 on the whole clip is 6.2 / 8.1 % at
  greedy speed; the re-decode is being tuned to get the same from greedy plus a beam only where it is needed.

## 3. What to review this week

1. The call review page (every real burst, every stack, the gold samples): mark which stack's errors are
   terms, numbers, or whole sentences. That decides between hint work, corrector work and model work.
2. The 8B against 4B paired table when it finishes, with its failure count: decide 8B's role (section 1).
3. The matrix rows `large-v3-turbo_hint_rd5b` and `medium_hint_rd5b` (the widened re-decode trigger).
4. The hint: run the test split with the old hint, today's hint and a hint without the alias that flipped,
   on medium; the cheapest accuracy experiment available.
5. The GPU whisper bench on cohendynamics (seconds per burst against 28 s).

## 4. LangChain, a state machine, dynamic RAG and prompting: what helps and what does not

**LangChain: no.** The pipeline is ten small Python modules with a test for each
(`transcribe.py`, `transcript_corrector.py`, `incident.py`, `lexicon_rag.py`, `textbook_rag.py`,
`prompt.py`, `llm.py`, `pipeline.py`). LangChain would add a dependency tree the size of the app, an
abstraction over calls we already make in a line, and it hides the exact prompt and the exact retrieval,
which is what we measure. Everything below is done in those modules, under the same scorer.

**A state machine: yes, and we half have one.** `incident.py` already runs one session per patient,
appends bursts, triggers extraction on quiet or a closing cue, and re-extracts the report from the whole
session transcript. What it lacks is typed state. Proposal, "slot state":

- The report is a typed record (unit, destination, age, sex, chief complaint, priority, vitals with time,
  interventions, alerts, ETA, allergies, history), each slot with a value, the burst it came from, the
  transcript span, and a confidence (whisper's average log probability over that span, the corrector's
  edit, the writer's own flag).
- Each burst produces a delta (new or changed slots), not a new report; the report is rendered from the
  state. The writer is asked only for the slots that changed or are still empty, with the state in the
  prompt as what is already known, so a 4B has a short, exact job per burst.
- States: `listening` (no session), `open` (bursts arriving, slots filling), `quiet` (extraction due),
  `closing` (closing cue heard, ETA known: final render, alert check), `handed off`, with the triggers
  already in `incident.py`. Each transition is logged with its cause, so a review can replay the call.
- Proof: the same 60 synthetic reports and the 51 real bursts, scored by the same weighted scorer, 4B with
  slot state against 4B as today (whole-transcript re-extraction); hallucination per report and the time
  per burst are the two numbers that must improve. Expectation: fewer hallucinations (the writer cannot
  invent a slot it was not asked for) and a 2 to 5 s report after each burst instead of a full rewrite.

**Dynamic RAG: yes, in two places.**

- *Dynamic whisper hint.* `whisper_hint()` is static (800 characters of terms). After the first burst the
  session knows the call family (stroke, STEMI, trauma, sepsis, OB, psychiatric, arrest) and the region
  (Montgomery County units and hospitals); the hint for the next bursts is built from that family's term
  list and the named hospital and unit, and the re-decode of a flagged window uses the same hint. Proof:
  MTER (missed terms) on the test split and the real bursts with the static hint against the dynamic one,
  the same decode otherwise; today's hint instability (section 2) says this must be measured per clip, not
  on the mean alone.
- *Dynamic writer context.* `lexicon_rag.py` and `textbook_rag.py` retrieve by the whole report text.
  Retrieve per slot instead (the drug table for interventions, the alert criteria for the alert slot, the
  stroke scale for a stroke call), and pick the one or two gold reports of the same family as examples.
  Proof: weighted score and hallucination on the 60 reports, rules on, 4B, static against dynamic context.

**Dynamic prompting: yes, driven by the state.** The prompt is assembled from the state: the known slots,
the slots wanted, the family's rules, the retrieved context for those slots, the examples for that family.
That is what makes the 4B competitive with the 120B at the grounding, and it is measured the same way.

**Distillation (uses the 8B's one strength).** The 8B and the 120B score the 60 synthetic reports and the
real bursts offline, with no time limit; their reports become training targets for the 4B (LoRA, on the
train split). The 4B then runs live with the 8B's judgement. Proof: the test split, 4B base against 4B
tuned, same rules, same scorer.

## 5. Order and cost

| step | work | proof run | where |
|---|---|---|---|
| 1 | hint experiment (old, today's, no-alias) on medium | 3 x 60 clips, CPU, 1.5 h | vox-ai |
| 2 | finish the re-decode tuning (rd5b rows), ship as default | running | vox-ai |
| 3 | GPU whisper bench, then large-v3 fp16 on the GPU | queued | cohendynamics |
| 4 | slot state in `incident.py` + per-slot writer prompt | 60 reports + 51 bursts, 4B | vox-ai GPU |
| 5 | dynamic whisper hint by family | 60 clips x 2 hints | vox-ai CPU or the GPU container |
| 6 | per-slot retrieval and family examples | 60 reports | vox-ai GPU |
| 7 | 8B/120B offline targets, 4B LoRA | train split, then test | cohendynamics (torch in the container) |

Each step lands as rows in the matrix with its posterior, as the goal document asks.
