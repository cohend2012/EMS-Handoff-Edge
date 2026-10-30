# Plan: the live system, its state on screen, and asking the operator (OODA without stepping on toes)

Written 2026-10-29, late. Extends `STATEFUL_PIPELINE.md` (which rejected LangChain and proposed slot
state). This document answers: what runs live, where Vosk went, what the state machine should show, when
the system may ask the crew for something, and what each framework under discussion would buy us.

## 1. The live system today (one Jetson), audio in to report out

```
  radio / mic audio (16 kHz)
      |
      v
  capture thread ............ blocks of PCM, level meter
      |
      +--> [Vosk captioner] ......... words on screen within ~0.3 s ("rough captions")
      |         (RepairingCaptioner: whisper re-does each phrase a few seconds later, when online)
      |
      +--> [Vosk phrase spotter] .... hears the stop phrase "confirm report ready" -> ends the recording
      |
      v
  burst ends (operator Done, stop phrase, or the quiet period: 25 s offline, 5 s online)
      |
      v
  [whisper medium, CPU int8] hint + span re-decode ........ 28 s per 40 s burst   (GPU: 4 to 7 s)
      |
      v
  [corrector] lexicon fixes, transcript checks ............. < 1 s
      |
      v
  [IncidentManager]  one session per patient; the burst joins the open session
      |   triggers: closing cue heard ("see you on arrival", "... clear"), quiet period, New patient
      v
  [writer qwen3:4b-instruct, GPU] whole-session transcript -> 25-field JSON ... 50 s
      |   grounding rules: abstain, no invented allergy / code status, level from stated priority
      v
  report on screen + "what this burst changed" lines + a copy of every report kept
```

**Vosk's place.** Vosk is the *caption* engine, not the *transcript* engine. It is a 40 MB streaming
Kaldi model that puts rough words on the screen while the crew is still talking and runs the stop-phrase
spotter. Nothing downstream reads its words: the transcript that the corrector, the checks and the writer
see comes from whisper over the finished burst. It was measured once (NGRAM_2026-10-26.md, three real
calls): 69 % WER, 0.22 average n-gram F, "captions, not a transcript", which is why it never got the
transcript job. If the Vosk model is missing the app falls back to a whisper captioner (`doctor.py` warns).
So: still in the live app, on screen only, never in an evaluation row, and the right role for it.

## 2. Why state, and what it is worth in seconds

The writer is now the whole wait (section 5.5 of the study; the split pipeline makes whisper 4 to 7 s per
burst). Its 50 s is mostly *output*: a report JSON is about 4,400 characters, roughly 1,100 tokens, and the
4B writes about 24 tokens a second on the Orin Nano, so about 45 of the 50 s are spent re-writing fields
that did not change. Every burst re-writes the whole report from the whole transcript.

With typed slot state (STATEFUL_PIPELINE.md section 4) a burst asks the writer only for the slots it can
have changed, as a delta of perhaps 5 to 15 fields, 100 to 250 tokens, 5 to 10 s. The report on screen is
rendered from the state, not from the model's last output. The expected gain is therefore 5x on the
per-burst report time, and the first proof run is exactly that number on the 51 real bursts, together with
the weighted score and hallucinations per report on the same 60 synthetic reports, against today's
whole-rewrite writer. Accuracy may improve for a second reason: a writer asked for five named slots cannot
invent a sixth, and the rules can check each slot against its own transcript span.

## 3. The state machine, shown on screen

```
 listening ──first burst──> open ──quiet / closing cue──> extracting ──delta applied──> open
                              │                                                          │
                              │ closing cue + required slots filled                      │
                              └──────────────────────────> closing ──operator confirms──> handed off
                                                               │
                                                               │ required slot empty
                                                               v
                                                            asking  (one line on screen, see section 4)
```

Each state has a cause string ("quiet for 25 s after burst 3", "closing cue in burst 4", "New patient
pressed") and the app already publishes state events over its event stream; the screen gains:

- a state badge (`listening`, `open: burst 3 of this call`, `writing (burst 3)`, `closing`, `handed off`);
- the slot board: every slot with its value, the burst it came from, and a mark for empty-but-required;
- the change lines per burst (already there), now derived from the slot delta rather than a text diff.

Showing the state is what makes the OODA reading honest: observe (bursts in), orient (slots filled, slots
missing for this alert type), decide (ask or wait), act (one line on screen, never on the air).

## 4. Asking the operator: when, how, and the toe rule

The system never transmits and never speaks. Its only act is a line on the receiving facility's screen
(and, later, a short tone). The rules:

| moment | condition | what the screen shows | who acts |
|---|---|---|---|
| closing cue heard ("see you on arrival") or Done pressed | a slot *required for the declared alert type* is empty (STEMI: 12-lead finding, aspirin given; stroke: last known well, LAMS; trauma: mechanism, GCS; any: age, sex, ETA, unit, destination) | `Missing for STEMI: last 12-lead time, ETA` | the receiver asks on the radio, or lets it go |
| burst ends mid-call | a value conflicts with an earlier burst (two different ETAs, two ages) | `ETA 10 (burst 2) vs 5 (burst 4)` with both sources | the receiver confirms |
| any burst | a drug dose outside the lexicon's range, or an intervention with no drug | `Check: epinephrine 10 mg?` | the receiver confirms |
| quiet period passes with the alert type still unknown | | `No alert type stated` | nothing required |

Never: a question while a burst is being spoken; a question for a slot that is optional for the alert type
(no "what about allergies?" on a trauma alert); more than one open line at a time; anything that changes a
value on its own. The required-slot table is data (`lexicon/required_slots.yaml`), reviewed by the
clinical co-authors, and every ask is logged with its cause so a review can count asks per call and asks
that were right. Proof: on the 22 real calls, asks per call, and for each ask whether the gold report has
the slot (an ask was right) or not (noise); target under one ask per call with precision above 80 %.

## 5. The frameworks: what each would give us, and the verdict

| candidate | what it is | what it would do here | verdict |
|---|---|---|---|
| LangChain | chains of model calls with retrievers and tools | wraps calls we make in one line; hides the prompt we measure | no (STATEFUL_PIPELINE.md) |
| LangGraph | a Python graph of nodes with typed state, checkpoints, and "interrupt" for human input | the ask-the-operator pause is its interrupt; checkpoints give replay of a call | borrow the design, not the package: `incident.py` already has sessions, triggers and a worker in 300 tested lines; a 1-day spike re-expressing it in LangGraph is cheap and gives a measured comparison (lines, test count, latency overhead per burst, memory) |
| XState | a JavaScript statechart library | would model the *screen's* state in the web UI | useful only if the UI grows beyond the badge and slot board; the state of record stays in Python |
| a hand-written statechart | states, transitions, causes, in `incident.py` | what we have, plus typed slots and the `asking` state | yes, this is the plan |
| dynamic RAG | retrieve per slot and per call family, not per whole report | shorter, exact context for the 4B; the family's term list into whisper's hint after burst 1 | yes, measured per clip (STATEFUL_PIPELINE.md section 4) |
| dynamic hints | whisper prompt built from the session (family, unit, hospital) | fewer missed terms in later bursts | yes; note the local hint is cut to its last 223 tokens, so the hint must be short and the terms must be at its end (today's ablation on the GPU: trimming or shortening the hint changed nothing on the synthetic set, 4.6 to 4.7 % WER on large-v3) |
| dynamic prompts | the writer prompt assembled from the state: known slots, wanted slots, the family's rules and examples | the 4B gets a short exact job per burst | yes, it is the slot-state writer |

So the study of "stateful behaviour" is not a library choice. It is three measurable changes to the code
we have: typed slot state with delta writes, state and slots on the screen, and the asking rule. LangGraph
gets one day as a control, so the paper can say the hand-written machine was compared with a framework.

## 6. Order, with the proof for each step

| step | work | proof | where |
|---|---|---|---|
| 1 | split pipeline live: whisper on the cohendynamics GPU, writer on vox-ai; time to report on CS008 and the 22 calls | seconds from last burst to report, against 143 s. 10-30: built (`HANDOFF_TRANSCRIBER=remote`, `scripts/whisper_server.py` in the container, commit a5af70f); CS008 end to end 70.4 / 71.0 / 74.5 s against 134.6 s on one box the same morning (transcript 15 to 21 s, writer 50 to 52 s): done, the writer is the remaining wait | both boxes |
| 2 | slot state in `incident.py`: typed record, per-slot source span, delta writer prompt, render from state | 60 synthetic reports + 51 real bursts: weighted, hallucinations, s per burst, against the whole-rewrite writer | vox-ai GPU |
| 3 | state badge and slot board on screen; required-slot table; `asking` state and the closing-cue check | asks per call and ask precision on the 22 calls; a short review with a medic co-author. 10-30: the state line and the ask rule are in (commits 820e6b3, 8d06a8d); the cumulative replay (10-30): 8 closing cues, 5 asks, 4 of 5 naming items the crew did not state, the fifth from a wrong alert type (STATE_REPLAY_2026-10-29.md) | vox-ai |
| 4 | dynamic hint by family after burst 1; per-slot retrieval and family examples | MTER per clip and weighted score on the same sets | GPU container, vox-ai |
| 5 | LangGraph spike, same behaviour, one day | lines, tests, latency per burst, memory | vox-ai |
| 6 | 8B / 120B offline targets, 4B LoRA (STATEFUL_PIPELINE.md) | test split, 4B base against tuned | cohendynamics |

Each step lands as rows in `docs/reports/SYSTEM_VS_BASE_2026-10-28.md` with a paired interval, as the goal
document asks, and nothing in the live app changes until its row is in.
