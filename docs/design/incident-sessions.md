# Incident sessions: building the patient picture burst by burst

Status: design note, agreed 2026-10-24. Implementation starts on the replay branch after
steps 1 to 5 of the current plan are finished and the post-fix corpus is scored.

## Problem

A radio call to the hospital is not one transmission. It is a run of short bursts over a
few minutes: the medic opens, the hospital answers, a physician is fetched, the full
report follows, questions go back and forth, the call closes. Today the app turns one
transcript into one report and keeps no state between reports. In the corpus batch the
bursts were stitched offline by hand-tuned rules, and a three-word burst from the hospital
side ("How old? Sorry.") was scored on its own and came out UNCLEAR. The words were fine;
the grouping was wrong. The product needs to hold a growing state of each patient and
update it as the radio traffic arrives.

## What the corpus says

Numbers are from the 22 call-sets and 51 bursts of the MoCo corpus (`~/labels/callsets.csv`,
`~/callsets/CS*/callset_info.json`, Stage 5 log), as of 2026-10-24.

| Measure | Value |
|---|---|
| Gap between bursts of the same call (n=29) | median 8 s, max 114 s; only 4 gaps over 30 s |
| Gap between calls on the same feed (n=19) | median 30 min, min 84 s; 3 gaps under 5 min, all with conflicting unit, age or sex |
| Words per call | median 172, max 396 |
| Audio per call | median 83 s, max 161 s |
| Span first burst to last | median 110 s, max 5 min |
| Bursts per call | 10 calls of 1 burst, 12 calls of 2 to 7 bursts |
| Words per burst | median 40; 25 bursts under 40 words, 11 under 15 |
| Local report time, qwen3:4b-instruct on the Jetson (n=45) | median 51 s, min 27 s, max 90 s |
| Cross-feed overlap | the hospital side of CS014 came on feed 4430 and overlapped the medic's transmission by 13 s |

Three conclusions follow. A whole call fits in a few hundred words, so re-reading all of it
costs nothing. Bursts inside a call come seconds apart while calls are minutes apart, so a
quiet period separates bursts and a long gap separates patients. The local model is the
bottleneck at about a minute per report, so it cannot run once per burst offline.

## Design

### Sessions

An incident session is one feed's traffic about one patient. Each burst is transcribed as
soon as it ends and appended to the session transcript, which is shown on screen at once.
The report is always re-extracted from the whole session transcript, never from a burst
alone and never by patching the previous JSON. Re-extraction is stateless: nothing the
model invented from burst 1 is carried into burst 5, and every report can be checked
against the text that produced it. The incremental update pattern saves tokens, but there
are no tokens to save at 400 words, and it is known to accumulate errors.

### Boundary rule

A new session starts when any of these holds:

- the gap since the last burst on this feed is over 5 minutes;
- the burst names a different unit, or a patient age or sex that conflicts with the session;
- the operator presses **New patient**.

Otherwise the burst joins the open session. The operator can press **Merge with previous**
to undo a split the heuristic made; the sessions' bursts are re-sorted by time and one
re-extraction runs on the merged transcript. These two buttons are the only override of
the heuristic. They are logged with the burst they applied to.

### Trigger rules

Transcription runs per burst, always. Extraction is triggered by a quiet period on the
channel, with the length set by where the model runs:

| Mode | Model | Quiet period | Also trigger on |
|---|---|---|---|
| Online | Groq | 5 s | every burst, after the quiet period |
| Offline | Jetson, local Ollama | 25 s | a closing cue ("see you on arrival", "clear") at once |

Online the dashboard follows the radio by a few seconds. Offline the picture lags by about
a minute and a five-burst call like CS014 produces two or three reports instead of five.

### One job in flight

Each session has at most one extraction running. If a burst arrives while a job runs, the
job finishes, its report is shown, and one more job is queued that includes everything
received so far. Several bursts arriving during a job coalesce into that one queued job.
A job is never cancelled mid-run on the Jetson, since the model would have to be re-primed.
Across sessions the local model is shared and jobs run one at a time; the online path may
run sessions in parallel.

### Updating, not blank

While a re-extraction runs the previous report stays on screen with an **updating** marker
and the number of bursts it covers ("report as of burst 3 of 5, updating"). There is no
spinner and no empty dashboard. When the new report arrives it replaces the old one and
the marker clears.

### Burst attribution per field

The session transcript carries a header per burst (time, feed, burst number), as the
annotated corpus transcripts already do. The prompt asks the model to record, for every
field it fills, the burst number that stated it. A field with no attribution is shown as
**suspect**, visually distinct, since the model either inferred it or invented it. In the
corpus the 4B model's recurring "NKDA" allergies and code status would have shown up this
way.

### What this burst changed

When a report replaces an earlier one, the app diffs the two and shows a one-line summary
of the change ("burst 5 added: vitals, interventions, ETA 10 min; alert PEDIATRIC RED").
The history of reports for the session is kept so the operator can step back through it.

## Replay mode and the acceptance scenario

Replay mode feeds the bursts of a call-set into a session with their real gaps, compressed
by a configurable factor (1x for fidelity, 10x or 20x for a demo). The transcript grows
burst by burst and the report updates under the trigger rules above. Loop mode cycles
through call-sets in replay for visitors.

Acceptance scenario, CS014 (Medics 729, 3-month-old female, Shady Grove), real gaps shown:

| # | Time | Burst | Words | Expected on screen |
|---|---|---|---|---|
| 1 | 0:00 | medic asks for a pediatric physician, "three-month-old female, trouble breathing" | 47 | transcript shows burst 1; after the quiet period a first report: PEDIATRIC, age 3 months, female, complaint "trouble breathing", most fields null and listed as missing |
| 2 | 0:10 (overlaps burst 1, other feed) | hospital: "How old? Sorry." | 3 | transcript shows burst 2; online: re-extraction, report unchanged, change line "burst 2: no new facts"; offline: no trigger yet |
| 3 | 2:14 | medic: "729, you have my doctor yet?" | 7 | transcript grows; no new facts |
| 4 | 2:16 | hospital: "the number is on the way" | 9 | transcript grows; offline the 25 s quiet period after burst 4 triggers a run covering bursts 1 to 4 |
| 5 | 2:35 | full report: respiratory distress, drooling, SpO2 low 90s, oxygen, dexamethasone 2 mg, ETA 10 | 168 | report shows **updating** over the previous one; new report: PEDIATRIC RED, vitals, two interventions, ETA 10, each attributed to burst 5; change line "burst 5 added: vitals, interventions, ETA, alert RED"; closing cue "see you when you get here" triggers at once offline |

Pass criteria: the transcript shows all 5 bursts in order; no blank dashboard at any point;
the final report matches the fixed pipeline's CS014 result; "How old? Sorry." never
produces a stand-alone UNCLEAR report; every filled clinical field has a burst number;
pressing **New patient** at burst 3 splits the session and **Merge with previous** restores
it with one re-extraction.

## Limitations to state in the demo

- Input is recorded clips, not live radio. The call-sets were grouped offline and one
  grouping needed human review (the CS014 join).
- Offline, the picture lags the radio by about a minute per update and the local model
  runs one job at a time.
- Whisper mis-hears radio audio ("glow-by-oxygen", "texamethasone", "CMA-kepi"). The
  corrector fixes some; the model infers the rest and says so in assumptions.
- The local 4B model invents filler such as "NKDA". Attribution marks these as suspect;
  the benchmark will say how much a larger model or Groq removes.
- A patched channel puts the hospital side on another feed. The live app must listen to
  every feed of the same hospital to hear both sides, and overlap in time is normal.
- The boundary heuristic is conservative. Short gaps with no identifiers stay in the open
  session; the operator buttons exist for the cases it gets wrong.

## Order of work

1. Finish steps 1 to 5 of the current plan.
2. Score the post-fix corpus against the gold answers.
3. Start the replay branch on this design: sessions, trigger rules, one-job queue,
   updating marker, attribution, change line, operator buttons, replay with compressed
   gaps, CS014 acceptance scenario.
