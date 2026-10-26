# Incident sessions: building the patient picture burst by burst

Status: design note, agreed 2026-10-24. Round two of the pipeline fixes is merged and tagged
v2.1.1-round2; the replay branch is the next piece of work.

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

1. Done: post-baseline fixes (v2.1.0-postfix) and round two (v2.1.1-round2); the corpus is
   scored for baseline, post-fix, round 2 and Groq (`~/reports_round2/ROUND2.md`).
2. Next: the replay branch on this design: sessions, trigger rules, one-job queue, updating
   marker, attribution, change line, operator buttons, replay with compressed gaps.

   One acceptance test: CS014 plays through the app. The captions and the transcript grow
   burst by burst, the report builds and then updates on the same screen, and the run is
   recorded to a video. The table above is the script; the pass criteria are unchanged.

## Recording the acceptance run (decided 2026-10-25)

Audio and video on one clock, on the Jetson. A PulseAudio null sink is created for replay.
The replay driver owns burst start timing: at the moment it starts feeding a burst's WAV
into the file source it also plays the same WAV into the sink, so the audio on the
recording and the captions on screen come from the same samples. One ffmpeg process
captures the app window at 1080p 15 fps (page zoomed 1.5x, so the 720p layout at twice the density;
1080p15 ultrafast measured at full rate with no dropped frames on the Jetson) and the sink's monitor together, with wallclock
timestamps on the grab so dropped frames do not shorten the video. No post-mux.

Sync check: at each burst start the driver plays a 50 ms click into the sink and flashes a
marker in the banner for two frames. `scripts/check_sync.py` locates the clicks in the
audio track and the flashes in the video track and prints the offset per burst; a
recording passes only if every burst is within 150 ms.

Output: `~/recordings/<date>_<commit>/<callset>.mp4` with the run's summary JSON and the
sync report beside it; copied to a laptop with scp. Speaker playback stays off by default;
with a speaker attached the sink can be the hardware device instead. The remote option (a
laptop browser on the LAN recording the page) remains the alternative for a laptop-side
video.

## Transcript sanity checks (step 7 of the replay branch)

- Words per second: fewer than one word per two seconds on a burst over ten seconds means
  Whisper most likely dropped the burst (the post-fix S038_01 two-word transcript). Retry
  once, then fall back to local Whisper, and record what happened in `_meta`.
- Silence: when a recording's level never rose above the noise floor, treat it as no
  speech, skip Whisper, and show "no speech detected" instead of a phantom transcript. A
  silent mic on 2026-10-25 came back from Whisper as "Thank you." and produced an UNCLEAR
  report; the level meter had read zero throughout.

## Parked (not before the replay branch passes its test)

- Recording on a laptop instead of the Jetson: a browser pointed at the Jetson over the
  LAN with capture there (the on-Jetson method above is the decided default).
- Second Jetson: the deployment doc for a second unit.
- Browser memory: how long a session's captions and transcript can grow in the page before
  the tab needs trimming.
- Provider caps: Groq's 200k tokens per day stops a 73-report corpus pass at about 16
  reports in a day with gpt-oss-120b; a corpus pass needs a plan for the cap or a paid tier.
- Alert level stability: the same CS008 audio on Groq gpt-oss-120b came out MEDICAL YELLOW,
  MEDICAL RED, MEDICAL YELLOW, STEMI RED and MEDICAL YELLOW in five replays on 2026-10-25
  (transcripts near-identical, 183 to 186 words); the type moves as well as the level. Three of
  the reports are saved side by side in `~/reports_replay/CS008_alert_level/`
  for comparison. Candidates: temperature already 0, so the variance is the provider's; a
  level rule from stated priority words ("priority 2") or a second vote.
- Round 3 pipeline candidates (details in `~/reports_round2/ROUND2.md`): the ETA guard (null
  unless minutes are stated), the alert-word rule tightened to positive statements with a
  complaint or mechanism present (S032_01 and the "stroke? No." fragment), and the Whisper
  sanity check (retry, then local fallback, when words per second is implausible; the
  post-fix S038_01 two-word transcript).
