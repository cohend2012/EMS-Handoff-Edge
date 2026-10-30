# State replay over the real corpus, 2026-10-29

`ems_handoff/state.py` (commit 8e8d06b and the UNCLEAR fix after it) run over the round-2 per-burst reports
of the 22 real call sets (`~/reports_round2/<segment>.json`, one report per burst, local qwen3:4b-instruct
with the grounding rules, 10-24), in burst order per call set. No model ran; this is the state module's
arithmetic and checklists over reports that already existed. Script: session scratchpad `state_replay.py`.

**Caveat on the input.** Round 2 wrote each burst's report from that burst alone, not from the session
transcript the live app uses, so a later burst that is a two-word reply produces a near-empty report. The
state handles that (empty values never overwrite, UNCLEAR is not a type change), but the conflict and
missing counts below are what the state sees over per-burst reports, and the live app, which re-extracts
from the whole session, will show fewer. The replay over session-cumulative reports is the next run.

## Totals

    calls 22 bursts 51 readings 21 flags 8 crossings 0 missing 31 conflicts 10
    alerts {'STROKE': 7, 'TRAUMA': 5, 'MEDICAL': 1, 'CARDIAC': 4, 'ENDOCRINE': 4, 'PEDIATRIC': 1}
    flags {'shock_index': 4, 'altered_mental_status': 1, 'rsig': 1, 'respiratory_rate': 1, 'glucose': 1}

## What the replay says

- **Flags fire on real calls and name their rule.** Eight flags over 22 calls: shock index at or above
  0.9 or 1.0 on four calls (CS002, CS004, CS007, CS020), and on CS004 (a trauma) the three trauma flags
  together (shock index, altered mental status, rSIG below 12.7). CS012 flagged a respiratory rate outside
  10 to 29, CS021 a glucose outside 60 to 400 on an endocrine call. Whether each is clinically right is the
  co-authors' review; each flag shows the reading and the burst.
- **No crossings.** Twenty-one readings over 22 calls: the radio report states vitals once. Only CS021
  (seven bursts) carried two readings. Serial-vitals observations need the monitor feed or the cabin
  audio; on the radio alone they will be rare, which is itself a finding for the paper.
- **The stroke documentation gap is real and matches the literature.** Over the merged slots seven calls
  carry a STROKE type at some burst; six of them never state anticoagulant use, four state no deficit or
  scale, four no blood glucose, three no last known well. Wisconsin EMS documented last known well in 19 %
  of stroke runs in 2018; the ask rule has its first target.
- **Trauma calls did not state a respiratory rate** in five of five. Respiratory rate is a field-triage
  red criterion (< 10 or > 29), so the missing line is worth showing.
- **NEWS2 is always partial on the radio** (temperature is never stated, consciousness often), so the
  screen must show it as "partial, missing temperature" or not at all. Scores of 7 and 8 appeared on
  CS004, CS012 and CS014.
- **Conflicts over per-burst reports are mostly ETA updates** (CS005 15 to 12, CS006 15 to 5, CS013 8 to
  5, CS014 10 to 15, CS021 15 to 12), which are legitimate changes the state should show as a series rather
  than a conflict; the alert-type conflicts (STROKE then TRAUMA on CS006, PEDIATRIC then ENDOCRINE on
  CS013) are per-burst artefacts to re-check on cumulative reports.
- **The checklist runs over the merged slots** (the latest non-empty value of every field across bursts), so a
  two-word reply burst cannot blank it; calls whose every burst was UNCLEAR get the common floor only (age, sex,
  complaint, ETA): the state does not guess a type.

## Per call

missing {'STROKE: anticoagulant use (yes or no)': 6, 'TRAUMA: respiratory rate': 5, 'STROKE: blood glucose': 4, 'STROKE: deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati)': 4, 'STROKE: age': 3, 'STROKE: last known well or onset time': 3, 'STROKE: sex': 2, 'MEDICAL: age': 1, 'TRAUMA: SpO2': 1, 'TRAUMA: ETA': 1, 'STROKE: ETA': 1}
| call set | bursts with a report | alert | readings | flags | NEWS2 | crossings | missing | conflicts |
|---|---|---|---|---|---|---|---|---|
| CS001 | 3 | STROKE | 1 | - | 0 (partial) | 0 | age, blood glucose, anticoagulant use (yes or no) | - |
| CS002 | 2 | STROKE | 1 | shock_index | 2 (partial) | 0 | last known well or onset time, blood glucose, deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati), anticoagulant use (yes or no) | - |
| CS003 | 3 | STROKE | 0 | - | - | 0 | age, sex, last known well or onset time, blood glucose, deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati), anticoagulant use (yes or no) | - |
| CS004 | 2 | TRAUMA | 1 | shock_index, altered_mental_status, rsig | 7 (partial) | 0 | respiratory rate | - |
| CS005 | 4 | MEDICAL | 1 | - | 0 (partial) | 0 | age | eta_minutes: 15 (burst 1) then 12 (burst 3); alert_type: TRAUMA (burst 1) then MEDICAL (burst 3) |
| CS006 | 6 | TRAUMA | 1 | - | 0 (partial) | 0 | respiratory rate, SpO2 | eta_minutes: 15 (burst 1) then 5 (burst 6); alert_type: STROKE (burst 1) then TRAUMA (burst 6) |
| CS007 | 1 | CARDIAC | 1 | shock_index | 4 (partial) | 0 | - | - |
| CS008 | 1 | CARDIAC | 1 | - | 1 (partial) | 0 | - | - |
| CS009 | 1 | TRAUMA | 1 | - | 3 (partial) | 0 | ETA, respiratory rate | - |
| CS010 | 2 | ENDOCRINE | 1 | - | 4 (partial) | 0 | - | alert_type: STROKE (burst 1) then ENDOCRINE (burst 2) |
| CS011 | 1 | TRAUMA | 1 | - | 0 (partial) | 0 | respiratory rate | - |
| CS012 | 1 | ENDOCRINE | 1 | respiratory_rate | 7 (partial) | 0 | - | - |
| CS013 | 2 | ENDOCRINE | 1 | - | 6 (partial) | 0 | - | eta_minutes: 8 (burst 1) then 5 (burst 2); alert_type: PEDIATRIC (burst 1) then ENDOCRINE (burst 2) |
| CS014 | 5 | PEDIATRIC | 1 | - | 8 (partial) | 0 | - | eta_minutes: 10 (burst 3) then 15 (burst 4) |
| CS016 | 2 | TRAUMA | 1 | - | 0 (partial) | 0 | respiratory rate | - |
| CS017 | 1 | CARDIAC | 1 | - | 5 (partial) | 0 | - | - |
| CS018 | 1 | STROKE | 0 | - | - | 0 | age, sex, ETA, last known well or onset time, blood glucose, deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati), anticoagulant use (yes or no) | - |
| CS019 | 1 | STROKE | 1 | - | 0 (partial) | 0 | deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati), anticoagulant use (yes or no) | - |
| CS020 | 1 | CARDIAC | 1 | shock_index | 3 (partial) | 0 | - | - |
| CS021 | 7 | ENDOCRINE | 2 | glucose | 0 (partial) | 0 | - | eta_minutes: 15 (burst 2) then 12 (burst 6); alert_type: ENDOCRINE (burst 2) then ENDOCRINE (burst 7) |
| CS022 | 1 | STROKE | 1 | - | 0 (partial) | 0 | - | - |
| CS023 | 3 | STROKE | 1 | - | 4 (partial) | 0 | anticoagulant use (yes or no) | - |
