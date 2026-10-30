# Plan: the patient-state graph, what published algorithms let us observe, and how it reaches the doctor

Written 2026-10-29, late. Companion to `LIVE_STATE_PLAN.md` (the live machine and the asking rule) and
`STATEFUL_PIPELINE.md` (slot state and the delta writer). This one is the larger picture: the state as a
graph, what serial observations let us say about the person in the ambulance, which published prehospital
rules we can apply with a citation, and how the state is communicated. The first implementation landed
today in `ems_handoff/state.py` (commit on 10-29 night) and is exposed as `state` in every session snapshot.

## 1. The state as a graph

Nodes, in four layers, each with a time and a source:

```
 evidence layer      burst 1 ──── burst 2 ──── burst 3 ──── (live audio, later: monitor, camera)
                        │            │  │          │
 observation layer   age 34      BP 118/76   BP 84/48     "see you on arrival"
                     male        HR 96       HR 128       ETA 5
                     MVC         GCS 15      GCS 15
                        │            │           │
 derived layer       shock index 0.81 ───────> 1.52    (crossed 1.0 between burst 2 and 3)
                     MAP 90 ────────────────> 60       (crossed 65)
                     NEWS2 partial 1 ───────> 8 (high)
                     field-triage red: SBP < 90, RR 30
                        │
 decision layer      alert TRAUMA RED (stated)   missing: nothing   ask: none
                     flags: shock index, hypotension, low MAP, rSIG
```

Edges carry provenance: every observation points at the burst (and later the transcript span and the
whisper log-probability over it); every derived value points at the observations it used; every flag points
at the derived value and the published rule. The report on screen is a rendering of this graph, not the
writer's last output. Two properties follow that the flat report does not have:

- **A value can change and the change is itself an observation.** Two blood pressures are not a corrected
  field; they are a series with a range and a threshold crossing, each with a source.
- **Every claim can be traced.** A doctor reading "shock index 1.52 at burst 3" can open burst 3 and hear it.

The probabilistic reading: each observation node holds a belief (the writer's value, the corrector's edit,
whisper's confidence over the span, agreement between bursts). Today the belief is binary (stated or not,
conflicting or not); the plan is to carry the whisper span confidence and a two-burst agreement as the
first two evidence weights, and to let a flag's text say "SBP 84 (burst 3, low confidence)". Nothing is
inferred that was not said; what the graph adds is arithmetic, thresholds and bookkeeping.

## 2. What serial readings let us observe (and what they do not)

The literature is clear and slightly disappointing. Over the short prehospital window, vital-sign *trends*
are not diagnostic by themselves: in air-transported trauma patients haemorrhage cases showed no directional
difference in SBP, HR or shock index over 21 minutes, only wider ranges (Chen, Reisner, Gribok, Reifman,
J Spec Oper Med 2010, [PubMed 21140982](https://pubmed.ncbi.nlm.nih.gov/21140982/)); a 601-patient cohort
found prehospital values change a lot on the way in and that GCS <= 12 and SBP <= 90 predict major trauma
while heart rate alone does not ([Paramedic Practice](https://www.paramedicpractice.com/content/features/changes-in-vital-signs-of-trauma-victims-from-prehospital-to-hospital-settings));
single time-point SBP and shock index remained the best mortality predictors in a 2024 TQIP analysis.
So what two readings give us, and what the state reports, is:

1. **A range.** Wider swings are the one serial signal with support. The state reports min and max.
2. **Threshold crossings.** SBP falling through 90, shock index rising through 1.0, MAP through 65, GCS
   through 13: a crossing between two bursts is reported with both sources. These are the published static
   thresholds applied twice, not a trend claim.
3. **Averages.** Chen et al. suggest averaging; with three or more readings the state can report the mean.
4. **Not** a slope, a prediction, or "deteriorating". The screen says what crossed and when.

For the doctor, the useful sentence is "SBP 118 at 14:02, 84 at 14:07, shock index 0.81 to 1.52", with the
bursts behind it, rather than any of our own interpretation.

## 3. The published rules the state applies, each with its source and its caveat

| rule | what the state computes | threshold and source | caveat |
|---|---|---|---|
| Shock index | HR / SBP per reading | >= 1.0: higher massive-transfusion and mortality risk in 144,951 TQIP trauma patients ([Jehan 2019](https://experts.arizona.edu/en/publications/pre-hospital-shock-index-correlates-with-transfusion-resource-uti/)); 0.9 separates transfusion need in a 184-patient cohort ([Pariente Juste 2020](https://revistaemergencias.org/wp-content/uploads/2023/09/Emergencias-2021_33_1_29-34-34_eng.pdf)); prehospital SI improves massive-transfusion prediction ([Injury 2015](https://pubmed.ncbi.nlm.nih.gov/25555919/)) | observational; a flag, not a diagnosis |
| rSIG | SBP / HR x GCS | < 12.7: 70 % sensitive, 70 % specific for 7-day mortality in severe trauma (Spanish prehospital cohort 2024, [Cambridge](https://www.cambridge.org/core/product/E4577A3E0D165C5B8E5AEDC0EA8A3266)) | needs all three values; modest accuracy |
| Field triage red vitals | per reading | RR < 10 or > 29, room-air SpO2 < 90, unable to follow commands, adult SBP threshold ([National Guideline 2021](https://www.facs.org/media/1sgarjm5/trauma-algorithm-vfinal-revise-word-doc.docx), [ACS summary](https://facs.org/for-medical-professionals/news-publications/news-and-articles/press-releases/2022/trauma-triage-field-guidelines), [Montana adaptation](https://prod-dphhs.mt.gov/assets/publichealth/EMSTS/Trauma/FieldTriageGuidelines2021.pdf)) | the report carries total GCS, not motor; GCS <= 13 is the stand-in; the adult SBP line must be confirmed against the ACS text (state adaptations use 90) |
| MAP | (SBP + 2 DBP) / 3 | < 65 mmHg: perfusion floor (Surviving Sepsis Campaign target) | needs a diastolic |
| NEWS2 | partial score from RR, SpO2, SBP, HR, consciousness, temperature | RCP 2017 bands; prehospital 48-hour mortality AUC 0.77 to 0.94 in 1,054 and 4,943-patient Spanish cohorts ([PMC6633971](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6633971/), [CONICET record](https://ri.conicet.gov.ar/handle/11336/164697)); better with lactate ([JCM 2020](https://doaj.org/article/493c027de4ed42239c6f49edae997b93)) | the state says which of the six inputs are missing and calls the score partial; Spanish systems, generalisation untested |
| qSOFA | not computed as a screen | prehospital sensitivity 16 to 41 %, specificity 88 to 98 % ([Dorsett 2017](https://profiles.wustl.edu/en/publications/qsofa-has-poor-sensitivity-for-prehospital-identification-of-seve/), [Dunn 2017](https://etd.library.emory.edu/concern/etds/m039k567h)); reasonable in a 131,700-patient Alberta cohort ([Lane 2020](https://vivo.weill.cornell.edu/display/pubid32152051)) | too insensitive to be a flag; the sepsis checklist asks for the six inputs instead |
| Stroke scales | documentation check, not a score | LAMS and RACE accuracy 0.89 and 0.88 across seven scales, sensitivity 38 to 62 %, specificity 80 to 93 % (Nguyen, JAMA Neurol 2020, [summary](https://neurologylive.com/view/lams-and-race-scales-lead-predictive-accuracy-of-symptomatic-large-anterior-vessel-occlusion)); FAST-ED cut-off 3 or 4, sensitivity 0.72 to 0.77 ([meta-analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC11842085)) | the crew does the exam; the state checks that a scale or a deficit was stated |
| Stroke documentation | missing list | LKW or time of discovery, glucose, deficit, anticoagulants, prenotification (AHA GWTG prehospital elements, [Mission: Lifeline Stroke](https://www.heart.org:443/-/media/Files/Professional/Quality-Improvement/Get-With-the-Guidelines/Get-With-The-Guidelines-Stroke/Workshops-and-Webinars/ML-Stroke-Call-to-Action-May-2019--v3.pdf)); LKW documented in 19 % of Wisconsin EMS stroke runs in 2018 ([Wisconsin DHS](https://dhs.wisconsin.gov/publications/p01870g.pdf)); on-scene goal 15 minutes | LKW is the field the ask rule is most likely to fire on, and the one with the best evidence that it is missed |
| STEMI documentation | missing list | prehospital 12-lead, transmit a positive ECG within 10 minutes, first-medical-contact-to-device <= 90 minutes ([AHA EMS criteria](https://www.heart.org/-/media/files/professional/quality-improvement/mission-lifeline/ml-ems-recognition-criteria82620.pdf?la=en)); prehospital ECG use 45 to 71 % and FMC-to-device 93 to 84 minutes across 485 hospitals 2008 to 2012 ([ACEP Now](https://acepnow.com/article/aha-initiative-boosts-quality-of-care-for-stemi)) | the state asks for the finding and aspirin; the ECG time slot comes with the slot-state writer |

Every flag the state emits names its rule in the text the doctor sees, so the citation travels with the
observation. Where the search did not return the authoritative threshold (the adult SBP line in the 2021
field triage guideline), the code says so in its docstring and the value is marked for confirmation by the
clinical co-authors.

## 4. How the state reaches the doctor

Three audiences, three renderings of the same graph:

- **The receiving facility, live.** The report as today, plus a flag line under the vitals ("shock index
  1.52 at burst 3, crossed 1.0 since burst 2") and the missing list for the alert type ("Stroke: last known
  well not stated"). One line, cited, with the burst behind it. The ask rule (LIVE_STATE_PLAN.md section 4)
  decides when the missing list becomes a prompt.
- **The handoff.** A printed or sent summary ordered the way the receiving clinician reads it: alert and
  ETA, patient, the series of vitals as a small table with times, the flags with their rules, interventions
  with times, what is unknown. Nothing in it that was not in a burst.
- **The debrief and the training record** (the paper's first use). The whole graph with its timeline:
  when each fact was first stated, what changed, what was asked and whether it was answered, what the
  published rules would have flagged and when. This is the "OODA audit" the abstract promises, and with the
  graph it is a replay of the call with every node timed, not a transcript with a report at the end.

## 5. Beyond radio: the same graph with more evidence layers

The graph was designed for bursts, but a node is a timed observation with a source, so the other sources
the project plans attach without changing the model: a monitor feed gives readings every few seconds (the
series then has the density Chen et al. said averaging needs); a camera gives intervention times; the
crew's own voice in the cabin is a burst with a different feed label. Each source gets a belief weight, and
the same flags, checklists and asks apply. The rule that nothing is inferred that was not observed stays.

## 6. What is done, what is next, and the proof for each

| step | status | proof |
|---|---|---|
| state module: slots with provenance, readings, derived values, flags with rules, partial NEWS2, series with crossings, per-alert checklist, conflicts; in every session snapshot | done 10-29, 10 tests, suite 255 | replay the 22 calls through the headless path and count: flags per call, crossings, missing items per alert type, conflicts; review the list with the clinical co-authors (which flags are right, which thresholds to confirm) |
| state on screen: badge, slot board, flag line, missing line | next | the same replay, viewed; a short review with a medic |
| ask rule on the closing cue | next, after the screen | asks per call and precision against gold on the 22 calls |
| whisper span confidence and two-burst agreement as evidence weights | measured 10-30: faster-whisper's per-word probability detects a wrong word at AUROC 0.864 [0.844, 0.882] on held-out generated scripts (LYAPUNOV_MATH section 11); next: carry it into the slots | flags that carry "low confidence" against the reference transcripts |
| averages with three or more readings; the ECG-time and LKW-time slots | with the slot-state writer | the 60 synthetic reports plus the real bursts |
| monitor and camera feeds as evidence layers | later, hardware | |

## 7. What this means for the abstract

The abstract can say, with today's code and sources: the system keeps a timed, sourced state of the
patient across the call; it applies published prehospital rules (shock index, field-triage vital-sign
criteria, NEWS2, the AHA stroke and STEMI documentation elements) to what was said and shows each flag with
its rule and its source burst; and it lists, per alert type, the documentation the call still lacks. What
it cannot yet say is how often those flags and missing items were right on real calls: that number comes
from the replay in section 6, which is the next run.
