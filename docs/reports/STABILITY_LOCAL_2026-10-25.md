# Call-set scores

## CS008: ['731'] to ['shady grove'], 1 burst(s), 188 words

### Transcript word error rate (lower is better; reference in ~/labels/reference)

| source | S025_01 | all |
|---|---|---|
| app 45a20c4 run3 | 11% | 11% (22/196) |
| app 45a20c4 run4 | 11% | 11% (22/196) |
| app 45a20c4 run5 | 11% | 11% (22/196) |
| groq_turbo_radio | 12% | 12% (23/196) |
| app 45a20c4 run1 | 12% | 12% (23/196) |
| groq_turbo | 13% | 13% (26/196) |
| app 45a20c4 run2 | 13% | 13% (26/196) |
| groq_large | 15% | 15% (29/196) |
| stage3 | 15% | 15% (29/196) |
| local_medium | 18% | 18% (36/196) |
| vosk_small | 66% | 66% (130/196) |

### Final report against the gold (benchmark buckets; weighted = (hit + abstained) weight share)

| run | provider | model | weighted | hit | omission | wrong | halluc. | abstained | type | level (expect) | ETA (expect) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25_45a20c4 run1 | ollama | qwen3:4b-instruct | 94.0 | 18 | 0 | 1 | 0 | 8 | CARDIAC | YELLOW (YELLOW) | 5 (5) |
| 2026-10-25_45a20c4 run2 | ollama | qwen3:4b-instruct | 94.0 | 18 | 0 | 1 | 0 | 8 | CARDIAC | YELLOW (YELLOW) | 5 (5) |
| 2026-10-25_45a20c4 run3 | ollama | qwen3:4b-instruct | 94.0 | 18 | 0 | 1 | 0 | 8 | CARDIAC | YELLOW (YELLOW) | 5 (5) |
| 2026-10-25_45a20c4 run4 | ollama | qwen3:4b-instruct | 94.0 | 18 | 0 | 1 | 0 | 8 | CARDIAC | YELLOW (YELLOW) | 5 (5) |
| 2026-10-25_45a20c4 run5 | ollama | qwen3:4b-instruct | 94.0 | 18 | 0 | 1 | 0 | 8 | CARDIAC | YELLOW (YELLOW) | 5 (5) |

Misses, by run:
- 2026-10-25_45a20c4 run1: alert_type wrong (CARDIAC)
- 2026-10-25_45a20c4 run2: alert_type wrong (CARDIAC)
- 2026-10-25_45a20c4 run3: alert_type wrong (CARDIAC)
- 2026-10-25_45a20c4 run4: alert_type wrong (CARDIAC)
- 2026-10-25_45a20c4 run5: alert_type wrong (CARDIAC)

## CS014: ['729'] to ['shady grove'], 5 burst(s), 234 words

### Transcript word error rate (lower is better; reference in ~/labels/reference)

| source | S054_01 | S055_01 | S013_01 | S014_01 | S015_01 | all |
|---|---|---|---|---|---|---|
| local_medium | 25% | 33% | 0% | 11% | 13% | 15% (39/252) |
| groq_large | 21% | 33% | 22% | 67% | 16% | 19% (48/252) |
| groq_turbo_radio | 27% | 0% | 33% | 22% | 21% | 22% (56/252) |
| stage3 | 42% | 0% | 56% | 11% | 17% | 23% (58/252) |
| groq_turbo | 52% | 0% | 56% | 11% | 17% | 25% (63/252) |
| app 45a20c4 run1 | 64% | 0% | 56% | 11% | 23% | 32% (80/252) |
| app 45a20c4 run2 | 64% | 0% | 56% | 11% | 23% | 32% (80/252) |
| app 45a20c4 run3 | 64% | 0% | 56% | 11% | 23% | 32% (80/252) |
| app 45a20c4 run4 | 64% | 0% | 56% | 11% | 23% | 32% (80/252) |
| app 45a20c4 run5 | 64% | 0% | 56% | 11% | 23% | 32% (80/252) |
| vosk_small | 71% | 100% | 78% | 89% | 72% | 73% (183/252) |

### Final report against the gold (benchmark buckets; weighted = (hit + abstained) weight share)

| run | provider | model | weighted | hit | omission | wrong | halluc. | abstained | type | level (expect) | ETA (expect) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25_45a20c4 run1 | ollama | qwen3:4b-instruct | 92.0 | 14 | 0 | 1 | 1 | 9 | PEDIATRIC | YELLOW (YELLOW) | 10 (10) |
| 2026-10-25_45a20c4 run2 | ollama | qwen3:4b-instruct | 92.0 | 14 | 0 | 1 | 1 | 9 | PEDIATRIC | YELLOW (YELLOW) | 10 (10) |
| 2026-10-25_45a20c4 run3 | ollama | qwen3:4b-instruct | 92.0 | 14 | 0 | 1 | 1 | 9 | PEDIATRIC | YELLOW (YELLOW) | 10 (10) |
| 2026-10-25_45a20c4 run4 | ollama | qwen3:4b-instruct | 92.0 | 14 | 0 | 1 | 1 | 9 | PEDIATRIC | YELLOW (YELLOW) | 10 (10) |
| 2026-10-25_45a20c4 run5 | ollama | qwen3:4b-instruct | 92.0 | 14 | 0 | 1 | 1 | 9 | PEDIATRIC | YELLOW (YELLOW) | 10 (10) |

Misses, by run:
- 2026-10-25_45a20c4 run1: assessment_findings.level_of_consciousness wrong ({"airway": "patent", "breathing": "respiratory distress with)
- 2026-10-25_45a20c4 run1: history.medications hallucination (["dexamethasone", "racemic epinephrine (cemic epi)"])
- 2026-10-25_45a20c4 run2: assessment_findings.level_of_consciousness wrong ({"airway": "compromised (excessive saliva and drool)", "brea)
- 2026-10-25_45a20c4 run2: history.medications hallucination (["dexamethasone", "racemic epinephrine (cemic epi)"])
- 2026-10-25_45a20c4 run3: assessment_findings.level_of_consciousness wrong ({"airway": "patent", "breathing": "respiratory distress, sho)
- 2026-10-25_45a20c4 run3: history.medications hallucination (["dexamethasone", "racemic epinephrine (cemic epi)"])
- 2026-10-25_45a20c4 run4: assessment_findings.level_of_consciousness wrong ({"airway": "compromised (excessive saliva and drool)", "brea)
- 2026-10-25_45a20c4 run4: history.medications hallucination (["dexamethasone", "racemic epinephrine (cemic epi)"])
- 2026-10-25_45a20c4 run5: assessment_findings.level_of_consciousness wrong ({"airway": "compromised (excessive saliva and drool)", "brea)
- 2026-10-25_45a20c4 run5: history.medications hallucination (["dexamethasone", "racemic epinephrine (cemic epi)"])

## CS001: not stated to not stated, 3 burst(s), 132 words

### Transcript word error rate (lower is better; reference in ~/labels/reference)

| source | S044_01 | S045_01 | S046_01 | all |
|---|---|---|---|---|
| groq_turbo_radio | 16% | 4% | 13% | 14% (23/163) |
| app 45a20c4 run1 | 21% | 4% | 13% | 17% (28/163) |
| app 45a20c4 run2 | 21% | 4% | 13% | 17% (28/163) |
| app 45a20c4 run3 | 21% | 4% | 13% | 17% (28/163) |
| app 45a20c4 run4 | 21% | 4% | 13% | 17% (28/163) |
| app 45a20c4 run5 | 21% | 4% | 13% | 17% (28/163) |
| groq_large | 18% | 15% | 20% | 18% (29/163) |
| local_medium | 22% | 30% | 7% | 22% (36/163) |
| groq_turbo | 38% | 15% | 13% | 32% (52/163) |
| stage3 | 39% | 22% | 20% | 34% (56/163) |
| vosk_small | 62% | 59% | 60% | 61% (100/163) |

### Final report against the gold (benchmark buckets; weighted = (hit + abstained) weight share)

| run | provider | model | weighted | hit | omission | wrong | halluc. | abstained | type | level (expect) | ETA (expect) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-25_45a20c4 run1 | ollama | qwen3:4b-instruct | 91.8 | 15 | 1 | 0 | 1 | 7 | STROKE | RED (RED) | 10 (10) |
| 2026-10-25_45a20c4 run2 | ollama | qwen3:4b-instruct | 91.8 | 15 | 1 | 0 | 1 | 7 | STROKE | RED (RED) | 10 (10) |
| 2026-10-25_45a20c4 run3 | ollama | qwen3:4b-instruct | 91.8 | 15 | 1 | 0 | 1 | 7 | STROKE | RED (RED) | 10 (10) |
| 2026-10-25_45a20c4 run4 | ollama | qwen3:4b-instruct | 91.8 | 15 | 1 | 0 | 1 | 7 | STROKE | RED (RED) | 10 (10) |
| 2026-10-25_45a20c4 run5 | ollama | qwen3:4b-instruct | 91.8 | 15 | 1 | 0 | 1 | 7 | STROKE | RED (RED) | 10 (10) |

Misses, by run:
- 2026-10-25_45a20c4 run1: vital_signs.blood_glucose omission (null)
- 2026-10-25_45a20c4 run1: ems_interventions hallucination ([{"details": "supplemental oxygen delivered at unspecified f)
- 2026-10-25_45a20c4 run2: vital_signs.blood_glucose omission (null)
- 2026-10-25_45a20c4 run2: ems_interventions hallucination ([{"details": "gauge not specified", "intervention": "establi)
- 2026-10-25_45a20c4 run3: vital_signs.blood_glucose omission (null)
- 2026-10-25_45a20c4 run3: ems_interventions hallucination ([{"details": "line placed (gauge not specified)", "intervent)
- 2026-10-25_45a20c4 run4: vital_signs.blood_glucose omission (null)
- 2026-10-25_45a20c4 run4: ems_interventions hallucination ([{"details": "supplemental oxygen delivered at unspecified f)
- 2026-10-25_45a20c4 run5: vital_signs.blood_glucose omission (null)
- 2026-10-25_45a20c4 run5: ems_interventions hallucination ([{"details": "gauge not specified", "intervention": "establi)
