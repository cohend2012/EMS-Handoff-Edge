# Corpus scripts

The pipeline that turned the MoCo radio archive into labelled call-sets, run in stage order.
Data lives outside the repo and is never committed: `~/calls` (m4a), `~/calls_wav`, `~/transcripts`,
`~/segments` (one wav + txt per burst), `~/callsets` (CS001.., one folder per call), `~/reports`
(frozen baseline), `~/labels` (csv/json bookkeeping, `run_log.txt` with every command).

| Script | Stage | Does |
|---|---|---|
| stage2_inventory.py | 2 | order clips by recording time, dedupe, convert to wav |
| stage3_transcribe.py | 3a | Groq Whisper transcription, resumable, local fallback |
| stage3_segment.py | 3b | burst boundaries from silences and tones; reviewed overrides in `segment_overrides.json` |
| stage4_callsets.py | 4 | group bursts into call-sets per feed; reviewed cross-feed joins in `callset_relations.json`; stable IDs |
| stage5_reports.py | 5 | run the app's pipeline over every call-set and every burst (Ollama, pinned model) |
| stage5_summary.py | 5 | summarise a Stage 5 log |
| check_hallucinations.py | 5 | report claims not found in the transcript, and transcript facts missing from the report |
| triage_candidates.py | 5 | triage the checker's rows for one folder into `review/triage_all.csv` and `review/review_sheet.csv` (likely_real + uncertain) |
| compare_runs.py | 5 | two report folders side by side: crashes, refusals, hallucination candidates, fields filled, confidence, per-set alert/sex/age diffs |
| run_samples.py | - | the four built-in sample cases on a pinned model, one report or one error file per case |
| stage6_label.py | 6 | one scenario label per call-set, low confidence to `~/scenarios/needs_review` |

Scoring against the gold answers for the four sample cases is `scripts/benchmark.py` with `tests/gold/`.

## Generated data and the air test

`spectral_survey.py`, `radioize.py`, `clone_voice.py`, `clone_voice_cloud.py`, `ngram_scores.py` and
`../air_check.py` work on generated data under `HANDOFF_DATASETS` (default `~/datasets`), never on the
corpus paths above. The layout and the rules are in `docs/DATA_LAYOUT.md`.
