# Where the data lives

Two trees, kept apart on purpose. Nothing in the repo contains either (see .gitignore).

## The real corpus and its runs (read-only inputs, the older tests)

| path | what | written by |
|---|---|---|
| `~/calls` | the raw scanner recordings (m4a) | hand-collected |
| `~/calls_wav`, `~/transcripts`, `~/segments` | stage 2-3 outputs: wav, Groq transcripts, one wav+txt per burst | scripts/corpus stage scripts |
| `~/callsets/CS001..` | one folder per call: `callset_info.json`, burst wavs and texts, combined transcript | stage4_callsets.py |
| `~/labels` | inventory, labels, `reference/<CS>/<seg>.ref.txt` and `.candidates.json`, `gold/` | stage scripts, draft_reference.py, people |
| `~/reports`, `~/reports_postfix`, `~/reports_replay` | report runs over the corpus (baseline frozen), replay runs | stage5_reports.py, replay_run.py |
| `~/recordings/<date>_<commit>/` | demo recordings with `<CS>.<provider>.summary.json` | demo_batch.py, record_all.py |
| `~/demo` | the collected demo set and INDEX.md | collect_demo.py |

The app's replay mode reads `~/callsets` (`HANDOFF_CALLSETS`). `score_callsets.py`, `benchmark.py`,
`compare_runs.py` and the demo batch read only these paths. Nothing below writes into them.

## Generated and derived data (`~/datasets`, the newer work)

Root is `HANDOFF_DATASETS` (default `~/datasets`); every script under scripts/ that makes or reads
generated data takes its default from it.

| path | what | written by |
|---|---|---|
| `voices/refs/` | 12 s single-speaker clips cut from the corpus, used as cloning references | by hand (ffmpeg); see SPECTRAL_2026-10-26.md |
| `voices/piper/` | Piper voice models | piper.download_voices |
| `noise/` | real channel-floor beds per feed (`feed4390_floor.wav`, 77 s; `feed4395_floor.wav`, 13 s) | the noise-bank cut (to be scripted) |
| `spectral/` | the FFT survey of the corpus: files.csv, spectra.csv, frames/, png/ | spectral_survey.py |
| `scripts_v0/` | the three practice scripts: text, spoken form, WAV per voice and variant (`<script>[_xtts_<ref>][_radio].wav`) | air_check.py --make-wavs, clone_voice*.py, radioize.py |
| `air_runs/` | over-the-air runs, one folder each, and INDEX.tsv | air_check.py |
| `synth_v1/` (planned) | the generated gold set: `scripts/`, `gold/`, `audio/<voice>/<snr>/`, `splits.json`, and `callsets/` | gen_scripts.py, render step |

### How the app uses generated data

- **Over the air:** `air_check.py` drives the running app through its public API exactly as a person
  would (start, speak, done). The app is not changed and does not know the data exists.
- **Replay:** generated sets are laid out as call-sets (`synth_v1/callsets/SY001/callset_info.json`
  plus burst wavs and texts, the same schema as `~/callsets`), so the app's replay mode plays them
  with `HANDOFF_CALLSETS=~/datasets/synth_v1/callsets`. Same code path, different folder; the real
  call-sets are never mixed in.
- **Scoring:** synthetic gold answers use the `tests/gold` field format so `benchmark.py` scores
  them unchanged; the synthetic reference transcripts are the script texts.

### Rules

1. Scripts that generate data write only under `HANDOFF_DATASETS`. Scripts that score the real corpus
   never read from it.
2. A new generation is a new versioned folder (`synth_v1`, `synth_v2`), never edited in place after a
   run has been scored against it.
3. The real call-sets are the final test and stay out of every training split. Clones made from
   CS008 and CS014 mean those two call-sets are reported separately when a clone-trained model is
   evaluated.
4. Behaviour changes in the app (for example the ETA transcript check of commit c6308c1, or the
   scorer's unit normalisation) apply to old and new data alike; when an old number is re-derived
   after such a change, say which commit produced it.
