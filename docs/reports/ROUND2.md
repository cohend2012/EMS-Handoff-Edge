# Round 2 — HANDOFF.AI pipeline over the MoCo call-set corpus, 2026-10-24

| | |
|---|---|
| local run | ~/reports_round2, fix/post-baseline d4ad7bb, Ollama qwen3:4b-instruct pinned, 19:0x -> 20:00, 73/73 reports, 0 failures; bursts from the Stage 3 transcripts (--text) |
| Groq run | ~/reports_groq_round2, same commit, openai/gpt-oss-120b pinned; 16/73 reports, then the 200k tokens-per-day cap (used 199,342): 57 targets remain, re-run the same command tomorrow |
| samples | ~/reports_round2/samples (local); Groq samples blocked by the same cap |
| review sheet | ~/reports_round2/review/review_sheet.csv, 50 rows (11 likely_real, 39 uncertain); post-fix 78, baseline 105 |
| comparisons | COMPARE_postfix_vs_round2.md here; ~/reports_groq_round2/COMPARE_groq_vs_groq_round2.md |

## Commits in this round (fix/post-baseline, each with tests; suite 124 passed)

- 0e765b2 Accept a null alert level for any alert type (the 8 Groq failures)
- 05b093c No invented allergies or code status: null unless spoken, the transcript decides
- e865b5f UNCLEAR only when no complaint, mechanism, or alert word was stated
- d4ad7bb Wait out an Ollama model-server restart instead of failing the report
- bad27b4 triage_candidates.py writes the review sheet; e5a3f79 compare_runs rate-limit rows
- merge gate before the round: cd1b791 on handoff-app, tag v2.1.0-postfix

## Local 4B: post-fix -> round 2 (73 reports each, same transcripts)

| | post-fix | round 2 |
|---|---|---|
| unsupported claims | 98 | 32 |
| invented allergies / code status | 50 / 23 | 0 / 0 |
| UNCLEAR reports | 20 | 16 |
| combined reports at HIGH confidence | 2 | 7 |
| clinical fields filled, mean per report | 10.9 | 10.3 |
| pipeline crashes | 4 (Ollama restarts) | 0 (4 reports waited out a restart: CS017, S005, S034, S055) |
| review sheet rows | 78 | 50 |

Item 2, the five sets with the most NKDA rows (allergies / code status 'no' rows): CS021 6/3 -> 0/0, CS006 5/2 -> 0/0,
CS001 3/1 -> 0/0, CS003 3/1 -> 0/0, CS005 3/3 -> 0/0. Groq was 0/0 before and after. Every round-2 report has
allergies null and code_status null: the corpus never states either.

Item 3, S038_01, S002_01 and the 11 fragments (post-fix -> round 2 -> Groq):
S038_01 "trauma ... priority 2" UNCLEAR -> TRAUMA (Groq: UNCLEAR). S002_01 (refusal discussion) stays UNCLEAR on
all three. S009_01 "CVA, brain bleed" STROKE on all three. Nine fragments stay UNCLEAR. Two changed: S030_01 "Did you
just use your stroke? No." -> STROKE by the alert-word rule (the word is there; the sense is a question), and S032_01
"No copies, we'll see you in arrival ... I'm 14 clear" -> TRAUMA from the model itself under the looser rule, which is
a new invention. Also changed: CS013 and S054_01 to PEDIATRIC (22-month-old, 3-month-old; Groq agrees on CS013).

New pattern the checker shows: eta_minutes claimed with no ETA in the transcript rose 9 -> 16 (values 5, 12, 15).
The next grounding guard is ETA: null unless a number stands near "minutes", "ETA" or "out".

## Groq: first pass -> round 2 (16 shared reports)

unsupported claims 4 -> 3; fields filled 14.9 -> 14.4 per report; CS002 MEDICAL -> STROKE; CS016 gained the age.
Nothing on the Groq side needed the allergy or code-status guard (0 before and after).

## Item 5: the sepsis sample

| field (weight) | baseline | post-fix | round 2 |
|---|---|---|---|
| alert_type (3) | wrong: ENDOCRINE | wrong: ENDOCRINE | wrong: ENDOCRINE |
| history.code_status (3) | wrong: "full treatment" | wrong: "full treatment (per POLST)" | hit: DNR kept |
| history.medications (3) | omission: Eliquis -> "Elquis" | omission | omission |
| weighted | 79.1 | 79.1 | 86.0 |

The DNR came back once the lexicon stopped reading "full treatment" as full code. The two left: the ENDOCRINE label
(glucose 186 and the ENDOCRINE guidance line pull the 4B model; Groq says SEPSIS) and the model's own misspelling
"Elquis", which the gold list (eliquis/apixaban) cannot match; a report-side spelling pass over medication names
would fix it. Four samples, weighted: quick 83.3, stemi 96.0 (a heart_rate field returned as the whole vitals object),
trauma 100.0, sepsis 86.0; mean 91.3 (baseline 83.5, post-fix 88.6).

## Item 4: Ollama resilience

The systemd unit already has Restart=always, RestartSec=3 (no change needed). The client now waits 2/3/5/10/15/20/35 s
(90 s) on connection refused/reset or a 5xx naming a stopped runner, reports "Model server restarting" as a stage, the
UI state carries detail "model server restarting", and _meta.model_server records reason, seconds waited and retries.
Four round-2 reports used it; the batch lost nothing.

Resident while a report runs on the Jetson (measured during the post-fix batch, 7.5 GB total, 4 GB swap):
- llama-server (Ollama's runner) 5.6 GB RSS for qwen3:4b-instruct Q4_K_M: ~2.5 GB weights, KV cache for the 4096-token
  context (~0.6 GB), compute buffers; unified memory so GPU allocations count. The context is already capped at 4096 by
  Ollama's default; the prompt is ~1.6k tokens in, ~1k out, so 3072 would fit most reports but not the 300-word bursts.
- the app process ~105 MB; Vosk captioner + phrase spotter ~40 MB each only in a live session; local Whisper
  (large-v3-turbo int8, ~1.5 GB) is loaded only when Groq transcription is unavailable, and the transcriber cache keeps
  it for the process lifetime: that is the one big thing to unload after the final transcript when it was used.
- GNOME shell 185 MB + Xorg 80 MB (headless would free ~0.3 GB), dockerd + containerd ~55 MB (not used by the app),
  the ollama daemon 45 MB, this Claude Code session ~330 MB.
- Options not yet taken: OLLAMA_KV_CACHE_TYPE=q8_0 (halves the KV cache), OLLAMA_KEEP_ALIVE short so the model
  unloads between reports (costs a reload), stopping docker and GNOME on a demo box.

## Round 3 candidates (not started; v2.1.1-round2 is the cut)

1. **ETA guard.** `eta_minutes` is null unless a number stands next to "ETA", "minutes", "min" or "out" in the
   transcript; otherwise null, "ETA" in missing_info, a note in assumptions. Round 2 has 10 reports with an ETA
   and no such words (CS003 combined, S027_01, S029_01, S032_01, S035_01, S038_01, S041_01, S047_01, S052_01,
   S054_01), the loosest count 16. It is now the largest remaining class of unsupported claims.
2. **Alert-word rule tightened to positive statements.** `alert_from_words` fires today on any alert word. It should
   fire only when the word is stated positively and a chief complaint or mechanism is present in the same transcript.
   Two fragments show why: S030_01 "Did you just use your stroke? No." became STROKE from the rule, and S032_01
   "No copies, we'll see you in arrival. Okay, Max, I'm 14 clear." became TRAUMA from the model under the looser
   prompt wording. Both must stay UNCLEAR. S038_01 ("trauma? ... priority 2") must still be TRAUMA.
3. **Whisper sanity check.** The post-fix S038_01 run transcribed a 15.2 s burst with 44 words on disk as "... ..."
   (2 words, 0.13 words per second) and the report was UNCLEAR for a transcriber reason. Speech on the radio runs
   about 1.5 to 4 words per second. When words per second falls outside that band for a burst longer than a few
   seconds, retry Groq Whisper once, then fall back to the local model, and record which path produced the text
   in `_meta`. Round 2 and the Groq runs used `--text` from the stage-3 transcripts to remove this variable; the
   live app cannot.

## Tomorrow's first command

The Groq daily cap resets about 24 h after it was hit (19:07 on 2026-10-24). 57 of 73 round-2 Groq targets remain;
the runner skips the 16 that exist.

    cd ~/EMS-Handoff-app && git checkout fix/post-baseline && \
    python3 ~/labels/scripts/stage5_reports.py --all --provider groq --text \
        --out ~/reports_groq_round2 --log ~/labels/stage5_groq_round2_log.csv && \
    .venv/bin/python scripts/corpus/compare_runs.py ~/reports_groq ~/reports_groq_round2 --md ~/reports_groq_round2/COMPARE_groq_vs_groq_round2.md && \
    .venv/bin/python scripts/corpus/check_hallucinations.py ~/reports_groq_round2 && \
    .venv/bin/python scripts/corpus/triage_candidates.py ~/reports_groq_round2

Then add the "Groq round 2" column to the corpus table on the benchmark page
(https://claude.ai/artifact/JxUByGNMpEfSPUKQfiEcsz, section 4) and refresh the Groq row of the review sheet.
