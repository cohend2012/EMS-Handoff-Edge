---
name: real-bursts
description: Score transcriber stacks on the EMS-Handoff real radio bursts that have human reference transcripts (nine bursts, 612 words, call sets CS001, CS008, CS014). Use whenever the user asks how a model or stack does "on real audio", "on the real calls", or wants the real-burst table refreshed after a decode change; covers the stack caches, the GPU harness on cohendynamics, the scorer, and how the result is reported beside the generated-set rows.
---

# Real bursts with references

The generated set measures the decoder under controlled noise; the nine referenced bursts measure it on the
feed. Every decode change gets both, because the two have disagreed (medium fp16 looped on a real burst that
the synthetic set never showed).

## Data

| what | where |
|---|---|
| audio | `~/callsets/<CS>/<seg>.wav` (CS001 S044/S045/S046, CS008 S025, CS014 S013/S014/S015/S054/S055) |
| references | `~/labels/reference/<CS>/<seg>.ref.txt` (one annotator; 612 words) |
| stack caches | `~/labels/reference/<CS>/<seg>.stack.json` with keys `local_base`, `local_ours`, `cloud_base`, `cloud_ours`, `cloud_large_ours` → `{text, seconds, notes}` |

Caches are merged key-wise when they move between the Jetsons (`cohendynamics` holds half the call sets); a
whole-file rsync overwrote the other box's keys once.

## Running

CPU and cloud stacks: `scripts/corpus/real_stack.py --local` / cloud flags (writes the cache keys). GPU
stacks: on cohendynamics, `~/gpu_whisper/real_gpu.py` inside the container (one container per model,
`~/gpu_whisper/run_real_gpu.sh`), writing `/out/real_gpu_<model>.json`; sync `~/gpu_whisper/out/` back and
score with the session's `score_real_gpu.py` (scratchpad; re-create from `score_callsets.wer(ref, hyp)` and
`normalize` if missing; the corrector is applied as `TranscriptCorrector().correct(text).text`).

Score = edits / 612 reference words, pooled, plus per call set and the n-gram recall columns the matrix shows.

## Reporting

The real-burst table in `docs/reports/SYSTEM_VS_BASE_2026-10-28.md` section 2 lists every stack with the
corrector; current rows: cloud large-v3 + stack 16.2 %, cloud turbo + stack 21.7 %, GPU large-v3 + stack
21.7 % (7 s a burst), CPU medium stack 22.7 %, GPU medium + stack 25.8 %, bare medium 25.7 %, bare cloud turbo
27.6 %. Per set: CS008 is the easy one (10 to 18 %), CS014 (five short bursts, two voices) the hard one (18 to
33 %), and the per-set order is not stable, so report pooled and per set and do not rank on one set.

Read failures before summarising: a loop ("Priority 1, protocol." fourteen times) or a hallucinated tail on a
3.7 s burst is a decode bug to fix in `_redecode_gaps`, not a model verdict, and both have happened here.
Nine bursts from three calls is a pilot; say so when a difference is under several points.
