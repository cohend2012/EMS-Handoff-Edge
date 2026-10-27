#!/usr/bin/env python3
"""
eval_set.py — transcribe a rendered generated set and score it against the scripts: WER and term error by SNR.

    python scripts/corpus/eval_set.py --split test --model medium                 # local faster-whisper medium, hint on
    python scripts/corpus/eval_set.py --split test --model medium --no-hint
    python scripts/corpus/eval_set.py --split test --model small --corrector      # the app's corrector after the model
    python scripts/corpus/eval_set.py --split test --provider groq               # Groq whisper (key from the secrets file)
    python scripts/corpus/eval_set.py --summary <set>/evals/medium_hint          # re-render the table from rows.tsv

Per clip (script x voice x SNR), against the script text as the reference, with score_callsets' normalisation
(number words as digits, spoken units as written):
    wer           word error rate (Levenshtein over normalised words)
    term_total / term_missing   occurrences of terms.txt entries in the reference, and how many the transcript lacks
                  (the medical-term error rate MTER = missing / total)
    halluc        transcript content words that occur nowhere in the reference, as a share of transcript words
    loop          1 if the transcript repeats the same 4-gram three or more times in a row (a decoder loop)
    tail_missing  share of the reference's last 20 % of words missing from the transcript (a dropped tail)
    ngram_f       mean of 1- to 4-gram F against the reference (scripts/corpus/ngram_scores.py)
Writes <set>/evals/<label>/rows.tsv (resumable: done clips are skipped), transcripts/<clip>.txt and summary.md
with one row per SNR. The label defaults to <provider or model>[_hint|_nohint][_corr].
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from ngram_scores import scores as ngram_scores  # noqa: E402
from score_callsets import normalize, wer  # noqa: E402

STOP = set("""a an the and or but of to in on at for with from by as is are was were be been being am it its it's this that these those
there here he she they we you i his her their our your him them us me my mine yours theirs so if then than too very just about over
out up down off not no yes do does did done have has had having will would can could should may might must shall into onto per
how what when where which who whom why all any some each every both few more most other such only own same again further once
s t ll re ve d m""".split())


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def term_counts(norm_words: List[str], terms: List[List[str]]) -> Counter:
    """How often each term (a normalised word list) occurs in a word sequence."""
    c: Counter = Counter()
    n = len(norm_words)
    for t in terms:
        k = len(t)
        if not k:
            continue
        hits = sum(1 for i in range(n - k + 1) if norm_words[i:i + k] == t)
        if hits:
            c[" ".join(t)] = hits
    return c


def metrics(ref: str, hyp: str, terms: List[List[str]]) -> Dict[str, object]:
    r, h = normalize(ref), normalize(hyp)
    rate, edits, nref = wer(ref, hyp)
    rt, ht = term_counts(r, terms), term_counts(h, terms)
    total = sum(rt.values())
    missing = sum(max(0, rt[t] - ht.get(t, 0)) for t in rt)
    rset = set(r)
    content = [w for w in h if w not in STOP and len(w) > 2 and not w.isdigit()]
    halluc = sum(1 for w in content if w not in rset) / len(h) if h else 0.0
    loop = 0
    if len(h) >= 12:
        grams = [tuple(h[i:i + 4]) for i in range(len(h) - 3)]
        run = 1
        for i in range(4, len(grams)):
            if grams[i] == grams[i - 4]:
                run += 1
                if run >= 3:
                    loop = 1; break
            else:
                run = 1
    tail = r[int(0.8 * len(r)):]
    hset = Counter(h)
    tail_missing = sum(1 for w in tail if hset[w] <= 0) / len(tail) if tail else 0.0
    ng = ngram_scores(ref, hyp)
    return {"ref_words": nref, "hyp_words": len(h), "wer": rate, "edits": edits, "term_total": total, "term_missing": missing,
            "halluc": round(halluc, 3), "loop": loop, "tail_missing": round(tail_missing, 3), "ngram_f": round(ng.get("avg_f", ng.get("f_avg", 0.0)), 3)}


def make_transcriber(provider: str, model: str, hint: bool):
    from ems_handoff.config import load_env_files, load_settings
    from ems_handoff.transcribe import GroqTranscriber, LocalTranscriber
    load_env_files()                     # the Groq key lives in ~/.config/handoff-ai/secrets.env
    settings = load_settings()
    if provider == "groq":
        t = GroqTranscriber(settings)
        if model.startswith("whisper-"):
            t.model = model            # e.g. whisper-large-v3 beside the default turbo
    else:
        t = LocalTranscriber(settings, model_size=model)
    if not hint:
        t.hint = ""
    return t


def summarize(rows: List[Dict[str, str]]) -> str:
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["snr"], []).append(r)
    order = sorted(by, key=lambda s: (0, 0) if s == "snrinf" else (1, -float(s[3:])))
    out = ["| SNR | clips | WER | MTER | n-gram F | halluc | loops | dropped tail | audio s | transcribe s |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in order:
        rs = by[s]
        f = lambda k: np.array([float(r[k]) for r in rs])
        w = f("ref_words"); e = f("edits")
        tt, tm = f("term_total").sum(), f("term_missing").sum()
        out.append(f"| {s[3:]} | {len(rs)} | {100 * e.sum() / w.sum():.1f} % | {100 * tm / tt if tt else 0:.1f} % | {f('ngram_f').mean():.3f} | "
                   f"{100 * f('halluc').mean():.1f} % | {int(f('loop').sum())} | {100 * f('tail_missing').mean():.1f} % | "
                   f"{f('seconds').sum():.0f} | {f('transcribe_s').sum():.0f} |")
    return "\n".join(out)


def write_summary(out: Path, label: str) -> None:
    lines = (out / "rows.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
    fams = sorted({r["family"] for r in rows})
    txt = [f"# {label}: {len(rows)} clips\n", "WER = edits / reference words over the clips; MTER = missing term occurrences / term occurrences.\n",
           summarize(rows), ""]
    for fam in fams:
        fr = [r for r in rows if r["family"] == fam]
        txt += [f"\n## {fam} ({len({r['id'] for r in fr})} scripts)\n", summarize(fr)]
    (out / "summary.md").write_text("\n".join(txt) + "\n")


def rescore(set_dir: Path, out: Path) -> int:
    """Rows and summary rebuilt from transcripts/ with the current metrics; timings are kept."""
    terms = [normalize(t) for t in (set_dir / "terms.txt").read_text().splitlines() if t.strip()]
    lines = (out / "rows.tsv").read_text().splitlines()
    cols = lines[0].split("\t")
    rows = [dict(zip(cols, l.split("\t"))) for l in lines[1:] if l.strip()]
    for r in rows:
        ref = (set_dir / "scripts" / f"{r['id']}.txt").read_text()
        hyp = (out / "transcripts" / f"{r['id']}_{r['voice']}_{r['snr']}.txt").read_text().strip()
        r.update({k: str(v) for k, v in metrics(ref, hyp, terms).items()})
    (out / "rows.tsv").write_text("\n".join(["\t".join(cols)] + ["\t".join(r[c] for c in cols) for r in rows]) + "\n")
    write_summary(out, out.name)
    print(f"[OK] {len(rows)} rows rescored; {out / 'summary.md'}\n")
    print((out / "summary.md").read_text().split("\n## ")[0])
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", type=Path, default=paths.datasets() / "synth_v1")
    ap.add_argument("--voice", default="piper")
    ap.add_argument("--snr", nargs="+", default=["inf", "20", "15", "10", "5"])
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--provider", choices=["local", "groq"], default="local")
    ap.add_argument("--model", default="medium", help="faster-whisper size for --provider local; a whisper-* name picks the Groq model")
    ap.add_argument("--no-hint", action="store_true")
    ap.add_argument("--fallback", action="store_true", help="local only: temperature fallback 0/0.2/0.4 with the no-speech gate off (LOCAL_WHISPER_FALLBACK)")
    ap.add_argument("--redecode", action="store_true", help="local only: decode a long wordless stretch again on its own (LOCAL_WHISPER_REDECODE_GAPS)")
    ap.add_argument("--hotwords", action="store_true", help="local only: pass terms.txt as faster-whisper hotwords (prepended to the prompt)")
    ap.add_argument("--corrector", action="store_true", help="apply the app's transcript corrector to the output")
    ap.add_argument("--label")
    ap.add_argument("--min-interval", type=float, default=None, help="seconds between transcriber calls; default 3.2 for groq (20 requests a minute), 0 locally")
    ap.add_argument("--from", dest="from_label", help="do not transcribe: reuse the transcripts of this eval label (for --corrector)")
    ap.add_argument("--summary", type=Path, help="only rebuild summary.md for this eval folder")
    ap.add_argument("--rescore", type=Path, help="recompute every row of this eval folder from its saved transcripts (after a scorer change)")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.set, args.rescore)
    if args.min_interval is None:
        args.min_interval = 3.2 if args.provider == "groq" else 0.0
    if args.summary:
        write_summary(args.summary, args.summary.name); print((args.summary / "summary.md").read_text()); return 0

    label = args.label or ((args.from_label or (("groq" + ("_" + args.model[8:] if args.model.startswith("whisper-") else "") if args.provider == "groq" else args.model) + ("_nohint" if args.no_hint else "_hint") + ("_hot" if args.hotwords else "") + ("_fb" if args.fallback else "") + ("_rd" if args.redecode else ""))) + ("_corr" if args.corrector else ""))
    out = args.set / "evals" / label
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.tsv"
    cols = ["id", "family", "voice", "snr", "ref_words", "hyp_words", "wer", "edits", "term_total", "term_missing", "halluc", "loop", "tail_missing", "ngram_f", "seconds", "transcribe_s"]
    done = set()
    if rows_path.exists():
        for l in rows_path.read_text().splitlines()[1:]:
            p = l.split("\t")
            if len(p) >= 4:
                done.add((p[0], p[2], p[3]))
    else:
        rows_path.write_text("\t".join(cols) + "\n")

    ids = sorted(p.stem for p in (args.set / "scripts").glob("SY*.txt") if not p.name.endswith(".spoken.txt"))
    if args.split:
        ids = [i for i in ids if i in set(json.loads((args.set / "splits.json").read_text())[args.split])]
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    terms = [normalize(t) for t in (args.set / "terms.txt").read_text().splitlines() if t.strip()]
    src_dir = (args.set / "evals" / args.from_label / "transcripts") if args.from_label else None
    transcriber = None if src_dir else make_transcriber(args.provider, args.model, not args.no_hint)
    if transcriber and args.fallback and hasattr(transcriber, "fallback"):
        transcriber.fallback = True
    if transcriber and args.redecode and hasattr(transcriber, "redecode"):
        transcriber.redecode = True
    if transcriber and args.hotwords:
        transcriber.hotwords = ", ".join(t.strip() for t in (args.set / "terms.txt").read_text().splitlines() if t.strip())
    if transcriber:
        transcriber.warm_up()
    if args.corrector:
        from ems_handoff.transcript_corrector import correct_transcript
    print(f"[{label}] {transcriber.describe() if transcriber else 'transcripts of ' + args.from_label}; hint {'off' if args.no_hint else 'on'}; "
          f"hotwords {'on' if args.hotwords else 'off'}; fallback {'on' if args.fallback else 'off'}; redecode {'on' if args.redecode else 'off'}; corrector {'on' if args.corrector else 'off'}; {len(ids)} scripts x {args.snr}; {len(done)} clips already done", flush=True)
    n_new = 0
    last_call = 0.0
    for sid in ids:
        ref = (args.set / "scripts" / f"{sid}.txt").read_text()
        family = json.loads((args.set / "gold" / f"{sid}.json").read_text()).get("family", "")
        for s in args.snr:
            tag = "snrinf" if s == "inf" else f"snr{int(float(s))}"
            if (sid, args.voice, tag) in done:
                continue
            wav = args.set / "audio" / args.voice / tag / f"{sid}.wav"
            if not wav.exists():
                print(f"[skip] {wav} not rendered yet", flush=True); continue
            x = load_wav(wav)
            t0 = time.time()
            if src_dir:
                src = src_dir / f"{sid}_{args.voice}_{tag}.txt"
                if not src.exists():
                    print(f"[skip] {src} not there yet", flush=True); continue
                hyp = src.read_text().strip()
            else:
                wait = args.min_interval - (time.time() - last_call)
                if wait > 0:
                    time.sleep(wait)
                hyp = None
                for attempt in (1, 2):
                    last_call = time.time()
                    try:
                        hyp = transcriber.transcribe(x); break
                    except Exception as e:  # noqa: BLE001
                        print(f"[error] {sid} {tag} (try {attempt}): {e}", flush=True); time.sleep(6)
                if hyp is None:
                    continue                 # no row: a rerun picks the clip up
            dt_ = time.time() - t0
            if args.corrector and hyp:
                hyp = correct_transcript(hyp).text
            (out / "transcripts" / f"{sid}_{args.voice}_{tag}.txt").write_text(hyp + "\n")
            m = metrics(ref, hyp, terms)
            row = {"id": sid, "family": family, "voice": args.voice, "snr": tag, **m, "seconds": round(len(x) / 16000, 1), "transcribe_s": round(dt_, 1)}
            with rows_path.open("a") as f:
                f.write("\t".join(str(row[c]) for c in cols) + "\n")
            n_new += 1
            print(f"{sid} {tag:<7} wer {m['wer']:.3f} terms {m['term_missing']}/{m['term_total']} halluc {m['halluc']:.2f} loop {m['loop']} tail {m['tail_missing']:.2f} ({dt_:.0f} s)", flush=True)
    write_summary(out, label)
    print(f"\n[OK] {n_new} new clips; {out / 'summary.md'}\n")
    print((out / "summary.md").read_text().split("\n## ")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
