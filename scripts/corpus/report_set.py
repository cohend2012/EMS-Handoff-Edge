#!/usr/bin/env python3
"""
report_set.py — run the app's report pipeline over a generated set's transcripts and score the reports against the gold.

    python scripts/corpus/report_set.py --split test --clean                  # the script texts themselves (the pasted-text path)
    python scripts/corpus/report_set.py --split test --from medium_hint       # the transcripts of an eval_set.py run, every SNR
    python scripts/corpus/report_set.py --split test --from medium_hint --snr 10 5 --model qwen3:4b-instruct

The pipeline is the app's ReportPipeline (corrector, lexicon context, extraction, grounding rules, ETA check) with
the Ollama model pinned through OLLAMA_MODEL; a report from another model is an error. One JSON per clip under
<set>/reports/<label>/<id>_<voice>_<snr>.json, then scripts/benchmark.py's buckets against <set>/gold/<id>.json and
score_callsets' alert truth (level from the stated priority, ETA). rows.tsv (resumable) and summary.md by SNR and by
family: weighted score, recall, precision, hallucinations per report, alert type accuracy, level and ETA agreement,
extraction seconds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark  # noqa: E402
import paths  # noqa: E402
from score_callsets import alert_truth  # noqa: E402

COLS = ["id", "family", "voice", "snr", "weighted", "recall", "precision", "hit", "omission", "wrong", "abstained", "hallucination",
        "alert_ok", "level_ok", "eta_ok", "alert_got", "extract_s", "wall_s"]


def summarize(rows: List[Dict[str, str]]) -> str:
    by: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by.setdefault(r["snr"], []).append(r)
    order = sorted(by, key=lambda s: (0, 0) if s in ("clean", "snrinf") else (1, -float(s[3:])) if s.startswith("snr") else (2, 0))
    out = ["| transcript | reports | weighted | recall | precision | halluc / report | alert type | level | ETA | extract s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for s in order:
        rs = by[s]
        f = lambda k: np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in rs])
        ok = lambda k: f"{100 * np.nanmean(f(k)):.0f} %" if not np.all(np.isnan(f(k))) else "-"
        out.append(f"| {s} | {len(rs)} | {np.nanmean(f('weighted')):.1f} | {np.nanmean(f('recall')):.1f} | {np.nanmean(f('precision')):.1f} | "
                   f"{f('hallucination').mean():.2f} | {ok('alert_ok')} | {ok('level_ok')} | {ok('eta_ok')} | {np.nanmean(f('extract_s')):.0f} |")
    return "\n".join(out)


def write_summary(out: Path, label: str) -> None:
    lines = (out / "rows.tsv").read_text().splitlines()
    head = lines[0].split("\t")
    rows = [dict(zip(head, l.split("\t"))) for l in lines[1:] if l.strip()]
    txt = [f"# {label}: {len(rows)} reports\n",
           "weighted = (hit + abstained) weight share (scripts/benchmark.py); level = alert_level agrees with the stated priority "
           "(1 RED, 2 YELLOW, 3 GREEN) where one was stated; ETA = eta_minutes agrees with the stated ETA where one was stated.\n",
           summarize(rows), ""]
    for fam in sorted({r["family"] for r in rows}):
        fr = [r for r in rows if r["family"] == fam]
        txt += [f"\n## {fam} ({len({r['id'] for r in fr})} scripts)\n", summarize(fr)]
    (out / "summary.md").write_text("\n".join(txt) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--set", type=Path, default=paths.datasets() / "synth_v1")
    ap.add_argument("--split", choices=["train", "val", "test"])
    ap.add_argument("--ids", nargs="*")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--clean", action="store_true", help="the script texts (voice 'text', snr 'clean')")
    src.add_argument("--from", dest="from_label", help="transcripts of this <set>/evals/<label>")
    ap.add_argument("--voice", default="piper")
    ap.add_argument("--snr", nargs="+", default=["inf", "20", "15", "10", "5"])
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="qwen3:4b-instruct")
    ap.add_argument("--label")
    ap.add_argument("--no-rules", action="store_true", help="grounding rules after the model off (HANDOFF_REPORT_RULES=0); label suffix _norules")
    ap.add_argument("--min-interval", type=float, default=0.0, help="seconds between report starts (cloud rate limits)")
    ap.add_argument("--summary", type=Path, help="only rebuild summary.md for this reports folder")
    ap.add_argument("--rescore", type=Path, help="rebuild rows.tsv and summary.md of this reports folder from its saved reports against the current gold")
    args = ap.parse_args()
    if args.rescore:
        out = args.rescore
        rows = []
        for rp in sorted(out.glob("SY*.json")):
            report = json.loads(rp.read_text()); run_ = report.get("_run", {})
            sid = run_.get("script") or rp.name[:5]
            gold = json.loads((args.set / "gold" / f"{sid}.json").read_text())
            s = benchmark.score(report, gold); truth = alert_truth(gold, report)
            alert_row = next((b for p_, k, b, d, w in s["rows"] if p_ == "alert_type"), "")
            rows.append({"id": sid, "family": gold.get("family", ""), "voice": run_.get("voice", ""), "snr": run_.get("snr", ""), "weighted": s["weighted"],
                         "recall": s["recall"], "precision": s["precision"], "hit": s["hit"], "omission": s["omission"], "wrong": s["wrong"],
                         "abstained": s["abstained"], "hallucination": s["hallucination"], "alert_ok": int(alert_row == "hit"),
                         "level_ok": "" if truth.get("level_ok") is None else int(bool(truth["level_ok"])),
                         "eta_ok": "" if truth.get("eta_ok") is None else int(bool(truth["eta_ok"])),
                         "alert_got": report.get("alert_type"), "extract_s": s.get("extract_s"), "wall_s": run_.get("wall_s")})
        (out / "rows.tsv").write_text("\t".join(COLS) + "\n" + "".join("\t".join("" if r[c] is None else str(r[c]) for c in COLS) + "\n" for r in rows))
        write_summary(out, out.name); print(f"[OK] rescored {len(rows)} reports\n"); print((out / "summary.md").read_text().split("\n## ")[0]); return 0
    if not (args.clean or args.from_label or args.summary):
        ap.error("one of --clean, --from, --summary or --rescore is required")
    if args.summary:
        write_summary(args.summary, args.summary.name); print((args.summary / "summary.md").read_text().split("\n## ")[0]); return 0

    os.environ["OLLAMA_MODEL"] = args.model
    os.environ.setdefault("HANDOFF_PROVIDER", args.provider)
    from ems_handoff.config import load_env_files, load_settings
    load_env_files()
    import dataclasses
    from ems_handoff.llm import LLMClient
    from ems_handoff.pipeline import ReportPipeline
    from ems_handoff.lexicon_rag import LexiconStore
    from ems_handoff.textbook_rag import TextbookRAG
    settings = dataclasses.replace(load_settings(), provider=args.provider)
    # the same engines server.default_engines builds, without importing the audio stack (a text-only tool
    # runs on a box with no sound libraries)
    lexicon = LexiconStore() if settings.use_lexicon else None
    textbook = TextbookRAG(store_dir=str(settings.textbook_dir)) if settings.use_textbook_rag else None
    pipeline = ReportPipeline(LLMClient(settings), stop_phrase=settings.stop_phrase,
                              lexicon_store=lexicon, textbook_rag=textbook, rules=not args.no_rules)

    label = args.label or ("clean" if args.clean else args.from_label) + "__" + args.model.replace(":", "-") + ("_norules" if args.no_rules else "")
    out = args.set / "reports" / label
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.tsv"
    if not rows_path.exists():
        rows_path.write_text("\t".join(COLS) + "\n")
    done = {tuple(l.split("\t")[:4][i] for i in (0, 2, 3)) for l in rows_path.read_text().splitlines()[1:] if l.strip()}

    ids = sorted(p.stem for p in (args.set / "scripts").glob("SY*.txt") if not p.name.endswith(".spoken.txt"))
    if args.split:
        ids = [i for i in ids if i in set(json.loads((args.set / "splits.json").read_text())[args.split])]
    if args.ids:
        ids = [i for i in ids if i in set(args.ids)]
    variants = [("text", "clean")] if args.clean else [(args.voice, "snrinf" if s == "inf" else f"snr{int(float(s))}") for s in args.snr]
    print(f"[{label}] provider {args.provider}, model {args.model}; {len(ids)} scripts x {[v[1] for v in variants]}; {len(done)} done", flush=True)
    n_new = 0
    for sid in ids:
        gold = json.loads((args.set / "gold" / f"{sid}.json").read_text())
        for voice, tag in variants:
            if (sid, voice, tag) in done:
                continue
            if args.clean:
                text = (args.set / "scripts" / f"{sid}.txt").read_text()
            else:
                src_path = args.set / "evals" / args.from_label / "transcripts" / f"{sid}_{voice}_{tag}.txt"
                if not src_path.exists():
                    print(f"[skip] {src_path} missing", flush=True); continue
                text = src_path.read_text().strip()
            if args.min_interval and n_new:
                time.sleep(max(0.0, args.min_interval - (time.time() - t0)))
            t0 = time.time()
            try:
                report = pipeline.run(text, source="synth")
            except Exception as e:  # noqa: BLE001
                print(f"[error] {sid} {tag}: {type(e).__name__}: {e}", flush=True); continue
            wall = time.time() - t0
            got_model = report.get("_meta", {}).get("model")
            if args.provider == "ollama" and got_model != args.model:
                print(f"[error] {sid} {tag}: model {got_model!r} is not the pinned {args.model!r}", flush=True); return 1
            report["_run"] = {"set": str(args.set), "script": sid, "voice": voice, "snr": tag, "source": "clean" if args.clean else args.from_label,
                              "pinned_model": args.model, "wall_s": round(wall, 1)}
            (out / f"{sid}_{voice}_{tag}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
            s = benchmark.score(report, gold)
            truth = alert_truth(gold, report)
            alert_row = next((b for p, k, b, d, w in s["rows"] if p == "alert_type"), "")
            row = {"id": sid, "family": gold.get("family", ""), "voice": voice, "snr": tag, "weighted": s["weighted"], "recall": s["recall"],
                   "precision": s["precision"], "hit": s["hit"], "omission": s["omission"], "wrong": s["wrong"], "abstained": s["abstained"],
                   "hallucination": s["hallucination"], "alert_ok": int(alert_row == "hit"),
                   "level_ok": "" if truth.get("level_ok") is None else int(bool(truth["level_ok"])),
                   "eta_ok": "" if truth.get("eta_ok") is None else int(bool(truth["eta_ok"])),
                   "alert_got": report.get("alert_type"), "extract_s": s.get("extract_s"), "wall_s": round(wall, 1)}
            with rows_path.open("a") as f:
                f.write("\t".join("" if row[c] is None else str(row[c]) for c in COLS) + "\n")
            n_new += 1
            print(f"{sid} {tag:<7} weighted {s['weighted']:>5} rec {s['recall']} prec {s['precision']} hal {s['hallucination']} "
                  f"alert {report.get('alert_type')} ({'ok' if alert_row == 'hit' else alert_row}) level_ok {row['level_ok']} eta_ok {row['eta_ok']} ({wall:.0f} s)", flush=True)
    write_summary(out, label)
    print(f"\n[OK] {n_new} new reports; {out / 'summary.md'}\n")
    print((out / "summary.md").read_text().split("\n## ")[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
