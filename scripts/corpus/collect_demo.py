#!/usr/bin/env python3
"""Gather every call-set's cloud and local recordings side by side for the demo.

    collect_demo.py [--out ~/demo] [--recordings ~/recordings]

For each call-set and provider the newest recording with a sync pass and no fallback is
linked (hard links, no extra disk) into ~/demo/<id>/<provider>.mp4, <provider>.share.mp4,
<provider>.summary.json, <provider>.sync.json, and <provider>.report.json (the final report
alone). ~/demo/INDEX.md lists, per call-set, the type, level and ETA from each provider and
whether they agree, the transcript word counts, and (where ~/labels/gold has it) the benchmark
score of each. The call-set's own transcript and info are linked too.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import benchmark
except Exception:  # noqa: BLE001
    benchmark = None

PROVIDERS = ("groq", "ollama")


def newest_good(recordings: Path, cid: str, provider: str) -> Optional[Path]:
    best = None
    for summ in recordings.glob(f"*/{cid}.*summary.json"):  # <cid>.<provider>.summary.json, or <cid>.summary.json before 2026-10-26
        d = json.loads(summ.read_text())
        if d.get("provider") != provider or d.get("fallback"):
            continue
        base = summ.name[: -len(".summary.json")]
        sync = summ.with_name(base + ".sync.json")
        if not sync.exists() or not json.loads(sync.read_text()).get("pass"):
            continue
        mp4 = summ.with_name(base + ".mp4")
        if not mp4.exists():
            continue
        if best is None or summ.stat().st_mtime > best.stat().st_mtime:
            best = summ
    return best


def link(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path.home() / "demo")
    ap.add_argument("--recordings", type=Path, default=Path.home() / "recordings")
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--gold-dir", type=Path, default=Path.home() / "labels" / "gold")
    args = ap.parse_args(argv)
    ids = sorted(p.name for p in args.callsets_dir.iterdir() if p.is_dir() and p.name.startswith("CS"))
    rows = []
    for cid in ids:
        folder = args.out / cid
        folder.mkdir(parents=True, exist_ok=True)
        for name in ("callset_info.json", "combined_transcript.txt"):
            src = args.callsets_dir / cid / name
            if src.exists():
                link(src, folder / name)
        info = json.loads((args.callsets_dir / cid / "callset_info.json").read_text())
        gold = json.loads((args.gold_dir / f"{cid}.json").read_text()) if (args.gold_dir / f"{cid}.json").exists() else None
        cells: Dict[str, Dict[str, object]] = {}
        for prov in PROVIDERS:
            summ = newest_good(args.recordings, cid, prov)
            if summ is None:
                cells[prov] = {}
                for stale in folder.glob(f"{prov}.*"):
                    stale.unlink()
                continue
            d = json.loads(summ.read_text())
            base = summ.name[: -len(".summary.json")]
            for ext in (".mp4", ".share.mp4", ".summary.json", ".sync.json"):
                src = summ.with_name(base + ext)
                if src.exists():
                    link(src, folder / f"{prov}{ext}")
            report = (d.get("session") or {}).get("report") or {}
            (folder / f"{prov}.report.json").write_text(json.dumps(report, indent=1))
            words = sum(t.get("words", 0) for t in d.get("transcriptions", []))
            score = benchmark.score(report, gold)["weighted"] if (gold and benchmark and report) else None
            cells[prov] = {"type": report.get("alert_type"), "level": report.get("alert_level"), "eta": report.get("eta_minutes"),
                           "words": words, "commit": d.get("commit"), "folder": summ.parent.name, "score": score,
                           "sync": json.loads(summ.with_name(base + ".sync.json").read_text()).get("max_abs_offset_ms")}
        g, o = cells.get("groq") or {}, cells.get("ollama") or {}
        agree = "-" if not (g and o) else ("yes" if (g["type"], g["level"], g["eta"]) == (o["type"], o["level"], o["eta"]) else "NO")
        rows.append((cid, info, g, o, agree))

    def cell(c: Dict[str, object]) -> str:
        if not c:
            return "not yet"
        s = f"{c['type']} {c['level']} ETA {c['eta']}"
        if c.get("score") is not None:
            s += f" · {c['score']}"
        return s

    lines = ["# Demo recordings: cloud (Groq gpt-oss-120b) and local (qwen3:4b-instruct) side by side", "",
             "Per call-set folder: `groq.mp4`, `ollama.mp4` (and `.share.mp4` copies under 30 MB), `*.summary.json` (the whole run: "
             "timings, transcriptions, session, report), `*.sync.json`, `*.report.json`, the call-set's info and transcript.", "",
             "| call-set | bursts | words | cloud (type level ETA · score) | local (type level ETA · score) | agree | cloud sync | local sync |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, info, g, o, agree in rows:
        lines.append(f"| {cid} | {info.get('num_bursts')} | {info.get('total_words')} | {cell(g)} | {cell(o)} | {agree} | "
                     f"{g.get('sync', '-') if g else '-'} ms | {o.get('sync', '-') if o else '-'} ms |")
    done_g = sum(1 for r in rows if r[2]); done_o = sum(1 for r in rows if r[3])
    lines += ["", f"Recorded: cloud {done_g}/{len(rows)}, local {done_o}/{len(rows)}. Score = benchmark weighted score against "
                  f"~/labels/gold where a gold report exists ({sum(1 for r in rows if (args.gold_dir / f'{r[0]}.json').exists())} call-sets)."]
    (args.out / "INDEX.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
