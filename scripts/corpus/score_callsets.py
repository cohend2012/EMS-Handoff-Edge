#!/usr/bin/env python3
"""Score the app on real call-sets: transcripts against the reference, reports against the gold.

    score_callsets.py CS008 CS014 CS001 [--summaries '~/recordings/*'] [--md FILE]

Transcripts: word error rate per burst for every candidate in ~/labels/reference/<id>/
<segment>.candidates.json (each transcriber), for the stage-3 transcript, and for the
transcript the app actually used in each replay recording (<folder>/<id>.summary.json, the
session transcript split at its burst headers). Reference: <segment>.ref.txt.

Reports: scripts/benchmark.py's buckets (hit, omission, wrong, abstained, hallucination) and
weighted score for the final report of each replay recording against ~/labels/gold/<id>.json,
plus the alert truth: the level the stated priority implies (1 RED, 2 YELLOW, 3 GREEN) against
the level the report gave, and the stated ETA against the report's.

Nothing is run here: the script only reads what earlier runs saved, so a score is always tied
to a commit folder.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import benchmark  # noqa: E402  (scripts/benchmark.py)

LEVEL_FOR_PRIORITY = {1: "RED", 2: "YELLOW", 3: "GREEN"}
_NUM = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14",
        "fifteen": "15", "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
        "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90",
        "hundred": "100"}


def normalize(text: str) -> List[str]:
    """Words for WER: lower case, no punctuation, hyphens split, % as percent, number words as digits."""
    t = text.lower().replace("%", " percent ").replace("-", " ").replace("/", " over ")
    t = re.sub(r"(\d)([a-z])", r"\1 \2", t)          # 324mg and 324 mg are the same words
    t = re.sub(r"\bmilligrams?\b", "mg", t)             # spoken units as written
    t = re.sub(r"\bmillilit(?:er|re)s?\b", "ml", t)
    t = re.sub(r"[^\w\s']", " ", t)
    words = [w.strip("'") for w in t.split()]
    return [_NUM.get(w, w) for w in words if w]


def wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    """(word error rate, edits, reference words) by Levenshtein distance over normalized words."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return (0.0 if not h else 1.0), len(h), 0
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i] + [0] * len(h)
        for j, hw in enumerate(h, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
        prev = cur
    return round(prev[-1] / len(r), 3), prev[-1], len(r)


def burst_texts_from_session(transcript: str) -> List[str]:
    """The text of each burst in an incident-session transcript, in header order."""
    parts = re.split(r"^\[Burst \d+ [^\]]*\][^\n]*$", transcript, flags=re.M)
    return [p.strip() for p in parts[1:]] if len(parts) > 1 else ([transcript.strip()] if transcript.strip() else [])


def load_callset(callsets_dir: Path, cid: str) -> Dict[str, Any]:
    return json.loads((callsets_dir / cid / "callset_info.json").read_text())


def transcript_scores(cid: str, info: Dict[str, Any], ref_dir: Path, callsets_dir: Path,
                      summaries: List[Path]) -> Dict[str, Dict[str, Tuple[float, int, int]]]:
    """{source: {segment: (wer, edits, ref_words)}} for every transcript source available."""
    out: Dict[str, Dict[str, Tuple[float, int, int]]] = {}
    segs = [b["segment_id"] for b in info["bursts"]]
    refs = {}
    for seg in segs:
        p = ref_dir / cid / f"{seg}.ref.txt"
        if p.exists():
            refs[seg] = p.read_text().strip()
    if not refs:
        return out
    for seg in segs:
        if seg not in refs:
            continue
        cand = ref_dir / cid / f"{seg}.candidates.json"
        if cand.exists():
            for name, text in json.loads(cand.read_text()).items():
                if name.startswith("_") or not isinstance(text, str) or text.startswith("<error"):
                    continue
                out.setdefault(name, {})[seg] = wer(refs[seg], text)
        s3 = callsets_dir / cid / f"{seg}.txt"
        if s3.exists():
            out.setdefault("stage3", {})[seg] = wer(refs[seg], s3.read_text())
    for summ in summaries:
        d = json.loads(summ.read_text())
        texts = burst_texts_from_session(((d.get("session") or {}).get("transcript")) or "")
        label = f"app {summ.parent.name.split('_')[-1]}" + (f" {summ.name.split('.')[1]}" if summ.name.count(".") > 2 else "")
        for seg, text in zip(segs, texts):
            if seg in refs:
                out.setdefault(label, {})[seg] = wer(refs[seg], text)
    return out


def alert_truth(gold: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
    stated = gold.get("stated") or {}
    pri = stated.get("priority")
    expect_level = LEVEL_FOR_PRIORITY.get(pri) if pri else None
    got_level = report.get("alert_level")
    eta_expect = stated.get("eta_minutes")
    eta_got = benchmark._first_int(report.get("eta_minutes"))
    return {"priority": pri, "expect_level": expect_level, "level": got_level,
            "level_ok": (got_level == expect_level) if expect_level else None,
            "eta_expect": eta_expect, "eta": eta_got, "eta_ok": (eta_got == eta_expect) if eta_expect is not None else None,
            "type": report.get("alert_type")}


def report_scores(cid: str, gold: Dict[str, Any], summaries: List[Path]) -> List[Dict[str, Any]]:
    rows = []
    for summ in summaries:
        d = json.loads(summ.read_text())
        rep = (d.get("session") or {}).get("report") or {}
        if not rep:
            rows.append({"run": summ.parent.name, "commit": d.get("commit"), "provider": d.get("provider"), "model": d.get("model"),
                         "error": "no final report in the summary"})
            continue
        s = benchmark.score(rep, gold)
        run = summ.parent.name + (f" {summ.name.split('.')[1]}" if summ.name.count(".") > 2 else "")
        s.update(run=run, commit=d.get("commit"), provider=d.get("provider"), model=d.get("model"), truth=alert_truth(gold, rep))
        rows.append(s)
    return rows


def fmt_pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{x:.0f}"


def render(cid: str, info: Dict[str, Any], tscores, rrows, gold_exists: bool) -> str:
    lines = [f"## {cid}: {info.get('units')} to {info.get('hospital')}, {len(info['bursts'])} burst(s), {info.get('total_words')} words"]
    segs = [b["segment_id"] for b in info["bursts"]]
    if tscores:
        lines.append("")
        lines.append("### Transcript word error rate (lower is better; reference in ~/labels/reference)")
        lines.append("")
        lines.append("| source | " + " | ".join(segs) + " | all |")
        lines.append("|---|" + "---|" * (len(segs) + 1))
        for name, per in sorted(tscores.items(), key=lambda kv: sum(v[1] for v in kv[1].values()) / max(1, sum(v[2] for v in kv[1].values()))):
            edits = sum(v[1] for v in per.values())
            words = sum(v[2] for v in per.values())
            cells = [f"{per[s][0] * 100:.0f}%" if s in per else "-" for s in segs]
            lines.append(f"| {name} | " + " | ".join(cells) + f" | {100 * edits / words:.0f}% ({edits}/{words}) |")
    else:
        lines.append("")
        lines.append("No reference transcript yet.")
    lines.append("")
    if not gold_exists:
        lines.append("No gold report yet.")
        return "\n".join(lines)
    lines.append("### Final report against the gold (benchmark buckets; weighted = (hit + abstained) weight share)")
    lines.append("")
    lines.append("| run | provider | model | weighted | hit | omission | wrong | halluc. | abstained | type | level (expect) | ETA (expect) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rrows:
        if r.get("error"):
            lines.append(f"| {r['run']} | {r.get('provider')} | {r.get('model')} | - | | | | | | {r['error']} | | |")
            continue
        t = r["truth"]
        lvl = f"{t['level']} ({t['expect_level']})" + ("" if t["level_ok"] in (None, True) else " **miss**")
        eta = f"{t['eta']} ({t['eta_expect']})" + ("" if t["eta_ok"] in (None, True) else " **miss**")
        lines.append(f"| {r['run']} | {r['provider']} | {r['model']} | {r['weighted']} | {r['hit']} | {r['omission']} | {r['wrong']} | "
                     f"{r['hallucination']} | {r['abstained']} | {t['type']} | {lvl} | {eta} |")
    bad = [(r["run"], p, b, d) for r in rrows if not r.get("error") for (p, k, b, d, w) in r["rows"] if b in ("wrong", "hallucination", "omission")]
    if bad:
        lines.append("")
        lines.append("Misses, by run:")
        for run, p, b, d in bad:
            lines.append(f"- {run}: {p} {b} ({d})")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("callsets", nargs="+")
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--reference-dir", type=Path, default=Path.home() / "labels" / "reference")
    ap.add_argument("--gold-dir", type=Path, default=Path.home() / "labels" / "gold")
    ap.add_argument("--summaries", default=f"{Path.home() / 'recordings' / '*'},{Path.home() / 'replays' / '*'}",
                    help="comma-separated globs of run folders (recordings and unrecorded replays)")
    ap.add_argument("--md", type=Path)
    args = ap.parse_args(argv)
    out = ["# Call-set scores", ""]
    for cid in args.callsets:
        info = load_callset(args.callsets_dir, cid)
        summaries = sorted(Path(p) for g in args.summaries.split(",") for p in glob.glob(f"{g.strip()}/{cid}*.summary.json"))
        ts = transcript_scores(cid, info, args.reference_dir, args.callsets_dir, summaries)
        gold_path = args.gold_dir / f"{cid}.json"
        rr = report_scores(cid, json.loads(gold_path.read_text()), summaries) if gold_path.exists() else []
        out.append(render(cid, info, ts, rr, gold_path.exists()))
        out.append("")
    text = "\n".join(out)
    print(text)
    if args.md:
        args.md.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
