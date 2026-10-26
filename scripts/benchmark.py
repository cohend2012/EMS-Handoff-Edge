"""
benchmark.py — score saved handoff reports against the gold answers in tests/gold/.

    python scripts/benchmark.py output/reports/*.json
    python scripts/benchmark.py --per-field run.json        # show every field
    python scripts/benchmark.py --csv out.csv run1.json ... # one row per report

Each gold field is checked and lands in one bucket:

    hit           the fact was stated and the report has it right
    omission      the fact was stated and the report left it out (null/missing)
    wrong         the fact was stated and the report has a different value
    abstained     nothing was stated and the report correctly says null/unknown
    hallucination nothing was stated and the report invented something,
                  or a list has items that were never said

Weighted score = weighted (hit + abstained) / weighted total, so a hallucination
on a weight-3 field costs as much as a wrong alert type. Precision and recall
are over stated facts: recall = hit / stated, precision = hit / (hit + wrong +
hallucination). The case is inferred from the transcript saved in the report.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
GOLD_DIR = ROOT / "tests" / "gold"

UNKNOWN_WORDS = ("unknown", "not stated", "not reported", "not mentioned", "none specified",
                 "not provided", "unavailable", "not available", "n/a", "not documented")


def _get(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict):
            return None
        d = d.get(part)
    return d


def _text(v: Any) -> str:
    if v is None:
        return ""
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
    return s.lower().replace("\u202f", " ").replace("\u00a0", " ")   # models write "324\u202fmg": a narrow space is a space


def _is_unknown(v: Any) -> bool:
    if v is None or v == [] or v == {} or v == "":
        return True
    t = _text(v).strip()
    if t in ("null", "none", "[]"):
        return True
    if isinstance(v, str) and any(w in t for w in UNKNOWN_WORDS):
        return True
    if isinstance(v, list) and all(_is_unknown(x) for x in v):
        return True
    return False


def _first_int(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d+", _text(v))
    return int(m.group()) if m else None


def _items(v: Any) -> List[str]:
    """A list field as lowercase strings; dict items are flattened to their values."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v.lower()]
    out = []
    for x in v if isinstance(v, list) else [v]:
        out.append(" ".join(str(y) for y in x.values()).lower() if isinstance(x, dict) else str(x).lower())
    return out


def check(field: Dict[str, Any], report: Dict[str, Any]) -> Tuple[str, str]:
    """Return (bucket, detail) for one gold field."""
    kind = field["kind"]
    value = _get(report, field["path"])
    if field.get("search"):
        value = " | ".join(_text(_get(report, p)) for p in field["search"])
    expect = field.get("expect")

    if kind == "unknown":
        return ("abstained", "null") if _is_unknown(value) else ("hallucination", _text(value)[:60])

    if kind == "not_contains":
        hits = [p for p in expect if p in _text(value)]
        return ("hallucination", ", ".join(hits)) if hits else ("abstained", "clean")

    if _is_unknown(value):
        return "omission", "null"

    if kind == "enum":
        ok = [field["expect"], *field.get("accept", [])]
        got = str(value).strip()
        return ("hit", got) if got.lower() in (o.lower() for o in ok) else ("wrong", got)

    if kind == "int":
        got = _first_int(value)
        return ("hit", str(got)) if got == expect else ("wrong", _text(value)[:40])

    if kind == "contains":
        t = _text(value)
        return ("hit", _text(value)[:40]) if any(p in t for p in expect) else ("wrong", _text(value)[:60])

    if kind == "gcs":
        t = _text(value)
        nums = [int(n) for n in re.findall(r"\d+", t)]
        if not nums or nums[0] != expect:
            return "wrong", t[:40]
        spoken = field.get("breakdown")
        extra = [n for n in nums[1:] if n <= 6]
        if spoken is None and len(extra) >= 3:
            return "hallucination", f"breakdown invented: {t[:40]}"
        if spoken and len(extra) >= 3 and extra[:3] != spoken:
            return "wrong", f"breakdown {extra[:3]} vs {spoken}"
        return "hit", t[:40]

    if kind == "list":
        items = _items(value)
        missing = [alts[0] for alts in expect if not any(a in it for a in alts for it in items)]
        if missing:
            return "omission", "missing " + ", ".join(missing)
        if field["path"].startswith("history."):
            extra = [it for it in items if not any(a in it for alts in expect for a in alts)
                     and not _is_unknown(it)]
            if extra:
                return "hallucination", "extra " + ", ".join(e[:25] for e in extra)
        return "hit", f"{len(items)} items"

    raise ValueError(f"unknown kind {kind}")


def infer_case(report: Dict[str, Any]) -> Optional[str]:
    t = (_get(report, "_meta.transcript") or "").lower()
    for key, marker in (("quick", "medic 21"), ("stemi", "unit 41"), ("trauma", "medic 7"), ("sepsis", "squad 12")):
        if marker in t:
            return key
    return None


def score(report: Dict[str, Any], gold: Dict[str, Any]) -> Dict[str, Any]:
    rows, buckets, wsum, wgood = [], {}, 0, 0
    stated = hit = wrong = halluc = 0
    for f in gold["fields"]:
        bucket, detail = check(f, report)
        w = f.get("weight", 1)
        rows.append((f["path"], f["kind"], bucket, detail, w))
        buckets[bucket] = buckets.get(bucket, 0) + 1
        wsum += w
        if bucket in ("hit", "abstained"):
            wgood += w
        if bucket in ("hit", "omission", "wrong"):
            stated += 1
        hit += bucket == "hit"
        wrong += bucket == "wrong"
        halluc += bucket == "hallucination"
    meta = report.get("_meta", {})
    tm = meta.get("timings", {})
    return {
        "case": gold["case"],
        "provider": meta.get("provider"), "model": meta.get("model"),
        "weighted": round(100 * wgood / wsum, 1),
        "recall": round(100 * hit / stated, 1) if stated else None,
        "precision": round(100 * hit / (hit + wrong + halluc), 1) if (hit + wrong + halluc) else None,
        **{b: buckets.get(b, 0) for b in ("hit", "omission", "wrong", "abstained", "hallucination")},
        "extract_s": tm.get("extract_s"), "tok_s": tm.get("tok_s"),
        "rows": rows,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("reports", nargs="+")
    ap.add_argument("--per-field", action="store_true")
    ap.add_argument("--csv")
    args = ap.parse_args(argv)

    golds = {p.stem: json.loads(p.read_text()) for p in GOLD_DIR.glob("*.json")}
    results = []
    for path in args.reports:
        try:
            report = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"{path}: unreadable ({e})")
            continue
        case = infer_case(report)
        if case not in golds:
            print(f"{path}: not one of the sample cases; skipped")
            continue
        r = score(report, golds[case])
        r["file"] = Path(path).name
        results.append(r)
        if args.per_field:
            print(f"\n{r['file']}  [{case} · {r['provider']} · {r['model']}]")
            for p, k, b, d, w in r["rows"]:
                flag = "" if b in ("hit", "abstained") else "  <--"
                print(f"  {b:<13} w{w} {p:<45} {d}{flag}")

    if not results:
        return 1
    print(f"\n{'file':<34} {'case':<7} {'provider':<7} {'model':<22} {'score':>5} {'rec':>5} {'prec':>5} "
          f"{'hit':>3} {'om':>3} {'wr':>3} {'abs':>3} {'hal':>3} {'sec':>6}")
    for r in results:
        print(f"{r['file'][:34]:<34} {r['case']:<7} {str(r['provider']):<7} {str(r['model'])[:22]:<22} "
              f"{r['weighted']:>5} {str(r['recall']):>5} {str(r['precision']):>5} "
              f"{r['hit']:>3} {r['omission']:>3} {r['wrong']:>3} {r['abstained']:>3} {r['hallucination']:>3} "
              f"{str(r['extract_s']):>6}")
    if args.csv:
        keys = [k for k in results[0] if k != "rows"]
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in results:
                w.writerow({k: r[k] for k in keys})
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
