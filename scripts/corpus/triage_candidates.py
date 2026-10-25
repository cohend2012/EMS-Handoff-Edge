#!/usr/bin/env python3
"""Triage the hallucination checker's rows for a report folder into a review sheet.

    triage_candidates.py ~/reports_round2 [--out ~/reports_round2/review]

Runs check_hallucinations.check_folder on the folder, gives every row a triage and a reason,
and writes two csvs into --out (default <folder>/review):
  triage_all.csv    every row with its triage (likely_noise / uncertain / likely_real)
  review_sheet.csv  likely_real + uncertain only, one row per set / field / value, with empty
                    human_verdict and notes columns for the reviewer
Rules (the ones used for the baseline sheet of 2026-10-24 16:36, written down this time):
  claim found in the transcript ................................. likely_noise
  claim partly found (number elsewhere, delivery method unclear) . uncertain
  NKDA or Full code never spoken ................................ likely_real
  a GCS breakdown when only the total was spoken ................ likely_real
  a single character or non-word as a medication ................ likely_real
  prompt placeholder text echoed as a value ..................... likely_noise
  a vital, age or ETA number not found .......................... uncertain
  any other named term absent from the transcript ............... likely_real
  missed: sex given by pronouns, report says nothing ............ likely_real
  missed: unit or destination hospital .......................... likely_noise (schema has no field)
  missed: age or ETA differ ..................................... uncertain"""
import argparse, csv, re, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_hallucinations as ch   # noqa: E402

PLACEHOLDERS = {"list or nkda", "full code / dnr / dni / other if mentioned", "list of medications", "list of conditions",
                "what was done", "specifics (dose, route, size, etc.)", "patient response if mentioned", "if mentioned", "not provided",
                "not mentioned", "not stated", "unknown", "none", "null", "n/a"}


def triage_forward(r):
    f, v, status = r["field"], r["claimed_value"].strip(), r["found_in_transcript"]
    base = f.split("[")[0]
    if status == "yes":
        return "likely_noise", "claim found in transcript"
    if status == "partial":
        return "uncertain", ("delivery method not in nearby transcript" if base == "ems_interventions" and re.search(r"oxygen|o2", v, re.I)
                             else "partly supported; check the phrase")
    if v.lower() in PLACEHOLDERS:
        return "likely_noise", "placeholder text, now scrubbed to null by the pipeline"
    if re.fullmatch(r"nkda|no known (?:drug )?allergies", v, re.I) or (base == "history.code_status" and re.fullmatch(r"full code", v, re.I)):
        return "likely_real", f"{v} never spoken (known 4B filler)"
    if base == "history.medications" and (len(v) <= 2 or not re.search(r"[a-z]{3}", v, re.I)):
        return "likely_real", "garbage token as a medication"
    if base == "vital_signs.gcs" and len([n for n in re.findall(r"\d+", v) if int(n) <= 6]) >= 3:
        return "likely_real", "GCS breakdown invented; only the total was spoken"
    if base.startswith("vital_signs") or base in ("patient.age", "eta_minutes"):
        return "uncertain", "number not found; may be spelled out or mis-heard"
    return "likely_real", "named term absent from transcript"


def triage_reverse(r):
    f = r["field"]
    if f == "patient.sex":
        return "likely_real", "pronouns give the sex; report says nothing"
    if f in ("ems_unit", "destination_hospital"):
        return "likely_noise", "report schema has no field for this (schema gap, not a model miss)"
    return "uncertain", "transcript and report disagree; check the phrase"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder"); ap.add_argument("--out")
    args = ap.parse_args(argv)
    folder = Path(args.folder).expanduser().resolve()
    out = Path(args.out).expanduser() if args.out else folder / "review"
    out.mkdir(parents=True, exist_ok=True)
    fwd, rev = ch.check_folder(reports_dir=folder)
    rows = []
    for r in fwd:
        t, why = triage_forward(r)
        rows.append({"source": "hallucination", "triage": t, "callset_id": r["callset_id"], "report": r["report"], "field": r["field"],
                     "value": r["claimed_value"], "transcript_phrase": r["nearest_transcript_phrase"], "reason": why})
    for r in rev:
        t, why = triage_reverse(r)
        rows.append({"source": "miss", "triage": t, "callset_id": r["callset_id"], "report": r["report"], "field": r["field"],
                     "value": f"transcript {r['transcript_value']} / report {r['report_value']}", "transcript_phrase": r["transcript_phrase"], "reason": why})
    with open(out / "triage_all.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["source", "triage", "callset_id", "report", "field", "value", "transcript_phrase", "reason"]); w.writeheader(); w.writerows(rows)
    sheet = {}
    for r in rows:
        if r["triage"] == "likely_noise":
            continue
        key = (r["callset_id"], r["field"].split("[")[0], r["value"].lower())
        if key in sheet:
            if r["report"] not in sheet[key]["report"].split(";"):
                sheet[key]["report"] += ";" + r["report"]
            continue
        sheet[key] = {"triage": r["triage"], "source": r["source"], "callset_id": r["callset_id"], "report": r["report"], "field": r["field"].split("[")[0],
                      "value": r["value"], "transcript_phrase": r["transcript_phrase"], "reason": r["reason"], "human_verdict": "", "notes": ""}
    order = {"likely_real": 0, "uncertain": 1}
    sheet_rows = sorted(sheet.values(), key=lambda r: (order[r["triage"]], r["callset_id"], r["field"]))
    with open(out / "review_sheet.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["triage", "source", "callset_id", "report", "field", "value", "transcript_phrase", "reason", "human_verdict", "notes"]); w.writeheader(); w.writerows(sheet_rows)
    c = Counter((r["source"], r["triage"]) for r in rows)
    print(f"{folder}: {len(fwd)} claims + {len(rev)} misses -> {dict(c)}")
    print(f"review sheet: {len(sheet_rows)} rows ({sum(r['triage'] == 'likely_real' for r in sheet_rows)} likely_real, "
          f"{sum(r['triage'] == 'uncertain' for r in sheet_rows)} uncertain) -> {out / 'review_sheet.csv'}")
    print("reasons:", dict(Counter(r["reason"] for r in sheet_rows).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
