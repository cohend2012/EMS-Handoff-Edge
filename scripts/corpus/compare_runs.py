#!/usr/bin/env python3
"""Compare two Stage 5 report folders, e.g. the frozen baseline and a post-fix run.

    compare_runs.py ~/reports ~/reports_postfix [--log-a CSV] [--log-b CSV] [--bursts] [--md FILE]

Side by side for the two folders:
  crashes          log rows that failed or timed out for a reason other than a refusal,
                   and wrong-model rejections, each listed with its error
  refusals         'too short to process' errors (the pre-fix refusal) and reports whose
                   alert_type is UNCLEAR (what a short transcript gets after the fix)
  missing reports  targets from ~/labels/callsets.csv with no report file in the folder
  hallucination    forward claims from check_hallucinations.check_folder: how many were
                   found in the transcript (yes / partial / no), 'no' by field group
  fields filled    clinical fields with a value, mean per report and total, counted over the
                   reports that exist in both folders so the two numbers are comparable
  confidence       data_quality.confidence, combined reports and burst reports separately
then per-set diffs of alert_type, patient.sex and patient.age: every combined report, and
the burst reports that changed (all bursts with --bursts).

The run log is found as <folder>/stage5_log.csv, else ~/labels/stage5_log.csv for ~/reports and
~/labels/stage5_<suffix>_log.csv for ~/reports_<suffix>; --log-a/--log-b override."""
import argparse, csv, json, statistics, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_hallucinations as ch   # noqa: E402 - sibling script, importable since it gained main()

HOME = Path.home(); LABELS = HOME / "labels"
REFUSAL = "too short to process"
UNKNOWN = ("unknown", "null", "none", "n/a", "not stated", "not specified", "not described", "not provided",
           "not mentioned", "not reported", "not assessed", "not documented", "not available", "unavailable")
# the fields a report can only fill from the transcript (recommendations and summary are generated)
CLINICAL = ["alert_type", "alert_level", "eta_minutes", "chief_complaint", "mechanism_or_onset",
            "patient.age", "patient.sex", "vital_signs.blood_pressure", "vital_signs.heart_rate",
            "vital_signs.respiratory_rate", "vital_signs.spo2", "vital_signs.temperature", "vital_signs.blood_glucose",
            "vital_signs.gcs", "assessment_findings.level_of_consciousness", "assessment_findings.airway",
            "assessment_findings.breathing", "assessment_findings.circulation", "assessment_findings.neuro",
            "assessment_findings.other_findings", "ems_interventions", "history.past_medical", "history.medications",
            "history.allergies", "history.code_status"]


def get(d, path):
    for part in path.split("."):
        d = d.get(part) if isinstance(d, dict) else None
    return d


def filled(v):
    if isinstance(v, list):
        return any(filled(x) for x in v)
    if isinstance(v, dict):
        return any(filled(x) for x in v.values())
    return v not in (None, "") and str(v).strip().lower() not in UNKNOWN


def find_log(folder, override):
    if override:
        return Path(override).expanduser()
    cands = [folder / "stage5_log.csv"]
    if folder == HOME / "reports":
        cands.append(LABELS / "stage5_log.csv")
    elif folder.name.startswith("reports_"):
        cands.append(LABELS / f"stage5_{folder.name[len('reports_'):]}_log.csv")
    return next((c for c in cands if c.exists()), None)


def load_run(folder, log_path):
    """Everything compare needs about one folder."""
    rows = list(csv.DictReader(open(LABELS / "callsets.csv")))
    sets = sorted({r["callset_id"] for r in rows})
    targets = [(cid, f"{cid}_combined", "combined") for cid in sets] + [(r["callset_id"], r["segment_id"], "burst") for r in rows]
    reports = {}
    for cid, rid, kind in targets:
        p = folder / f"{rid}.json"
        if p.exists():
            try:
                reports[rid] = json.load(open(p))
            except ValueError as e:
                print(f"  {p}: unreadable ({e})", file=sys.stderr)
    log = list(csv.DictReader(open(log_path))) if log_path and log_path.exists() else []
    bad = [r for r in log if r["status"] not in ("ok", "skipped")]
    # a target that later succeeded is not a crash of the run, but it is still a failed attempt; list both
    ok_targets = {r["target"] for r in log if r["status"] == "ok"}
    crashes = [r for r in bad if r["status"] != "rejected_wrong_model" and REFUSAL not in r["error"].lower()]
    wrong_model = [r for r in bad if r["status"] == "rejected_wrong_model"]
    refused = [r for r in bad if REFUSAL in r["error"].lower()]
    unclear = [rid for rid, rep in reports.items() if str(rep.get("alert_type")).upper() == "UNCLEAR"]
    missing = [rid for _, rid, _ in targets if rid not in reports]
    fwd, _ = ch.check_folder(reports_dir=folder)
    return {"folder": folder, "log": log_path, "targets": targets, "reports": reports, "ok_targets": ok_targets,
            "crashes": crashes, "wrong_model": wrong_model, "refused": refused, "unclear": unclear, "missing": missing, "fwd": fwd}


def fields_filled(run, shared):
    per = [sum(filled(get(run["reports"][rid], f)) for f in CLINICAL) for rid in shared if rid in run["reports"]]
    return (statistics.mean(per) if per else 0.0, sum(per), len(per))


def confidence(run, kind):
    kinds = {rid: k for _, rid, k in run["targets"]}
    return Counter(str((rep.get("data_quality") or {}).get("confidence")) for rid, rep in run["reports"].items() if kinds[rid] == kind)


def age(rep):
    a, u = get(rep, "patient.age"), get(rep, "patient.age_unit")
    if not filled(a):
        return "-"
    return f"{a}" + (f" {u}" if filled(u) and str(u).lower() not in ("years", "year", "y") else "")


def sex(rep):
    s = get(rep, "patient.sex")
    return str(s)[:1].upper() if filled(s) else "-"


def alert(rep):
    a = get(rep, "alert_type")
    return str(a) if filled(a) else "-"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--log-a"); ap.add_argument("--log-b")
    ap.add_argument("--bursts", action="store_true", help="show every burst in the per-set diff, not only the changed ones")
    ap.add_argument("--md", help="also write the output to this file")
    args = ap.parse_args(argv)
    A, B = Path(args.a).expanduser().resolve(), Path(args.b).expanduser().resolve()
    runs = [load_run(A, find_log(A, args.log_a)), load_run(B, find_log(B, args.log_b))]
    na, nb = A.name, B.name
    w = max(len(na), len(nb), 18)
    out = []
    say = out.append

    def pair(label, fa, fb):
        say(f"{label:<28} {str(fa):<{w}}  {str(fb):<{w}}")

    say(f"{'':<28} {na:<{w}}  {nb:<{w}}")
    pair("folder", A, B)
    pair("run log", runs[0]["log"] or "none found", runs[1]["log"] or "none found")
    pair("reports present", f"{len(runs[0]['reports'])}/{len(runs[0]['targets'])}", f"{len(runs[1]['reports'])}/{len(runs[1]['targets'])}")
    say("")
    say("CRASHES AND REFUSALS (from the run logs; a target that later succeeded is still listed)")
    pair("crashes (failed/timeout)", len(runs[0]["crashes"]), len(runs[1]["crashes"]))
    pair("wrong-model rejections", len(runs[0]["wrong_model"]), len(runs[1]["wrong_model"]))
    pair("refusals (too short)", len(runs[0]["refused"]), len(runs[1]["refused"]))
    pair("UNCLEAR reports", len(runs[0]["unclear"]), len(runs[1]["unclear"]))
    pair("missing reports", len(runs[0]["missing"]), len(runs[1]["missing"]))
    for run, name in zip(runs, (na, nb)):
        for r in run["crashes"] + run["refused"]:
            later = " (succeeded later)" if r["target"] in run["ok_targets"] else ""
            say(f"  {name}: {r['status']:<8} {r['target']:<16} {r['error'][:110]}{later}")
        if run["unclear"]:
            say(f"  {name}: UNCLEAR {', '.join(run['unclear'])}")
        if run["missing"]:
            say(f"  {name}: missing {', '.join(run['missing'])}")
    say("")
    say("HALLUCINATION CANDIDATES (forward claims checked against the transcript)")
    for status in ("yes", "partial", "no"):
        pair(f"  claims found: {status}", *[sum(1 for r in run["fwd"] if r["found_in_transcript"] == status) for run in runs])
    pair("  claims total", *[len(run["fwd"]) for run in runs])
    groups = ("ems_interventions", "history.medications", "history.past_medical", "history.allergies", "history.code_status", "vital_signs", "patient.age", "eta_minutes")
    for g in groups:
        pair(f"  'no' in {g}", *[sum(1 for r in run["fwd"] if r["found_in_transcript"] == "no" and r["field"].startswith(g)) for run in runs])
    shared = [rid for rid in runs[0]["reports"] if rid in runs[1]["reports"]]
    say(f"  over the {len(shared)} reports present in both folders:")
    for status in ("yes", "partial", "no"):
        pair(f"    claims found: {status}", *[sum(1 for r in run["fwd"] if r["found_in_transcript"] == status and r["report"] in shared) for run in runs])
    say("")
    say(f"FIELDS FILLED ({len(CLINICAL)} clinical fields; over the {len(shared)} reports present in both folders)")
    fa, fb = fields_filled(runs[0], shared), fields_filled(runs[1], shared)
    pair("  mean per report", f"{fa[0]:.1f}", f"{fb[0]:.1f}")
    pair("  total", fa[1], fb[1])
    say("")
    say("CONFIDENCE (data_quality.confidence)")
    for kind in ("combined", "burst"):
        ca, cb = confidence(runs[0], kind), confidence(runs[1], kind)
        for level in ("HIGH", "MEDIUM", "LOW", "None"):
            if ca[level] or cb[level]:
                pair(f"  {kind} {level}", ca[level], cb[level])
    say("")
    say(f"PER-SET DIFFS (alert_type / sex / age; {na} -> {nb}; * = changed, - = empty, missing = no report)")
    kinds = {rid: k for _, rid, k in runs[0]["targets"]}
    say(f"  {'set':<6} {'report':<15} {'alert_type':<24} {'sex':<16} {'age':<24}")
    for cid, rid, kind in runs[0]["targets"]:
        ra, rb = runs[0]["reports"].get(rid), runs[1]["reports"].get(rid)
        cells = []
        changed = False
        for fn in (alert, sex, age):
            va = fn(ra) if ra else "missing"; vb = fn(rb) if rb else "missing"
            changed |= va != vb
            cells.append(va if va == vb else f"{va} -> {vb}")
        if kind == "burst" and not changed and not args.bursts:
            continue
        say(f"{'*' if changed else ' '} {cid:<6} {rid:<15} {cells[0]:<24} {cells[1]:<16} {cells[2]:<24}")
    text = "\n".join(out)
    print(text)
    if args.md:
        Path(args.md).expanduser().write_text(f"# {na} vs {nb}\n\n```\n{text}\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
