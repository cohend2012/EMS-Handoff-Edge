#!/usr/bin/env python3
"""Stage 6: one scenario per call-set.
  1st: the combined report's alert_type + chief_complaint (per-burst reports as a cross-check)
  2nd: transcript keywords
  scenario in {stroke, STEMI, sepsis, trauma, respiratory, cardiac_arrest, other, unclear}
  confidence: 0.9 report and transcript agree (strong keywords); 0.75 report only; 0.6 transcript
  only (no report); 0.5 report and transcript conflict; 0.4 nothing usable -> 'unclear'.
  -0.1 if a per-burst report disagrees with the combined report. < 0.7 -> ~/scenarios/needs_review/.
Writes ~/labels/labels.csv and copies (never moves) each set folder to ~/scenarios/<scenario>/<id>/."""
import csv, json, re, shutil
from pathlib import Path

HOME = Path.home(); SETS = HOME/"callsets"; LABELS = HOME/"labels"; OUT = HOME/"scenarios"
KW = {  # transcript keywords, scenario -> phrases
    "stroke": ["stroke", "lam score", "lams", "cincinnati", "last known well", "last-known well", "facial droop", "slurred", "arm drift", "aphasia", "hemipare", "tia", "vision change", "change in vision"],
    "STEMI": ["stemi", "st elevation", "st-elevation", "cath lab", "mi alert", "heart attack"],
    "sepsis": ["sepsis", "septic", "infection", "fever", "febrile", "lactate", "hypotens"],
    "trauma": ["trauma", "fell", "fall", "laceration", "motorcycle", "scooter", "struck", "tackled", "head injury", "backboard", "c-collar", "collar", "mvc", "gunshot", "stab", "bleeding"],
    "respiratory": ["respiratory distress", "trouble breathing", "shortness of breath", "wheez", "albuterol", "duoneb", "copd", "asthma", "difficulty breathing", "dyspnea", "saliva", "drool", "stridor"],
    "cardiac_arrest": ["cardiac arrest", "cpr", "compressions", "rosc", "asystole", "v-fib", "vfib", "pulseless", "code blue"],
}
ALERT_MAP = {"STROKE": "stroke", "STEMI": "STEMI", "SEPSIS": "sepsis", "TRAUMA": "trauma", "CARDIAC_ARREST": "cardiac_arrest"}
CC_MAP = [("respiratory", r"respiratory|breath|dyspnea|wheez|asthma|copd|drool|saliva"), ("stroke", r"stroke|vision|slurred|droop|weakness|altered mentation|neuro"),
          ("trauma", r"trauma|fall|fell|lacerat|motorcycle|scooter|struck|injur|tackl"), ("sepsis", r"sepsis|septic|infection|fever"),
          ("cardiac_arrest", r"arrest|cpr|pulseless"), ("STEMI", r"stemi|st elevation")]

def from_report(rep):
    if not rep: return None, "no combined report"
    at = str(rep.get("alert_type") or "").upper(); cc = str(rep.get("chief_complaint") or "").lower()
    if at in ALERT_MAP: return ALERT_MAP[at], f"alert_type={at}"
    for sc, pat in CC_MAP:
        if re.search(pat, cc): return sc, f"alert_type={at or 'null'}, chief_complaint matches {sc}: '{cc[:60]}'"
    return "other", f"alert_type={at or 'null'}, chief_complaint '{cc[:60]}' fits no named scenario"

def from_transcript(t):
    hits = {sc: [k for k in kws if k in t] for sc, kws in KW.items()}
    hits = {sc: h for sc, h in hits.items() if h}
    if not hits: return None, {}
    best = max(hits, key=lambda s: len(hits[s]))
    return best, hits

def phrases(t, words, n=4):
    out = []
    for w in words[:n]:
        i = t.find(w)
        if i >= 0: out.append(t[max(0, i-30): i+len(w)+30].replace("\n", " ").strip())
    return " | ".join(out)

rows = []; (OUT/"needs_review").mkdir(parents=True, exist_ok=True)
for d in sorted(SETS.glob("CS*")):
    cid = d.name; info = json.load(open(d/"callset_info.json")); t = (d/"combined_transcript.txt").read_text().lower()
    rep = json.load(open(d/"combined_report.json")) if (d/"combined_report.json").exists() else None
    bursts = [json.load(open(p)) for p in sorted(d.glob("S*_report.json"))]
    r_sc, r_why = from_report(rep); t_sc, hits = from_transcript(t)
    just = [r_why]
    if r_sc and t_sc and r_sc == t_sc: sc, conf = r_sc, 0.9; just.append(f"transcript keywords agree: {hits[t_sc][:4]}")
    elif r_sc and t_sc and r_sc == "other" and len(hits[t_sc]) >= 2: sc, conf = t_sc, 0.6; just.append(f"report said other but transcript keywords point to {t_sc}: {hits[t_sc][:4]}")
    elif r_sc and t_sc: sc, conf = r_sc, 0.5; just.append(f"CONFLICT: transcript keywords point to {t_sc}: {hits[t_sc][:4]}")
    elif r_sc: sc, conf = r_sc, 0.75; just.append("no scenario keywords in transcript")
    elif t_sc: sc, conf = t_sc, 0.6; just.append(f"transcript keywords only: {hits[t_sc][:4]}")
    else: sc, conf = "unclear", 0.4; just.append("no report and no keywords")
    if sc == "other" and conf >= 0.75 and len(t.split()) < 40: sc, conf = "unclear", 0.5; just.append("too little speech to classify")
    b_types = {str(b.get("alert_type")) for b in bursts if b.get("alert_type")}
    if rep and b_types and str(rep.get("alert_type")) not in b_types and len(b_types) > 0:
        conf = round(conf - 0.1, 2); just.append(f"burst reports disagree: {sorted(b_types)} vs combined {rep.get('alert_type')}")
    alert_level = (rep or {}).get("alert_level") or ""
    rows.append({"callset_id": cid, "scenario": sc, "alert_level": alert_level, "num_calls": info["num_bursts"], "confidence": conf,
                 "justification": "; ".join(just), "key_phrases": phrases(t, hits.get(t_sc, []) if t_sc else []), "bin": "needs_review" if conf < 0.7 else sc})
    dest = OUT/("needs_review" if conf < 0.7 else sc)/cid
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(d, dest)
with open(LABELS/"labels.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["callset_id", "scenario", "alert_level", "num_calls", "confidence", "justification", "key_phrases"], extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print("per scenario:", {s: sum(1 for r in rows if r["scenario"] == s) for s in sorted({r["scenario"] for r in rows})})
print("needs_review:", [r["callset_id"] for r in rows if r["bin"] == "needs_review"])
print("calls per set:", {n: sum(1 for r in rows if r["num_calls"] == n) for n in sorted({r["num_calls"] for r in rows})})
print("\nexamples:")
for r in rows[:5]: print(f"  {r['callset_id']} {r['scenario']} ({r['confidence']}): {r['justification']}")
