#!/usr/bin/env python3
"""First-pass support check of pipeline reports against their transcripts (for the human
reviewer, not a verdict).
  forward : each concrete claim in a report (interventions, medications, allergies, history,
            vital values, age, ETA, code status) -> is the term present in the transcript?
            yes / partial / no, with the nearest transcript phrase.
  reverse : things clearly in the transcript the report missed: sex from pronouns, age,
            unit, destination hospital, ETA.
Outputs ~/labels/hallucination_candidates.csv and ~/labels/missed_candidates.csv.
Combined reports are checked against the set's combined transcript; a single-burst report is
checked against the transcript the pipeline actually saw (its _meta.raw_transcript)."""
import csv, json, re
from pathlib import Path

HOME = Path.home(); SETS = HOME/"callsets"; REPORTS = HOME/"reports"; LABELS = HOME/"labels"
SYN = {  # claim term -> ways it is spoken on the radio
    "oxygen": ["nasal cannula", "non-rebreather", "nonrebreather", "nrb", "liters", "blow-by", "blow by", "glow-by", "on oxygen", "oxygen via", "oxygen by", "supplemental oxygen", "o2 via", "high flow", "on 2 l", "on 4 l", "on 15 l"],
    "iv": ["iv", "gauge", "saline lock", "working on a line", "working on the line", "line placed", "line established", "line in", "iv access", "vascular access", "working on access", "got a line", "have a line", "18 in the"],
    "aspirin": ["aspirin", "asa", "324"], "nitroglycerin": ["nitro", "nitroglycerin", "0.4", "400 micrograms"], "nitro": ["nitro", "nitroglycerin"],
    "naloxone": ["narcan", "naloxone"], "narcan": ["narcan", "naloxone"], "midazolam": ["versed", "midazolam"], "versed": ["versed", "midazolam"],
    "fentanyl": ["fentanyl"], "adenosine": ["adenosine"], "dexamethasone": ["dexamethasone", "texamethasone", "decadron"],
    "albuterol": ["albuterol", "duoneb", "duo-neb", "neb"], "ipratropium": ["duoneb", "ipratropium", "atrovent"], "nebulizer": ["neb", "duoneb", "albuterol"],
    "c-collar": ["collar", "c-collar"], "cervical collar": ["collar", "c-collar"], "collar": ["collar"], "backboard": ["backboard", "board"],
    "bag-valve-mask": ["bag valve", "bvm", "back valve", "ventilat"], "bvm": ["bag valve", "bvm", "back valve", "ventilat"], "ventilation": ["ventilat", "bag valve", "back valve"],
    "12-lead": ["12 lead", "12-lead", "twelve lead", "ekg", "ecg"], "ekg": ["12 lead", "12-lead", "ekg", "ecg"], "ecg": ["12 lead", "12-lead", "ekg", "ecg"],
    "cardiac monitor": ["monitor"], "monitor": ["monitor"], "glucose": ["glucose", "blood sugar", "sugar"], "blood glucose": ["glucose", "blood sugar", "sugar"],
    "nkda": ["nkda", "no known allergies", "no allergies", "no known drug", "allergies none"], "no known allergies": ["nkda", "no known allergies", "no allergies", "no known drug"],
    "sternal rub": ["sternum rub", "sternal rub"], "trauma pads": ["trauma pad"], "hemorrhage control": ["bleeding is controlled", "pressure", "trauma pad"],
    "bleeding control": ["bleeding is controlled", "bleeding controlled", "pressure", "trauma pad"], "splint": ["splint"], "tourniquet": ["tourniquet"],
    "cpr": ["cpr", "compressions"], "defibrillation": ["shock", "defib"], "epinephrine": ["epi", "epinephrine"], "amiodarone": ["amiodarone", "amio"],
    "dnr": ["dnr", "do not resuscitate"], "full code": ["full code"], "polst": ["polst", "molst"],
    "hypertension": ["hypertension", "htn", "high blood pressure"], "diabetes": ["diabetes", "diabetic", "dm"], "stroke": ["stroke", "cva", "tia"],
    "atrial fibrillation": ["afib", "a-fib", "atrial fib"], "afib": ["afib", "a-fib", "atrial fib"], "dementia": ["dementia"], "cabg": ["bypass", "cabg"],
    "coronary artery bypass": ["bypass", "cabg"], "dialysis": ["dialysis"], "kidney failure": ["kidney failure", "renal", "dialysis"], "ckd": ["kidney", "renal", "dialysis"],
    "angina": ["angina"], "valve repair": ["valve"], "metoprolol": ["metoprolol", "metaprol"], "lisinopril": ["lisinopril", "lisinoprol"], "amlodipine": ["amlodipine", "amilodipine"],
}
STOP = set("the a an and or of to with for in on at by via was were is are be been given administered administration applied placed established patient pt per mg mcg ml l rate dose x1 one two unknown not no none null line delivered flow method site unspecified specified ambient supplemental air room provided performed initiated tended resisted controlled applied likely meaning care".split())
UNIT = re.compile(r"\b(?:medic|medics|medix|ambulance|unit|paramedic|m)s?\s*(7\d\d)\b", re.I)
AGE = re.compile(r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)[- ]?(year|month|week)s?[- ]old\b|\bage (?:about |of |approximately )?(\d{1,3})\b", re.I)
WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
ETA = re.compile(r"\b(?:eta|be there in|be at your facility in|see you in|arrive in|out)\b[^.]{0,30}?\b(\d{1,2})\b[- ]?(?:minute|min)|\b(\d{1,2})[- ]?(?:minute|min)s?\s*(?:eta|out|from your facility|to your facility)", re.I)
HOSP = {"holy cross germantown": ["holy cross germantown", "holycroft", "germantown"], "holy cross": ["holy cross"], "shady grove": ["shady grove", "shagos", "shady grubble"], "suburban": ["suburban"]}

NUMWORDS = {"one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10,"eleven":11,"twelve":12,"thirteen":13,"fourteen":14,"fifteen":15,"sixteen":16,"seventeen":17,"eighteen":18,"nineteen":19,"twenty":20,"thirty":30,"forty":40,"fifty":50,"sixty":60,"seventy":70,"eighty":80,"ninety":90,"hundred":100}
def norm(s):
    s = re.sub(r"\s+", " ", str(s).lower())
    s = re.sub(r"\b(\d)-(\d)\b", lambda m: m.group(1)+m.group(2), s)          # "3-0" -> 30
    return re.sub(r"\b(" + "|".join(NUMWORDS) + r")\b", lambda m: str(NUMWORDS[m.group(1)]), s)
VITAL_CTX = {"blood_pressure": r"(?:pressure|bp)\b[^.]{0,40}", "heart_rate": r"(?:heart rate|pulse|hr|rate)\b[^.]{0,30}", "respiratory_rate": r"(?:resp\w*|breathing|rr)\b[^.]{0,30}",
             "spo2": r"(?:sat\w*|spo2|pulse ox\w*|oxygen saturation|o2)\b[^.]{0,30}", "blood_glucose": r"(?:sugar|glucose|bgl|cbg)\b[^.]{0,30}", "gcs": r"(?:gcs|dcs|glasgow)\b[^.]{0,30}", "temperature": r"(?:temp\w*)\b[^.]{0,30}"}
def vital_in_context(t, key, value):
    """numbers of the value must appear inside a phrase about that vital; else partial (number elsewhere) or no"""
    nums = re.findall(r"\d+(?:\.\d+)?", str(value))
    if not nums: return None, ""
    ctxs = [m.group(0) for m in re.finditer(VITAL_CTX.get(key, r"$^"), t)]
    in_ctx = [n for n in nums if any(re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", c) for c in ctxs)]
    anywhere = [n for n in nums if re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", t)]
    if len(in_ctx) == len(nums): return "yes", "..." + next(c for c in ctxs if re.search(r"(?<![\d.])" + re.escape(nums[0]) + r"(?![\d])", c))[:120] + "..."
    if anywhere: return "partial", (("number found but not as this vital; transcript says: ..." + ctxs[0][:100] + "...") if ctxs else context(t, anywhere[0]))
    return "no", ("transcript says: ..." + ctxs[0][:100] + "...") if ctxs else nearest(t, str(value))
def sentences(t): return [x.strip() for x in re.split(r"(?<=[.?!])\s+|\n+", t) if x.strip()]
def context(t, term, w=45):
    i = t.find(term); return "" if i < 0 else "..." + t[max(0, i-w): i+len(term)+w].replace("\n", " ") + "..."
def nearest(t, claim):
    words = [w for w in re.findall(r"[a-z0-9./-]+", norm(claim)) if w not in STOP]
    best, score = "", 0
    for s in sentences(t):
        sc = sum(1 for w in words if w in s.lower())
        if sc > score: best, score = s, sc
    return best[:160]

def check_claim(t, claim):
    """-> (yes|partial|no, phrase). Term presence with synonyms; numbers must appear as tokens."""
    c = norm(claim); hits, misses = [], []
    keys = [k for k in SYN if re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", c)]   # whole words: "iv" must not match "delivery"
    for k in keys:
        found = [s for s in SYN[k] if re.search(r"(?<![a-z0-9])" + re.escape(s) + r"s?(?![a-z0-9])", t)]
        (hits if found else misses).append((k, found[0] if found else None))
    words = [w for w in re.findall(r"[a-z0-9./-]+", c) if w not in STOP and len(w) > 2 and not any(w in k for k in keys)]
    for w in words:
        base = w.rstrip("s")
        if re.search(r"(?<![a-z0-9])" + re.escape(base), t): hits.append((w, base))
        else: misses.append((w, None))
    if keys:   # the verdict rests on the named terms; loose words only decorate the phrase
        key_hits = [h for h in hits if h[0] in keys]
        if not key_hits: return "no", nearest(t, claim)
        if len(key_hits) == len(keys): return "yes", context(t, key_hits[0][1])
        return "partial", context(t, key_hits[0][1])
    if not hits: return "no", nearest(t, claim)
    return ("yes" if not misses else "partial"), context(t, hits[0][1])

def numbers_present(t, value):
    nums = re.findall(r"\d+(?:\.\d+)?", str(value)); 
    if not nums: return None, []
    found = [n for n in nums if re.search(r"(?<![\d.])" + re.escape(n) + r"(?![\d])", t)]
    return (("yes" if len(found) == len(nums) else "partial" if found else "no"), found)

def is_unknown(v): return v in (None, "", [], {}) or str(v).strip().lower() in ("unknown", "null", "none", "n/a", "not stated", "not specified")

def forward(cid, rid, rep, t):
    rows = []
    def add(field, val, status, phrase): rows.append({"callset_id": cid, "report": rid, "field": field, "claimed_value": re.sub(r"[,\n]+", ";", str(val))[:120], "found_in_transcript": status, "nearest_transcript_phrase": re.sub(r"[,\n]+", ";", phrase)})
    for i, iv in enumerate(rep.get("ems_interventions") or []):
        claim = " ".join(str(iv.get(k) or "") for k in ("intervention", "details")) if isinstance(iv, dict) else str(iv)
        if not is_unknown(claim.strip()): add(f"ems_interventions[{i}]", claim, *check_claim(t, claim))
    h = rep.get("history") or {}
    for i, m in enumerate(h.get("medications") or []):
        if not is_unknown(m): add(f"history.medications[{i}]", m, *check_claim(t, m))
    for i, m in enumerate(h.get("past_medical") or []):
        if not is_unknown(m): add(f"history.past_medical[{i}]", m, *check_claim(t, m))
    al = h.get("allergies")
    for a in (al if isinstance(al, list) else [al]):
        if not is_unknown(a): add("history.allergies", a, *check_claim(t, a))
    cs = h.get("code_status")
    if not is_unknown(cs): add("history.code_status", cs, *check_claim(t, cs))
    for k, v in (rep.get("vital_signs") or {}).items():
        if is_unknown(v): continue
        st, phrase = vital_in_context(t, k, v)
        if st is None: add(f"vital_signs.{k}", v, *check_claim(t, v))
        else: add(f"vital_signs.{k}", v, st, phrase)
    p = rep.get("patient") or {}
    if not is_unknown(p.get("age")):
        st, found = numbers_present(t, p["age"]); add("patient.age", p["age"], st or "no", context(t, found[0]) if found else nearest(t, "year old"))
    if not is_unknown(rep.get("eta_minutes")):
        st, found = numbers_present(t, rep["eta_minutes"]); add("eta_minutes", rep["eta_minutes"], st or "no", context(t, found[0]) if found else nearest(t, "minutes"))
    return rows

def reverse(cid, rid, rep, t, whole):
    rows = []
    def add(field, tv, rv, phrase): rows.append({"callset_id": cid, "report": rid, "field": field, "transcript_value": tv, "report_value": str(rv)[:80], "transcript_phrase": phrase[:160]})
    p = rep.get("patient") or {}
    male = len(re.findall(r"\b(?:he|his|him|male|gentleman|guy)\b", t)); female = len(re.findall(r"\b(?:she|her|hers|female|lady|woman)\b", t))
    sex = "Male" if male >= 2 and male > 2*female else "Female" if female >= 2 and female > 2*male else None
    if sex and norm(p.get("sex")) != sex.lower():
        m = re.search(r"\b(?:his|he|him|male)\b" if sex == "Male" else r"\b(?:she|her|female)\b", t); add("patient.sex", f"{sex} (pronouns he/his {male}, she/her {female})", p.get("sex"), context(t, m.group(0)) if m else "")
    ages = []
    for m in AGE.finditer(t):
        n = m.group(1) or m.group(3); n = WORDNUM.get(n.lower(), n) if isinstance(n, str) else n
        ages.append((int(n), (m.group(2) or "year").lower(), m.group(0)))
    if ages:
        a0 = ages[0]; ra = p.get("age"); ru = norm(p.get("age_unit") or "years")
        if is_unknown(ra) or int(float(ra)) != a0[0] or not ru.startswith(a0[1]): add("patient.age", f"{a0[0]} {a0[1]}s", f"{ra} {p.get('age_unit')}", context(t, a0[2]))
    units = sorted({m.group(1) for m in UNIT.finditer(t)})
    if units and not any(u in whole for u in units): add("ems_unit", ",".join(units), "(not in report)", context(t, units[0]))
    hosp = [h for h, al in HOSP.items() if any(a in t for a in al)]
    if "holy cross germantown" in hosp and "holy cross" in hosp: hosp.remove("holy cross")
    if hosp and not any(h in whole for h in hosp): add("destination_hospital", ",".join(hosp), "(not in report)", context(t, next(a for a in HOSP[hosp[0]] if a in t)))
    etas = [int(m.group(1) or m.group(2)) for m in ETA.finditer(t)]
    if etas and (is_unknown(rep.get("eta_minutes")) or int(float(rep["eta_minutes"])) not in etas): add("eta_minutes", ",".join(map(str, etas)), rep.get("eta_minutes"), context(t, str(etas[-1])))
    return rows

def check_folder(reports_dir=REPORTS, sets_dir=SETS):
    """(forward rows, reverse rows) for every report in reports_dir. A combined report is checked
    against its set's combined transcript, a burst report against the transcript it saw."""
    fwd, rev = [], []
    for d in sorted(sets_dir.glob("CS*")):
        cid = d.name; combined = norm((d/"combined_transcript.txt").read_text())
        bursts = [r["segment_id"] for r in csv.DictReader(open(LABELS/"callsets.csv")) if r["callset_id"] == cid]
        targets = [(f"{cid}_combined", reports_dir/f"{cid}_combined.json", combined)] + [(sid, reports_dir/f"{sid}.json", None) for sid in bursts]
        for rid, path, t in targets:
            if not path.exists(): continue
            rep = json.load(open(path)); t = t or norm(rep.get("_meta", {}).get("raw_transcript") or rep.get("_meta", {}).get("transcript") or "")
            whole = norm(json.dumps({k: v for k, v in rep.items() if k != "_meta"}))
            fwd += forward(cid, rid, rep, t); rev += reverse(cid, rid, rep, t, whole)
    return fwd, rev

def main():
    fwd, rev = check_folder()
    for name, rows in (("hallucination_candidates.csv", fwd), ("missed_candidates.csv", rev)):
        with open(LABELS/name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["callset_id"]); w.writeheader(); w.writerows(rows)
    n_rep = len({r["report"] for r in fwd} | {r["report"] for r in rev})
    print(f"reports checked: {n_rep}")
    print("forward claims:", len(fwd), {s: sum(1 for r in fwd if r['found_in_transcript'] == s) for s in ('yes', 'partial', 'no')})
    print("unsupported ('no') by field group:", {g: sum(1 for r in fwd if r['found_in_transcript'] == 'no' and r['field'].startswith(g)) for g in ('ems_interventions', 'history.medications', 'history.past_medical', 'history.allergies', 'history.code_status', 'vital_signs', 'patient.age', 'eta_minutes')})
    print("reverse (missed):", len(rev), {g: sum(1 for r in rev if r['field'] == g) for g in ('patient.sex', 'patient.age', 'ems_unit', 'destination_hospital', 'eta_minutes')})

if __name__ == "__main__":
    main()
