#!/usr/bin/env python3
"""
gen_scripts.py — generate radio handoff reports with gold answers: Phase 0 of docs/plans/LYAPUNOV_STT_PLAN.md.

    python scripts/corpus/gen_scripts.py                       # 200 scripts into $HANDOFF_DATASETS/synth_v1
    python scripts/corpus/gen_scripts.py --n 50 --seed 7 --out ~/datasets/synth_try
    python scripts/corpus/gen_scripts.py --show SY017           # print one script, its spoken form and gold

Each script is a paramedic's report to a hospital in the phrasing of the MoCo radio corpus (unit to
hospital, "how do you copy", "priority two", vitals "as follows", "we'll be at your facility in about
five minutes"), built from sampled facts: patient, complaint, onset, vitals, exam, history, home meds,
allergies, code status, interventions with doses, request, ETA. Eleven scenario families cover every
alert type the report schema knows, plus short fragments. Every fact is either spoken, and then checked
in the gold, or left unspoken, and then the gold demands null for it: that is the barrier rule
(ETA, allergies, code status, GCS breakdown) turned into test cases.

Written under <out>/:
    scripts/<id>.txt          the written script (what the transcriber should produce)
    scripts/<id>.spoken.txt   the spoken form for a TTS voice (scripts/air_check.py spoken())
    gold/<id>.json            tests/gold field format, plus "stated" (priority, eta, alert words) for
                              score_callsets.alert_truth and "marker" (the unit phrase) for benchmark.py
    index.tsv                 id, family, alert type, priority, ETA, words
    splits.json               train / val / test ids, 70/15/15 by script, stratified by family
    terms.txt                 every medical term surface used, for the medical-term error rate
Deterministic: the same --seed and --n give the same set; script i depends on seed and i only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import paths  # noqa: E402
from score_callsets import normalize  # noqa: E402


def spoken(text: str) -> str:
    import air_check  # scripts/air_check.py; the one spoken-form table
    return air_check.spoken(text)


# ----------------------------------------------------------------------------- the world

UNITS = [("Medic", (701, 759)), ("Ambulance", (701, 759)), ("Paramedic", (701, 759)),
         ("Medic", (7, 49)), ("Squad", (3, 29)), ("Unit", (11, 59))]
HOSPITALS = ["Shady Grove", "Suburban", "Holy Cross", "Montgomery General", "White Oak", "Germantown",
             "Sibley", "Memorial", "Riverside", "County General", "Washington Adventist", "Fairfax"]
TOD = ["morning", "afternoon", "evening"]
IV_SITES = ["left AC", "right AC", "left forearm", "right forearm", "left hand", "right hand"]

# (written surface, aliases the gold accepts, lexicon-ish category)
DRUGS: Dict[str, Tuple[List[str], str]] = {
    "aspirin": (["aspirin", "asa"], "drug"), "nitro": (["nitro", "nitroglycerin", "ntg"], "drug"),
    "nitroglycerin": (["nitro", "nitroglycerin", "ntg"], "drug"),
    "fentanyl": (["fentanyl"], "drug"), "ondansetron": (["ondansetron", "zofran"], "drug"),
    "Zofran": (["zofran", "ondansetron"], "drug"), "midazolam": (["midazolam", "versed"], "drug"),
    "Versed": (["versed", "midazolam"], "drug"), "albuterol": (["albuterol"], "drug"),
    "ipratropium": (["ipratropium", "atrovent", "duoneb"], "drug"), "epinephrine": (["epi"], "drug"),
    "epi": (["epi"], "drug"), "Benadryl": (["benadryl", "diphenhydramine"], "drug"),
    "diphenhydramine": (["diphenhydramine", "benadryl"], "drug"), "Narcan": (["narcan", "naloxone"], "drug"),
    "naloxone": (["naloxone", "narcan"], "drug"), "D10": (["d10", "dextrose", "d-10"], "drug"),
    "glucagon": (["glucagon"], "drug"), "amiodarone": (["amiodarone"], "drug"),
    "normal saline": (["saline", "fluid", "bolus", "ns"], "drug"), "lactated Ringer's": (["ringer", "lr", "fluid", "bolus"], "drug"),
    "ketamine": (["ketamine"], "drug"), "Toradol": (["toradol", "ketorolac"], "drug"),
    "Haldol": (["haldol", "haloperidol"], "drug"), "diltiazem": (["diltiazem", "cardizem"], "drug"),
    "adenosine": (["adenosine"], "drug"), "Solu-Medrol": (["solu-medrol", "methylprednisolone", "solumedrol"], "drug"),
    "magnesium": (["magnesium", "mag"], "drug"), "TXA": (["txa", "tranexamic"], "drug"),
    "oral glucose": (["glucose"], "drug"), "acetaminophen": (["acetaminophen", "tylenol"], "drug"),
}

PMH: Dict[str, List[str]] = {   # written surface -> gold aliases
    "hypertension": ["hypertension", "htn"], "type 2 diabetes": ["diabetes", "dm"], "diabetes": ["diabetes", "dm"],
    "type 1 diabetes": ["diabetes", "dm"], "CAD": ["cad", "coronary"], "coronary artery disease": ["coronary", "cad"],
    "hyperlipidemia": ["hyperlipidemia", "cholesterol", "hld"], "CHF": ["chf", "heart failure"],
    "AFib": ["afib", "a-fib", "atrial fib", "a fib"], "atrial fibrillation": ["atrial fib", "afib", "a-fib", "a fib"],
    "COPD": ["copd"], "asthma": ["asthma"], "prior stroke": ["stroke", "cva"], "prior CVA": ["cva", "stroke"],
    "seizure disorder": ["seizure", "epilep"], "epilepsy": ["epilep", "seizure"], "CKD": ["ckd", "kidney", "renal"],
    "chronic kidney disease": ["kidney", "ckd", "renal"], "ESRD on dialysis": ["esrd", "dialysis", "renal"],
    "dementia": ["dementia", "alzheimer"], "Alzheimer's dementia": ["alzheimer", "dementia"],
    "depression": ["depression"], "anxiety": ["anxiety"], "bipolar disorder": ["bipolar"],
    "schizophrenia": ["schizophrenia"], "hypothyroidism": ["hypothyroid", "thyroid"],
    "prior MI": ["mi", "myocardial", "heart attack"], "stents": ["stent"], "pacemaker": ["pacemaker"],
    "opioid use": ["opioid", "substance", "drug use"], "alcohol use": ["alcohol", "etoh"],
    "GERD": ["gerd", "reflux"], "obesity": ["obes"],
}
HOME_MEDS: Dict[str, List[str]] = {
    "metformin": ["metformin"], "lisinopril": ["lisinopril"], "atorvastatin": ["atorvastatin", "lipitor"],
    "metoprolol": ["metoprolol"], "amlodipine": ["amlodipine"], "losartan": ["losartan"],
    "Eliquis": ["eliquis", "apixaban"], "Xarelto": ["xarelto", "rivaroxaban"], "warfarin": ["warfarin", "coumadin"],
    "Plavix": ["plavix", "clopidogrel"], "insulin": ["insulin"], "Lantus": ["lantus", "insulin"],
    "furosemide": ["furosemide", "lasix"], "Lasix": ["lasix", "furosemide"], "donepezil": ["donepezil", "aricept"],
    "levothyroxine": ["levothyroxine", "synthroid"], "Keppra": ["keppra", "levetiracetam"],
    "gabapentin": ["gabapentin"], "albuterol inhaler": ["albuterol", "inhaler"], "Symbicort": ["symbicort"],
    "sertraline": ["sertraline", "zoloft"], "Seroquel": ["seroquel", "quetiapine"], "lithium": ["lithium"],
    "omeprazole": ["omeprazole", "prilosec"], "aspirin 81": ["aspirin", "asa"], "prenatal vitamins": ["prenatal"],
}
ALLERGENS = [("penicillin", ["penicillin", "pcn"]), ("sulfa", ["sulfa"]), ("morphine", ["morphine"]),
             ("codeine", ["codeine"]), ("shellfish", ["shellfish"]), ("latex", ["latex"]),
             ("contrast dye", ["contrast"]), ("ibuprofen", ["ibuprofen", "nsaid"]), ("amoxicillin", ["amoxicillin"])]

CODE_STATUS = [("Patient is a DNR.", ["dnr", "do not resuscitate"]),
               ("Patient is DNR DNI per paperwork.", ["dnr", "dni"]),
               ("Patient is full code.", ["full code", "full"]),
               ("Patient is DNR but full treatment per POLST form, which we have with us.", ["dnr", "polst"]),
               ("Family states patient is a full code.", ["full code", "full"])]

PRIORITY_LEVEL = {1: "RED", 2: "YELLOW", 3: "GREEN"}
PRIORITY_WORD = {1: "one", 2: "two", 3: "three"}


# ----------------------------------------------------------------------------- the scenario

@dataclass
class Vitals:
    bp: Optional[str] = None
    hr: Optional[int] = None
    rhythm: str = ""                 # " and irregular", ", sinus tach"
    rr: Optional[int] = None
    rr_note: str = ""                # " and shallow"
    spo2: Optional[int] = None
    o2: Optional[str] = None         # "on room air", "on 4 liters nasal cannula"; None = not stated
    o2_aliases: List[str] = field(default_factory=list)
    temp: Optional[str] = None       # "102.4"
    bg: Optional[Any] = None         # 186 or "high"
    gcs: Optional[int] = None
    gcs_parts: Optional[Tuple[int, int, int]] = None


@dataclass
class Intervention:
    text: str                       # how it is written in the script
    aliases: List[str]              # gold "list" alternatives (lowercase substrings)
    numbers: List[str] = field(default_factory=list)   # dose/size numbers that must appear ("324", "0.4", "18")
    weight: int = 3


@dataclass
class Scenario:
    family: str
    alert_type: str
    alert_accept: List[str]
    age: int
    age_unit: str                   # years | months
    sex: str                        # Male | Female
    cc_short: str                   # for the intro line
    cc_aliases: List[str]
    alert_words: List[str]
    story: List[str]                # onset / mechanism sentences, read before the vitals
    vitals: Vitals
    treat: List[str] = field(default_factory=list)   # interventions and responses, read after the exam
    exam: List[str] = field(default_factory=list)
    pmh: List[str] = field(default_factory=list)            # written surfaces (keys of PMH); [] = unknown
    pmh_stated: Optional[bool] = None                       # None = not stated; False = "no history"
    meds: List[str] = field(default_factory=list)
    meds_stated: Optional[bool] = None
    allergies: Optional[Tuple[str, List[str]]] = None       # (sentence, aliases) or None = not stated
    code: Optional[Tuple[str, List[str]]] = None
    interventions: List[Intervention] = field(default_factory=list)
    no_interventions: bool = False
    request: Optional[Tuple[str, List[str]]] = None         # (sentence, aliases)
    destination: Optional[str] = None
    priority: Optional[int] = None
    eta: Optional[int] = None
    extra_fields: List[Dict[str, Any]] = field(default_factory=list)
    terms: List[str] = field(default_factory=list)
    fragment: bool = False


def chance(r: random.Random, p: float) -> bool:
    return r.random() < p


def pick(r: random.Random, seq: Sequence):
    return seq[r.randrange(len(seq))]


def some(r: random.Random, seq: Sequence, k: int) -> list:
    return r.sample(list(seq), min(k, len(seq)))


def age_phrase(sc: Scenario) -> str:
    unit = "month" if sc.age_unit == "months" else "year"
    return f"{sc.age}-{unit}-old {sc.sex.lower()}"


def article(n: int) -> str:
    """"an 8-", "an 11-", "an 18-", "an 80-year-old"; otherwise "a"."""
    return "an" if n in (8, 11, 18) or 80 <= n <= 89 else "a"


def join_list(items: List[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + (", and " if len(items) > 2 else " and ") + items[-1]


# ---- common builders

def pick_vitals(r: random.Random, bp=(110, 150, 60, 92), hr=(64, 104), rr=(14, 22), spo2=(94, 99)) -> Vitals:
    sys_ = r.randint(bp[0], bp[1]); dia = r.randint(bp[2], min(bp[3], sys_ - 25))
    v = Vitals(bp=f"{sys_}/{dia}", hr=r.randint(*hr), rr=r.randint(*rr), spo2=r.randint(*spo2))
    return v


def set_o2(r: random.Random, v: Vitals, kind: str) -> None:
    if kind == "ra":
        v.o2, v.o2_aliases = "on room air", ["room air", "ra"]
    elif kind == "nc":
        n = pick(r, [2, 2, 4, 4, 6])
        v.o2, v.o2_aliases = f"on {n} liters nasal cannula", [f"{n} l", f"{n}l", "nasal", "cannula"]
    elif kind == "nrb":
        v.o2, v.o2_aliases = "on 15 liters non-rebreather", ["15", "non-rebreather", "nrb", "non rebreather"]
    elif kind == "cpap":
        v.o2, v.o2_aliases = "on CPAP", ["cpap"]
    elif kind == "bvm":
        v.o2, v.o2_aliases = "with BVM ventilations", ["bvm", "bag", "ventilat"]
    elif kind == "blowby":
        v.o2, v.o2_aliases = "on blow-by oxygen", ["blow", "oxygen", "o2"]
    else:
        v.o2 = None


def rhythm_for(r: random.Random, hr: int, afib: bool = False) -> str:
    """A rhythm phrase that agrees with the rate, or nothing."""
    if hr > 100:
        opts = ["", ", sinus tach", ", sinus tach"] + ([", AFib with RVR"] if afib and hr > 130 else [])
    elif hr < 60:
        opts = ["", ", sinus brady", " and irregular"]
    else:
        opts = ["", "", ", normal sinus", ", normal sinus with PVCs", " and irregular"]
    return pick(r, opts)


def set_gcs(r: random.Random, v: Vitals, total: int, p_breakdown: float = 0.4) -> None:
    v.gcs = total
    if chance(r, p_breakdown):
        parts = {15: (4, 5, 6), 14: (4, 4, 6), 13: (3, 4, 6), 12: (3, 3, 6), 11: (3, 3, 5), 10: (3, 2, 5),
                 9: (2, 2, 5), 8: (2, 2, 4), 7: (1, 2, 4), 6: (1, 1, 4), 5: (1, 1, 3), 4: (1, 1, 2), 3: (1, 1, 1)}
        v.gcs_parts = parts[total]


def history(r: random.Random, sc: Scenario, pmh_pool: List[str], med_pool: List[str],
            p_pmh=0.75, p_meds=0.6, p_allergy=0.5, p_code=0.12, p_nkda=0.6) -> None:
    roll = r.random()
    if roll < p_pmh:
        sc.pmh = some(r, pmh_pool, r.randint(1, 3)); sc.pmh_stated = True
    elif roll < p_pmh + 0.08:
        sc.pmh_stated = False                              # "no significant history"
    if sc.pmh_stated and chance(r, p_meds):
        sc.meds = some(r, med_pool, r.randint(1, 3)); sc.meds_stated = True
    elif chance(r, 0.05):
        sc.meds_stated = False
    if chance(r, p_allergy):
        if chance(r, p_nkda):
            sc.allergies = (pick(r, ["No known allergies.", "No known drug allergies.", "NKDA."]), ["no known", "nkda", "none"])
        else:
            al = some(r, ALLERGENS, r.randint(1, 2))
            sent = ("Allergic to " if len(al) == 1 else "Allergies to ") + join_list([a[0] for a in al]) + "."
            sc.allergies = (sent, [x for a in al for x in a[1]])
    if chance(r, p_code):
        sc.code = pick(r, CODE_STATUS)


def iv_line(r: random.Random, sc: Scenario, gauge_pool=(18, 20, 22), p=0.8, bore: bool = False) -> Optional[str]:
    if not chance(r, p):
        return None
    if bore:
        sc.interventions.append(Intervention("two large bore IVs", ["iv", "large bore"], [], 3))
        sc.terms += ["large bore", "IV"]
        return pick(r, ["Two large bore IVs established, running wide open.", "We have two large bore IVs, fluids running wide open."])
    g = pick(r, gauge_pool); site = pick(r, IV_SITES)
    sc.interventions.append(Intervention(f"{g}-gauge IV {site}", ["iv", str(g), "line", "saline lock"], [str(g)], 3))
    sc.terms += ["IV", "gauge"]
    art = "an" if str(g).startswith("8") or g == 18 else "a"
    return pick(r, [f"We've established {art} {g}-gauge IV {site}.", f"IV established, {g}-gauge {site}.",
                    f"{g}-gauge IV in the {site}, saline lock."])


def drug(sc: Scenario, name: str, dose: str, route: str, numbers: List[str], weight: int = 3) -> str:
    aliases = DRUGS[name][0]
    sc.interventions.append(Intervention(f"{dose} {name} {route}".strip(), aliases, numbers, weight))
    sc.terms.append(name)
    return f"{dose} {name} {route}".strip()


def eta_and_priority(r: random.Random, sc: Scenario, p_eta=0.85, prio_weights=(0.15, 0.45, 0.15, 0.25)) -> None:
    """prio_weights: P(1), P(2), P(3), P(not stated)."""
    if chance(r, p_eta):
        sc.eta = pick(r, [2, 3, 4, 5, 5, 6, 7, 8, 8, 10, 10, 12, 15, 20])
    x = r.random()
    sc.priority = 1 if x < prio_weights[0] else 2 if x < sum(prio_weights[:2]) else 3 if x < sum(prio_weights[:3]) else None


# ---- the families

def fam_stemi(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Male", "Female"]); age = r.randint(48, 84)
    v = pick_vitals(r, bp=(96, 176, 58, 102), hr=(54, 118), rr=(16, 24), spo2=(91, 98))
    v.rhythm = rhythm_for(r, v.hr)
    set_o2(r, v, pick(r, ["ra", "ra", "nc", "nc", None]))
    leads = pick(r, [("V1 through V4", "anterior"), ("II, III and aVF", "inferior"), ("V5, V6, I and aVL", "lateral"), ("V2 through V5", "anterior")])
    onset = pick(r, [f"Patient was at home when the pain started approximately {pick(r, [20, 30, 45, 60, 90])} minutes ago.",
                     f"Onset was around {pick(r, ['7', '8', '9', '10', '11'])} {pick(r, ['AM', 'PM'])} while {pick(r, ['watching TV', 'mowing the lawn', 'at work', 'walking the dog', 'eating dinner'])}.",
                     f"Pain started about {pick(r, ['an hour', 'two hours', 'half an hour'])} ago, {pick(r, ['substernal', 'left-sided'])}, radiating to the {pick(r, ['left arm', 'jaw', 'back', 'left shoulder'])}."])
    sc = Scenario("stemi", "STEMI", [], age, "years", sex, "chest pain", ["chest pain"], ["STEMI", "ST elevation", "chest pain"],
                  story=[onset], vitals=v)
    sc.exam = [pick(r, ["Patient is diaphoretic and pale.", "Patient is alert and oriented, diaphoretic.", "Skin is cool and diaphoretic.", "Patient is nauseous, no vomiting."]),
               f"12-lead shows ST elevation in leads {leads[0]}, consistent with {leads[1]} STEMI."]
    if chance(r, 0.5):
        sc.exam.append(f"Pain is {r.randint(5, 9)} out of 10.")
    sc.terms += ["12-lead", "ST elevation", "STEMI", "diaphoretic"]
    history(r, sc, ["hypertension", "type 2 diabetes", "CAD", "hyperlipidemia", "prior MI", "stents", "CHF"],
            ["metformin", "lisinopril", "atorvastatin", "metoprolol", "amlodipine", "Plavix", "aspirin 81"])
    parts = [p for p in [iv_line(r, sc)] if p]
    given = []
    if chance(r, 0.9):
        if chance(r, 0.8):
            given.append(drug(sc, "aspirin", "324 mg", pick(r, ["chewed", "PO", ""]), ["324"]))
        else:
            given.append(drug(sc, "aspirin", "81 mg times four", pick(r, ["chewed", ""]), []))
            sc.extra_fields.append({"path": "ems_interventions", "kind": "contains", "expect": ["81", "324"], "weight": 3, "note": "aspirin 81 mg times four"})
    if chance(r, 0.75):
        n = pick(r, ["times one", "times two", "times three"])
        given.append(drug(sc, "nitro", "0.4 mg", f"sublingual {n}", ["0.4"]))
        relief = pick(r, ["Some relief with the nitro.", "No relief with the nitro.", "Pain down to 4 out of 10 after the nitro."])
    if chance(r, 0.25):
        mcg = pick(r, [25, 50, 100])
        given.append(drug(sc, "fentanyl", f"{mcg} mcg", "IV", [str(mcg)]))
    if given:
        parts.append("Administered " + join_list(given) + ".")
        if any("nitro" in g for g in given):
            parts.append(relief)
    sc.treat += parts
    sc.request = pick(r, [("Requesting cath lab activation.", ["cath"]), ("Requesting a STEMI alert and cath lab activation.", ["cath", "stemi"]),
                          ("12-lead has been transmitted to your facility; requesting cath lab.", ["cath"])])
    sc.destination = "Cath Lab"
    eta_and_priority(r, sc, prio_weights=(0.45, 0.2, 0.0, 0.35))
    return sc


def fam_chest_pain(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Female"]); age = r.randint(30, 88)
    v = pick_vitals(r, bp=(118, 190, 70, 112), hr=(56, 160), rr=(16, 22), spo2=(94, 99))
    v.rhythm = rhythm_for(r, v.hr, afib=True)
    set_o2(r, v, pick(r, ["ra", "ra", "ra", "nc", None]))
    sc = Scenario("chest_pain", "MEDICAL", ["CARDIAC"], age, "years", sex, "chest pain", ["chest pain"], ["chest pain"],
                  story=[pick(r, [f"Patient was sitting on the couch around {pick(r, [6, 7, 8, 9, 10])} PM when {('he' if sex == 'Male' else 'she')} began to feel {pick(r, ['substernal chest pain', 'chest pressure', 'chest tightness'])} radiating to the {pick(r, ['back between the shoulder blades', 'left arm', 'jaw', 'neck'])}.",
                                   f"Patient woke up this morning with chest pain and {pick(r, ['shortness of breath', 'nausea', 'palpitations', 'upper back pain'])}.",
                                   f"Patient reports {pick(r, ['palpitations', 'a heaviness on the chest', 'chest discomfort'])} for the last {pick(r, ['two hours', 'three hours', 'hour'])}."])],
                  vitals=v)
    sc.exam = [pick(r, ["12-lead is unremarkable.", "12-lead was unremarkable, no ST changes.", "12-lead shows sinus tach with no ST elevation.", "12-lead has been sent to your facility."]),
               pick(r, ["Patient is alert and oriented times four.", f"Currently complaining of {r.randint(3, 9)} out of 10 chest pain.", "Lungs clear bilaterally."])]
    sc.terms += ["12-lead"]
    history(r, sc, ["hypertension", "hyperlipidemia", "diabetes", "AFib", "GERD", "anxiety", "CAD", "ESRD on dialysis"],
            ["metoprolol", "amlodipine", "atorvastatin", "Eliquis", "omeprazole", "metformin", "lisinopril"])
    parts = [p for p in [iv_line(r, sc, p=0.6)] if p]
    given = []
    if chance(r, 0.7):
        given.append(drug(sc, "aspirin", "324 mg", pick(r, ["", "chewed"]), ["324"]))
    if chance(r, 0.5):
        given.append(drug(sc, "nitro", pick(r, ["0.4 mg", "400 mcg"]), "sublingual", []))
        sc.interventions[-1].numbers = ["400"] if "400" in given[-1] else ["0.4"]
    if given:
        parts.append(pick(r, ["We administered ", "Patient has received "]) + join_list(given) + ".")
    if not given and not parts and chance(r, 0.5):
        parts.append("No interventions at this time."); sc.no_interventions = True
    sc.treat += parts
    if chance(r, 0.4):
        sc.request = ("Notification only, no orders needed.", ["notification", "none"])
    eta_and_priority(r, sc, prio_weights=(0.1, 0.5, 0.15, 0.25))
    return sc


def fam_stroke(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Female"]); age = r.randint(52, 92)
    v = pick_vitals(r, bp=(150, 214, 80, 118), hr=(58, 110), rr=(14, 22), spo2=(94, 99))
    v.rhythm = pick(r, ["", "", ", irregular", ", AFib"])
    set_o2(r, v, pick(r, ["ra", "ra", "nc", None]))
    v.bg = r.randint(78, 220)
    side = pick(r, ["left", "right"]); other = "right" if side == "left" else "left"
    lkw = pick(r, [f"{pick(r, [7, 8, 9, 10, 11])}:{pick(r, ['00', '15', '30', '45'])} this morning", f"about {pick(r, [30, 45, 60, 90])} minutes ago",
                   "last night at bedtime, found this way this morning", f"{pick(r, [1, 2, 3])} hours ago"])
    sc = Scenario("stroke", "STROKE", [], age, "years", sex, pick(r, ["possible stroke", "stroke symptoms", "stroke alert"]),
                  ["stroke", "cva", "droop", "weakness"], ["stroke", "CVA", "facial droop"],
                  story=[f"Last known well {lkw}.",
                         pick(r, [f"Family noticed {side}-sided facial droop and slurred speech.", f"Patient has {side} arm drift and {side} facial droop, speech is slurred.",
                                  f"Sudden onset {side}-sided weakness and {pick(r, ['aphasia', 'slurred speech', 'confusion'])}."])],
                  vitals=v)
    sc.exam = [pick(r, [f"Cincinnati is positive, {pick(r, ['two', 'three'])} out of three.", f"LAMS is {r.randint(2, 5)}.", "Cincinnati stroke scale positive for arm drift and speech."]),
               pick(r, ["Pupils equal and reactive.", "Patient is alert, oriented to person only.", f"Grips weaker on the {side}.", "No trauma noted."])]
    sc.terms += ["Cincinnati", "facial droop", "slurred speech", "stroke", "arm drift"]
    history(r, sc, ["hypertension", "AFib", "diabetes", "prior stroke", "hyperlipidemia", "CAD"],
            ["Eliquis", "Xarelto", "warfarin", "lisinopril", "metoprolol", "atorvastatin", "metformin"], p_code=0.15)
    parts = [p for p in [iv_line(r, sc, p=0.7)] if p]
    if chance(r, 0.4):
        parts.append("No medications given."); 
    sc.treat += parts
    sc.request = pick(r, [("Requesting a stroke alert.", ["stroke"]), ("Requesting stroke team activation.", ["stroke"]),
                          ("Calling a stroke alert, requesting CT on arrival.", ["stroke", "ct"])])
    if sc.vitals.bg is not None:
        sc.terms.append("blood glucose")
    eta_and_priority(r, sc, prio_weights=(0.5, 0.2, 0.0, 0.3))
    return sc


def fam_trauma(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Male", "Female"]); age = r.randint(16, 90)
    mech = pick(r, [("high-speed MVC with rollover, restrained driver", "mvc"), ("MVC, unrestrained passenger, significant front end damage", "mvc"),
                    (f"fall from approximately {pick(r, [8, 10, 12, 15, 20])} feet", "fall"), ("ground level fall on blood thinners", "fall"),
                    ("pedestrian struck by a vehicle at approximately 30 miles an hour", "ped"), ("motorcycle crash, helmeted", "mcc"),
                    ("GSW to the left thigh", "gsw"), ("stab wound to the right upper quadrant", "stab"), ("assault with a blunt object to the head", "assault")])
    mech_aliases = {"mvc": ["mvc", "motor vehicle", "rollover", "collision", "crash"], "fall": ["fall", "fell"], "ped": ["pedestrian", "struck"],
                    "mcc": ["motorcycle", "crash"], "gsw": ["gsw", "gunshot", "shot"], "stab": ["stab"], "assault": ["assault", "blunt"]}[mech[1]]
    severe = mech[1] in ("gsw", "stab", "ped") or chance(r, 0.4)
    v = pick_vitals(r, bp=(84, 118, 50, 72) if severe else (110, 150, 64, 92), hr=(104, 138) if severe else (72, 104), rr=(20, 30), spo2=(88, 99))
    v.rr_note = pick(r, ["", "", " and shallow", " and labored"])
    set_o2(r, v, pick(r, ["nrb", "nrb", "ra", "nc", None]))
    if chance(r, 0.85):
        set_gcs(r, v, pick(r, [15, 15, 14, 14, 13, 12, 9, 8, 7]), 0.45)
    sc = Scenario("trauma", "TRAUMA", [], age, "years", sex, mech[0], mech_aliases + ["trauma"],
                  [mech_aliases[0].upper() if mech[1] in ("mvc", "gsw") else mech_aliases[0]], story=[], vitals=v)
    loc = pick(r, ["Positive loss of consciousness at scene per bystanders.", "No loss of consciousness.", "Loss of consciousness unknown.", ""])
    if loc:
        sc.story.append(loc)
    injuries = {"mvc": ["a large scalp laceration with controlled bleeding", "deformity to the right forearm", "seatbelt sign across the lower abdomen", "chest wall tenderness", "left femur deformity"],
                "fall": ["a hematoma to the occiput", "right hip pain with shortening and rotation", "left wrist deformity", "a laceration to the forehead", "pelvic instability"],
                "ped": ["bilateral lower leg deformities", "abrasions to the back", "an open left tib-fib fracture", "a scalp laceration"],
                "mcc": ["road rash to the left side", "left clavicle deformity", "a right ankle deformity", "chest wall tenderness"],
                "gsw": ["a single wound to the left thigh with heavy bleeding, tourniquet applied", "no exit wound found"],
                "stab": ["a two inch wound to the right upper quadrant, bleeding controlled with pressure", "abdominal guarding"],
                "assault": ["a large hematoma to the left temple", "a laceration above the left eye", "epistaxis"]}[mech[1]]
    inj = some(r, injuries, r.randint(1, min(3, len(injuries))))
    sc.exam = ["Injuries include " + join_list(inj) + "."]
    if "tourniquet" in sc.exam[0]:
        sc.interventions.append(Intervention("tourniquet", ["tourniquet", "tq"], [], 3)); sc.terms.append("tourniquet")
    if chance(r, 0.7):
        sc.exam.append(pick(r, ["Patient is c-spine immobilized on a backboard.", "C-collar in place, spinal motion restriction.", "Patient extricated by fire, c-collar applied."]))
        sc.interventions.append(Intervention("c-collar / spinal immobilization", ["collar", "spine", "spinal", "backboard", "immobiliz"], [], 2))
        sc.terms += ["c-spine", "backboard"]
    sc.exam.append(pick(r, ["Pupils equal and reactive.", "Breath sounds bilateral but diminished on the right.", "Patient confused but following commands.",
                             "Abdomen soft, non-tender.", "Pelvis stable.", "Moving all extremities."]))
    sc.terms += [i.split(" with ")[0].split(" to ")[0].replace("a ", "").replace("an ", "") for i in inj]
    if chance(r, 0.35):
        sc.pmh_stated = None; sc.meds_stated = None
        sc.story.append(pick(r, ["No medical history available.", "Patient is a visitor from out of state, no history available, allergy status unknown."]))
    else:
        history(r, sc, ["hypertension", "diabetes", "AFib", "COPD", "seizure disorder", "alcohol use"],
                ["Eliquis", "warfarin", "metoprolol", "lisinopril", "metformin", "Keppra"], p_pmh=0.6, p_meds=0.5, p_allergy=0.45)
    parts = [p for p in [iv_line(r, sc, p=0.85, bore=chance(r, 0.5))] if p]
    given = []
    if chance(r, 0.5):
        given.append(drug(sc, "fentanyl", f"{pick(r, [50, 100])} mcg", "IV", []))
        sc.interventions[-1].numbers = [re.search(r"\d+", given[-1]).group()]
    if chance(r, 0.2):
        given.append(drug(sc, "TXA", "1 gram", "IV", ["1"]))
    if chance(r, 0.25):
        given.append(drug(sc, "ondansetron", "4 mg", "IV", ["4"]))
    if given:
        parts.append("Administered " + join_list(given) + ".")
    elif chance(r, 0.4):
        parts.append("No medications administered.")
    sc.treat += parts
    if severe or chance(r, 0.5):
        sc.request = pick(r, [("Requesting trauma activation.", ["trauma"]), ("Requesting trauma team activation.", ["trauma"]), ("Requesting a trauma alert.", ["trauma"])])
        sc.destination = "Trauma Bay"
    eta_and_priority(r, sc, prio_weights=(0.45 if severe else 0.15, 0.3, 0.1, 0.25 if severe else 0.45))
    return sc


def fam_sepsis(r: random.Random) -> Scenario:
    sex = pick(r, ["Female", "Female", "Male"]); age = r.randint(64, 97)
    v = pick_vitals(r, bp=(78, 108, 42, 66), hr=(100, 136), rr=(22, 32), spo2=(88, 95))
    set_o2(r, v, pick(r, ["nc", "nc", "nrb", "ra", None]))
    v.temp = pick(r, ["101.8", "102.4", "103.1", "100.9", "104.0", "96.1"])
    if chance(r, 0.5):
        v.bg = r.randint(120, 320)
    set_gcs(r, v, pick(r, [14, 13, 12, 11]), 0.3) if chance(r, 0.5) else None
    source = pick(r, [("urinary", "foley catheter in place, urine appears cloudy"), ("respiratory", "coarse rhonchi in the bilateral bases, productive cough"),
                      ("skin", "a stage 4 sacral decubitus with foul drainage"), ("unknown", "no obvious source")])
    sc = Scenario("sepsis", "SEPSIS", [], age, "years", sex, pick(r, ["altered mental status and fever", "possible sepsis", "fever and hypotension"]),
                  ["sepsis", "altered", "fever", "infection"], ["sepsis", "fever", "altered mental status"],
                  story=[pick(r, [f"Patient is from {pick(r, ['Sunrise Nursing Facility', 'a skilled nursing facility', 'an assisted living facility', 'home with hospice'])}.",
                                  "Per nursing staff, patient was at baseline until this morning.", "Family reports patient has been more confused for two days."]),
                         pick(r, ["Patient became increasingly confused and febrile.", "Staff report decreased oral intake, lethargy and a temperature.", "Patient was found lethargic and hypotensive on our arrival."])],
                  vitals=v)
    sc.exam = [pick(r, ["Skin is warm and flushed.", "Skin is hot, dry and mottled at the knees.", "Patient oriented to name only."]),
               f"Suspected {source[0]} source, " + source[1] + "." if source[0] != "unknown" else "No obvious source."]
    sc.terms += ["sepsis", "febrile", "hypotensive", "rhonchi" if "rhonchi" in source[1] else "foley" if "foley" in source[1] else "lethargic"]
    history(r, sc, ["Alzheimer's dementia", "CHF", "atrial fibrillation", "chronic kidney disease", "type 2 diabetes", "COPD", "hypertension"],
            ["metoprolol", "furosemide", "donepezil", "Eliquis", "insulin", "levothyroxine", "lisinopril"], p_pmh=0.85, p_meds=0.7, p_allergy=0.6, p_code=0.45)
    parts = [p for p in [iv_line(r, sc, p=0.85)] if p]
    if sc.interventions and chance(r, 0.85):
        ml = pick(r, [500, 500, 1000])
        parts.append(pick(r, [f"{ml} mL normal saline bolus initiated.", f"Running a {ml} mL bolus.", f"{ml} mL of normal saline in so far."]))
        sc.interventions.append(Intervention(f"{ml} mL bolus", ["bolus", "saline", "fluid", "ns"], [str(ml)], 3)); sc.terms.append("bolus")
    if chance(r, 0.25):
        sc.treat.append(f"Lactate is {pick(r, ['3.2', '4.1', '5.6', '2.8'])}."); sc.terms.append("lactate")
    sc.treat += parts
    sc.request = pick(r, [("Requesting a sepsis alert.", ["sepsis"]), ("We're calling a sepsis alert.", ["sepsis"]), None, None])
    eta_and_priority(r, sc, prio_weights=(0.35, 0.4, 0.0, 0.25))
    return sc


def fam_arrest(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Male", "Female"]); age = r.randint(38, 88)
    rosc = chance(r, 0.55)
    v = Vitals()
    if rosc:
        v = pick_vitals(r, bp=(78, 132, 44, 80), hr=(48, 128), rr=(0, 0), spo2=(86, 97))
        v.rr = None; set_o2(r, v, pick(r, ["bvm", "bvm", None]))
    else:
        set_o2(r, v, "bvm") if chance(r, 0.5) else None
    witnessed = pick(r, ["witnessed", "unwitnessed"])
    sc = Scenario("arrest", "CARDIAC_ARREST", [], age, "years", sex, pick(r, ["cardiac arrest", "cardiac arrest with ROSC" if rosc else "CPR in progress"]),
                  ["arrest", "cpr", "rosc"], ["cardiac arrest", "CPR", "ROSC"],
                  story=[f"{witnessed.capitalize()} arrest {pick(r, ['at home', 'at a restaurant', 'at a gym', 'in a parking lot'])}, {pick(r, ['bystander CPR in progress on our arrival', 'no bystander CPR', 'CPR started by family'])}.",
                         pick(r, ["Initial rhythm V-fib.", "Initial rhythm asystole.", "Initial rhythm PEA.", "Initial rhythm V-fib, now PEA."])],
                  vitals=v)
    shocks = r.randint(0, 4) if chance(r, 0.7) else 0
    epi = r.randint(1, 4)
    given = []
    if shocks:
        sc.interventions.append(Intervention(f"defibrillated times {shocks}", ["defib", "shock"], [str(shocks)], 3)); sc.terms.append("defibrillat")
        given.append(f"defibrillated times {shocks}")
    given.append(drug(sc, "epi", f"{epi} rounds of" if epi > 1 else "one round of", "", [str(epi)] if epi > 1 else []))
    sc.interventions[-1].text = f"epinephrine {epi} rounds"
    if shocks >= 2 and chance(r, 0.7):
        given.append(drug(sc, "amiodarone", "300 mg", "IV", ["300"]))
    airway = pick(r, [("Airway is an iGel.", ["igel", "i-gel", "supraglottic"]), ("Patient is intubated, tube confirmed with waveform capnography.", ["intubat", "tube"]),
                      ("Airway managed with BVM and an OPA.", ["bvm", "opa", "airway"])])
    sc.treat.append(airway[0]); sc.interventions.append(Intervention(airway[0], airway[1], [], 2)); sc.terms.append("airway")
    sc.treat.append(pick(r, ["CPR has been ongoing for", "Total down time approximately"]) + f" {pick(r, [8, 12, 15, 20, 25, 30])} minutes." )
    sc.treat.append("We have " + join_list(given) + ".")
    iv = pick(r, [("IO established right tibia.", ["io", "tibia"]), ("18-gauge IV left AC.", ["iv", "18"]), (None, [])])
    if iv[0]:
        sc.treat.append(iv[0]); sc.interventions.append(Intervention(iv[0], iv[1], ["18"] if "18" in iv[0] else [], 2))
    if rosc:
        sc.treat.append(pick(r, ["ROSC achieved approximately", "We got ROSC about"]) + f" {pick(r, [3, 5, 8, 10])} minutes ago.")
        sc.exam = [pick(r, ["Patient remains unresponsive, GCS 3.", "Patient is unresponsive, GCS 3, pupils sluggish."])]
        set_gcs(r, sc.vitals, 3, 0.0)
    else:
        sc.exam = ["CPR in progress, mechanical CPR device in place." if chance(r, 0.5) else "CPR continues en route."]
    history(r, sc, ["CAD", "CHF", "diabetes", "hypertension", "ESRD on dialysis", "COPD"], ["metoprolol", "Lasix", "insulin", "Plavix", "lisinopril"],
            p_pmh=0.45, p_meds=0.3, p_allergy=0.3, p_code=0.2)
    sc.request = pick(r, [("Requesting the resuscitation bay.", ["resus", "code"]), ("Please have the team ready.", ["team", "resus", "code"]), None])
    sc.destination = "Resuscitation Bay"
    eta_and_priority(r, sc, prio_weights=(0.7, 0.0, 0.0, 0.3))
    return sc


def fam_pediatric(r: random.Random) -> Scenario:
    months = chance(r, 0.45)
    age = r.randint(2, 23) if months else r.randint(2, 11)
    sex = pick(r, ["Male", "Female"])
    kind = pick(r, ["resp", "resp", "seizure", "allergic", "dehydration"])
    v = Vitals()
    if kind == "resp":
        v = pick_vitals(r, bp=(84, 110, 50, 70), hr=(120, 178), rr=(32, 60), spo2=(86, 95))
        set_o2(r, v, pick(r, ["blowby", "blowby", "nc", "nrb"]))
        cc, aliases = pick(r, ["respiratory distress", "trouble breathing", "respiratory distress and trouble breathing"]), ["respiratory", "breathing"]
        sc = Scenario("pediatric", "PEDIATRIC", ["MEDICAL"], age, "months" if months else "years", sex, cc, aliases, ["pediatric", "respiratory distress"],
                      story=[pick(r, ["Mom called because the patient was working harder to breathe with a lot of drool and a cough.", "Patient has had a cough and fever for two days, worse tonight.", "Audible wheezing, mom states a history of asthma." if not months else "Barky cough since this evening, retractions noted."])],
                      vitals=v)
        sc.exam = [pick(r, ["Retractions and nasal flaring noted.", "Wheezes bilaterally.", "Stridor at rest.", "Patient is slightly altered but consolable."])]
        if chance(r, 0.6):
            given = drug(sc, "albuterol", "2.5 mg", "nebulized", ["2.5"])
            sc.treat.append(f"We administered {given}" + (", with some improvement." if chance(r, 0.6) else "."))
        sc.terms += ["retractions", "wheez", "nebulized"]
    elif kind == "seizure":
        v = pick_vitals(r, bp=(84, 112, 50, 72), hr=(100, 160), rr=(20, 36), spo2=(92, 99)); set_o2(r, v, pick(r, ["blowby", "ra", None])); v.temp = pick(r, ["102.8", "103.5", "101.9"]); v.bg = r.randint(80, 140)
        sc = Scenario("pediatric", "PEDIATRIC", ["MEDICAL", "NEURO"], age, "months" if months else "years", sex, "febrile seizure", ["seizure", "febrile"], ["pediatric", "seizure"],
                      story=[f"Generalized seizure lasting approximately {pick(r, [1, 2, 3, 5])} minutes, witnessed by parents, resolved on our arrival.", pick(r, ["Patient is postictal but improving.", "Patient is crying and consolable now."])], vitals=v)
        sc.exam = ["No history of seizures." if chance(r, 0.6) else "One prior febrile seizure."]
        if chance(r, 0.3):
            sc.treat.append("We administered " + drug(sc, "midazolam", f"{pick(r, ['2 mg', '2.5 mg'])}", "intranasal", []) + ".")
            sc.interventions[-1].numbers = [re.search(r"[\d.]+", sc.interventions[-1].text).group()]
        sc.terms += ["seizure", "postictal", "febrile"]
    elif kind == "allergic":
        v = pick_vitals(r, bp=(80, 108, 46, 68), hr=(110, 160), rr=(24, 40), spo2=(90, 98)); set_o2(r, v, pick(r, ["nrb", "ra", "blowby"]))
        sc = Scenario("pediatric", "PEDIATRIC", ["MEDICAL"], age, "months" if months else "years", sex, "allergic reaction", ["allergic", "anaphyla"], ["pediatric", "allergic reaction", "anaphylaxis"],
                      story=[f"Hives and facial swelling after eating {pick(r, ['peanuts', 'shellfish', 'a cookie', 'unknown food'])} about {pick(r, [15, 20, 30])} minutes ago.", pick(r, ["Some wheezing on arrival, now improved.", "No airway swelling noted."])], vitals=v)
        sc.exam = ["Hives to the face and trunk."]
        sc.treat.append("We administered " + join_list([drug(sc, "epi", "0.15 mg", "IM", ["0.15"]), drug(sc, "Benadryl", f"{pick(r, [12.5, 25])} mg", "IV", [])]) + ".")
        sc.interventions[-1].numbers = [re.search(r"[\d.]+", sc.interventions[-1].text).group()]
        sc.allergies = (f"Known allergy to {pick(r, ['peanuts', 'tree nuts', 'shellfish'])}.", ["peanut", "nut", "shellfish"]) if chance(r, 0.5) else None
        sc.terms += ["hives", "anaphylaxis", "IM"]
    else:
        v = pick_vitals(r, bp=(80, 104, 46, 66), hr=(130, 180), rr=(28, 44), spo2=(94, 99)); set_o2(r, v, pick(r, ["ra", None])); v.bg = r.randint(60, 110)
        sc = Scenario("pediatric", "PEDIATRIC", ["MEDICAL"], age, "months" if months else "years", sex, "vomiting and diarrhea", ["vomit", "diarrhea", "dehydrat"], ["pediatric"],
                      story=["Three days of vomiting and diarrhea, decreased wet diapers, lethargic per mom." if months else "Two days of vomiting and diarrhea, not keeping fluids down."], vitals=v)
        sc.exam = ["Delayed cap refill, dry mucous membranes."]
        if chance(r, 0.5):
            sc.treat.append("22-gauge IV established, 20 mL per kilo bolus running.")
            sc.interventions.append(Intervention("22-gauge IV, 20 mL/kg bolus", ["iv", "22", "bolus", "fluid"], ["22", "20"], 3))
        sc.terms += ["cap refill", "dehydration"]
    if kind != "allergic":
        history(r, sc, ["asthma", "seizure disorder"] if not months else ["prematurity"], ["albuterol inhaler", "Keppra"], p_pmh=0.3, p_meds=0.25, p_allergy=0.5, p_code=0.0, p_nkda=0.8)
    if "prematurity" in sc.pmh:
        PMH.setdefault("prematurity", ["premat"])
    sc.request = pick(r, [("Requesting a pediatric physician online for orders.", ["physician", "pediatric", "peds"]), None, None])
    eta_and_priority(r, sc, prio_weights=(0.25, 0.45, 0.05, 0.25))
    return sc


def fam_obstetric(r: random.Random) -> Scenario:
    age = r.randint(19, 41); weeks = r.randint(24, 41)
    kind = pick(r, ["labor", "labor", "bleeding", "imminent", "postpartum", "eclampsia"])
    v = pick_vitals(r, bp=(104, 142, 60, 88), hr=(84, 124), rr=(16, 24), spo2=(96, 100))
    if kind == "eclampsia":
        v.bp = f"{r.randint(160, 196)}/{r.randint(100, 118)}"
    if kind in ("bleeding", "postpartum"):
        v.bp = f"{r.randint(84, 104)}/{r.randint(48, 64)}"; v.hr = r.randint(110, 136)
    set_o2(r, v, pick(r, ["ra", "ra", "nc", None]))
    g, p = r.randint(1, 5), 0
    p = r.randint(0, g - 1)
    cc = {"labor": "active labor", "bleeding": "vaginal bleeding", "imminent": "imminent delivery", "postpartum": "postpartum hemorrhage", "eclampsia": "possible pre-eclampsia"}[kind]
    sc = Scenario("obstetric", "OBSTETRIC", [], age, "years", "Female", f"{weeks} weeks pregnant, {cc}", [cc.split()[-1], "pregnan", "labor", "obstetric", "ob"], ["pregnant", "labor", "delivery"],
                  story=[f"G{g} P{p}, {weeks} weeks by her dates." if chance(r, 0.7) else f"Patient is {weeks} weeks along, prenatal care at {pick(r, HOSPITALS)}."], vitals=v)
    if kind == "labor":
        sc.story.append(f"Contractions every {pick(r, [2, 3, 4, 5])} minutes lasting about a minute, water {pick(r, ['broke an hour ago, clear fluid', 'intact', 'broke this morning'])}.")
        sc.exam = ["No crowning, no urge to push." if chance(r, 0.6) else "Patient has the urge to push, no crowning on our exam."]
        sc.terms += ["contractions"]
    elif kind == "bleeding":
        sc.story.append(f"Painless bright red vaginal bleeding starting about {pick(r, [30, 45, 60])} minutes ago, soaked {pick(r, ['two', 'three'])} pads.")
        sc.exam = ["Patient is pale and anxious, fetal movement reported present."]; sc.terms += ["vaginal bleeding"]
    elif kind == "imminent":
        sc.story.append("Crowning noted on our arrival, contractions every two minutes.")
        sc.exam = ["Preparing for field delivery; OB kit open."]; sc.terms += ["crowning"]
    elif kind == "postpartum":
        sc.story.append(f"Delivered a {pick(r, ['viable', 'healthy'])} {pick(r, ['male', 'female'])} infant at the residence approximately {pick(r, [10, 15, 20])} minutes ago, now with heavy bleeding, placenta {pick(r, ['delivered', 'not yet delivered'])}.")
        sc.exam = ["Fundal massage in progress, bleeding estimated over 500 mL.", "Infant is pink and crying, with the mother."]
        sc.interventions.append(Intervention("fundal massage", ["fundal", "massage"], [], 2)); sc.terms += ["fundal massage", "placenta"]
    else:
        sc.story.append("Severe headache, visual changes and swelling to the hands and face for two days.")
        sc.exam = ["Hyperreflexic, no seizure activity so far."]
        if chance(r, 0.4):
            sc.treat.append("Administered " + drug(sc, "magnesium", "4 grams", "IV", ["4"]) + ".")
        sc.terms += ["pre-eclampsia", "hyperreflexic"]
    parts = [p_ for p_ in [iv_line(r, sc, p=0.7)] if p_]
    if kind in ("bleeding", "postpartum") and sc.interventions:
        parts.append("500 mL bolus running."); sc.interventions.append(Intervention("500 mL bolus", ["bolus", "fluid", "saline"], ["500"], 3))
    sc.treat += parts
    history(r, sc, ["gestational diabetes", "hypertension", "asthma"], ["prenatal vitamins", "insulin", "albuterol inhaler"], p_pmh=0.4, p_meds=0.4, p_allergy=0.55, p_code=0.0, p_nkda=0.75)
    PMH.setdefault("gestational diabetes", ["gestational", "diabetes"])
    sc.request = pick(r, [("Requesting OB to meet us at the door.", ["ob", "obstetric", "labor"]), ("Requesting labor and delivery be notified.", ["labor", "l&d", "ob"]), None])
    eta_and_priority(r, sc, prio_weights=(0.3, 0.4, 0.05, 0.25))
    return sc


def fam_endocrine(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Female"]); age = r.randint(14, 82)
    kind = pick(r, ["dka", "dka", "hypo", "hypo", "hyper"])
    v = pick_vitals(r, bp=(92, 140, 54, 86), hr=(80, 130), rr=(18, 32), spo2=(94, 99))
    set_o2(r, v, pick(r, ["ra", "ra", None]))
    if kind == "dka":
        v.bg = pick(r, ["high", "high", 486, 560, 612])
        sc = Scenario("endocrine", "ENDOCRINE", ["MEDICAL"], age, "years", sex, pick(r, ["possible DKA", "hyperglycemia, possible DKA"]), ["dka", "ketoacidosis", "hyperglyc"], ["DKA", "hyperglycemia"],
                      story=[pick(r, ["Two days of nausea, vomiting and increased urination.", "Patient ran out of insulin three days ago.", "Found by family lethargic with a fruity odor on the breath."]) ,
                             pick(r, ["Kussmaul respirations noted.", "Patient is alert but lethargic, skin dry.", "Mucous membranes dry, poor skin turgor."])], vitals=v)
        sc.exam = []
        sc.terms += ["DKA", "Kussmaul", "insulin"]
        if chance(r, 0.6):
            sc.treat.append("18-gauge IV established, 1000 mL normal saline bolus running.")
            sc.interventions.append(Intervention("18-gauge IV, 1000 mL bolus", ["iv", "18", "bolus", "saline", "fluid"], ["18", "1000"], 3))
        history(r, sc, ["type 1 diabetes", "type 2 diabetes", "hypertension", "hypothyroidism"], ["insulin", "Lantus", "metformin", "levothyroxine"], p_pmh=0.85, p_meds=0.75, p_allergy=0.5)
    elif kind == "hypo":
        pre = r.randint(28, 52); post = r.randint(98, 170)
        v.bg = post
        sc = Scenario("endocrine", "ENDOCRINE", ["MEDICAL"], age, "years", sex, pick(r, ["hypoglycemia", "diabetic emergency, low blood sugar"]), ["hypoglyc", "low blood sugar", "glucose", "diabetic"], ["hypoglycemia"],
                      story=[f"Family found the patient {pick(r, ['confused and diaphoretic', 'unresponsive', 'combative'])}, initial blood glucose {pre}."], vitals=v)
        given = pick(r, [("D10", "25 grams", "IV", ["25"]), ("D10", "125 mL", "IV", ["125"]), ("glucagon", "1 mg", "IM", ["1"]), ("oral glucose", "15 grams", "PO", ["15"])])
        sc.treat.append(f"We administered {drug(sc, *given)}, repeat glucose is {post} and the patient is now alert and oriented.")
        sc.exam = []
        sc.terms += ["hypoglycemia", "D10" if given[0] == "D10" else given[0]]
        if given[0] in ("D10",) and chance(r, 0.8):
            sc.interventions.append(Intervention("IV", ["iv"], [], 1))
        history(r, sc, ["type 1 diabetes", "type 2 diabetes", "CKD", "hypertension"], ["insulin", "metformin", "Lantus", "lisinopril"], p_pmh=0.8, p_meds=0.7, p_allergy=0.5)
    else:
        v.bg = r.randint(340, 480)
        sc = Scenario("endocrine", "ENDOCRINE", ["MEDICAL"], age, "years", sex, "hyperglycemia", ["hyperglyc", "glucose", "sugar"], ["hyperglycemia"],
                      story=["Patient's home meter read high; complaining of thirst and weakness for a week.", "Patient alert and oriented, no vomiting."], vitals=v)
        sc.exam = []
        sc.terms += ["hyperglycemia"]
        if chance(r, 0.5):
            sc.treat.append(iv_line(r, sc, p=1.0))
        history(r, sc, ["type 2 diabetes", "hypertension", "hyperlipidemia"], ["metformin", "insulin", "atorvastatin"], p_pmh=0.85, p_meds=0.7)
    sc.terms.append("blood glucose")
    eta_and_priority(r, sc, prio_weights=(0.15, 0.45, 0.1, 0.3))
    return sc


def fam_medical(r: random.Random) -> Scenario:
    kind = pick(r, ["seizure", "copd", "overdose", "allergic", "syncope", "abdominal", "gi_bleed", "afib", "weakness"])
    sex = pick(r, ["Male", "Female"]); age = r.randint(18, 94)
    v = pick_vitals(r)
    set_o2(r, v, pick(r, ["ra", "ra", "nc", None]))
    if kind == "seizure":
        v.hr = r.randint(96, 130); set_gcs(r, v, pick(r, [14, 13, 12, 15]), 0.3) if chance(r, 0.6) else None
        v.bg = r.randint(80, 160) if chance(r, 0.6) else None
        sc = Scenario("medical", "MEDICAL", ["NEURO", "NEUROLOGICAL", "SEIZURE"], age, "years", sex, "seizure", ["seizure"], ["seizure"],
                      story=[pick(r, [f"Witnessed generalized seizure lasting approximately {pick(r, [1, 2, 3, 4])} minutes, resolved prior to our arrival.", "Found down after a seizure per bystanders, incontinent of urine.", "Two seizures in the last hour per family, the second one we witnessed."])], vitals=v)
        sc.exam = [pick(r, ["Patient is postictal, confused but protecting the airway.", "Patient now alert and oriented times three.", "Tongue laceration noted, no other trauma."])]
        if chance(r, 0.45):
            d = drug(sc, "Versed", f"{pick(r, [5, 5, 2.5, 10])} mg", pick(r, ["IM", "IV", "intranasal"]), [])
            sc.interventions[-1].numbers = [re.search(r"[\d.]+", d).group()]
            sc.treat.append(f"We administered {d} and the seizure stopped.")
        sc.terms += ["seizure", "postictal"]
        history(r, sc, ["seizure disorder", "epilepsy", "alcohol use", "diabetes", "hypertension"], ["Keppra", "gabapentin", "metformin"], p_pmh=0.7, p_meds=0.6)
    elif kind == "copd":
        age = max(age, 50); sc_v = v; sc_v.rr = r.randint(24, 36); sc_v.spo2 = r.randint(82, 92); sc_v.hr = r.randint(96, 128)
        set_o2(r, sc_v, pick(r, ["cpap", "nc", "nrb", "nc"]))
        sc = Scenario("medical", "MEDICAL", ["RESPIRATORY"], age, "years", sex, pick(r, ["shortness of breath", "respiratory distress", "COPD exacerbation"]), ["breath", "respiratory", "copd", "dyspnea"], ["shortness of breath", "COPD"],
                      story=[pick(r, ["Increasing shortness of breath for two days, worse tonight, unable to speak full sentences.", "Patient ran out of home oxygen; found tripoding, speaking two to three word sentences."])], vitals=sc_v)
        sc.exam = [pick(r, ["Diminished breath sounds with expiratory wheezes throughout.", "Wheezes bilaterally, accessory muscle use.", "Rales in the bases, pedal edema, patient has a history of CHF as well."])]
        given = []
        if chance(r, 0.8):
            given.append(drug(sc, "albuterol", "2.5 mg", "", ["2.5"])); given.append(drug(sc, "ipratropium", "0.5 mg", "nebulized", ["0.5"]))
        if chance(r, 0.4):
            given.append(drug(sc, "Solu-Medrol", "125 mg", "IV", ["125"]))
        if given:
            sc.treat.append("We've given " + join_list(given) + (" with some improvement." if chance(r, 0.6) else "."))
        if "CPAP" in (sc_v.o2 or ""):
            sc.interventions.append(Intervention("CPAP", ["cpap"], [], 3))
        sc.terms += ["wheez", "COPD", "CPAP", "nebulized"]
        history(r, sc, ["COPD", "CHF", "hypertension", "asthma"], ["Symbicort", "albuterol inhaler", "Lasix", "lisinopril"], p_pmh=0.85, p_meds=0.6)
    elif kind == "overdose":
        age = r.randint(19, 62); v.rr = r.randint(6, 12); v.spo2 = r.randint(80, 92); v.hr = r.randint(50, 90)
        set_o2(r, v, pick(r, ["bvm", "nrb", "nc"])); set_gcs(r, v, pick(r, [3, 6, 8, 10]), 0.2) if chance(r, 0.6) else None
        sc = Scenario("medical", "MEDICAL", ["OVERDOSE", "TOXICOLOGY", "TOX"], age, "years", sex, pick(r, ["suspected opioid overdose", "overdose", "unresponsive, suspected overdose"]), ["overdose", "opioid", "unresponsive"], ["overdose"],
                      story=[pick(r, ["Found unresponsive in a bathroom with drug paraphernalia nearby, pinpoint pupils.", "Friends state the patient used fentanyl about 30 minutes ago.", "Found unresponsive by police, agonal respirations."])], vitals=v)
        n = pick(r, ["2 mg", "4 mg", "0.4 mg"]); route = "intranasal" if n != "0.4 mg" else "IV"
        d = drug(sc, "Narcan", n, route, [n.split()[0]])
        resp = pick(r, ["with improvement in respirations, patient now responsive to voice.", "times two with minimal response.", "and the patient woke up combative."])
        sc.treat.append(f"We administered {d} {resp}")
        sc.exam = []
        sc.terms += ["Narcan", "pinpoint pupils", "overdose"]
        history(r, sc, ["opioid use", "depression", "hepatitis C"], ["Suboxone", "sertraline"], p_pmh=0.5, p_meds=0.3, p_allergy=0.3)
        PMH.setdefault("hepatitis C", ["hepatitis", "hep c"]); HOME_MEDS.setdefault("Suboxone", ["suboxone", "buprenorphine"])
    elif kind == "allergic":
        v.hr = r.randint(100, 136); v.bp = f"{r.randint(82, 110)}/{r.randint(48, 68)}"; set_o2(r, v, pick(r, ["nrb", "ra", "nc"]))
        sc = Scenario("medical", "MEDICAL", ["ALLERGIC", "ANAPHYLAXIS"], age, "years", sex, pick(r, ["allergic reaction", "anaphylaxis"]), ["allergic", "anaphyla"], ["allergic reaction", "anaphylaxis"],
                      story=[f"Hives, lip swelling and {pick(r, ['throat tightness', 'wheezing', 'difficulty swallowing'])} after {pick(r, ['a bee sting', 'eating shellfish', 'taking a new antibiotic', 'eating at a restaurant'])} approximately {pick(r, [15, 20, 30, 45])} minutes ago."], vitals=v)
        sc.treat.append("We administered " + join_list([drug(sc, "epi", "0.3 mg", "IM", ["0.3"]), drug(sc, "Benadryl", f"{pick(r, [25, 50])} mg", "IV", [])]) + pick(r, [", swelling improving.", " with good effect.", "."]))
        sc.interventions[-1].numbers = [re.search(r"\d+", sc.interventions[-1].text).group()]
        sc.exam = [pick(r, ["Lungs with scattered wheezes.", "Airway patent, voice is hoarse.", "Hives to the trunk and arms."])]
        sc.terms += ["hives", "anaphylaxis", "IM"]
        history(r, sc, ["asthma", "hypertension"], ["albuterol inhaler", "lisinopril"], p_pmh=0.4, p_meds=0.3, p_allergy=0.7, p_nkda=0.2)
    elif kind == "syncope":
        v.hr = pick(r, [r.randint(38, 52), r.randint(60, 100), r.randint(120, 160)]); v.rhythm = pick(r, ["", ", sinus brady", ", irregular", ", third degree block" if v.hr < 55 else ""])
        v.bg = r.randint(80, 170) if chance(r, 0.5) else None
        sc = Scenario("medical", "MEDICAL", ["CARDIAC", "SYNCOPE"], age, "years", sex, "syncopal episode", ["syncop", "passed out", "fainted"], ["syncope"],
                      story=[pick(r, ["Patient passed out while standing in church, caught by family, down less than a minute.", "Syncopal episode on the toilet, no trauma, now alert.", "Patient became dizzy and passed out after standing up, witnessed by spouse."])], vitals=v)
        sc.exam = [pick(r, ["Patient is alert and oriented times four, denies chest pain.", "12-lead shows sinus bradycardia, no ST changes.", "No injuries from the fall."])]
        if chance(r, 0.6):
            sc.treat.append(iv_line(r, sc, p=1.0))
        sc.terms += ["syncope", "bradycardia" if v.hr < 55 else "12-lead"]
        history(r, sc, ["hypertension", "AFib", "CAD", "pacemaker", "diabetes"], ["metoprolol", "Eliquis", "amlodipine", "lisinopril"], p_pmh=0.7, p_meds=0.6)
    elif kind == "abdominal":
        v.hr = r.randint(80, 120)
        side = pick(r, ["right lower quadrant", "right upper quadrant", "epigastric", "left lower quadrant", "diffuse"])
        sc = Scenario("medical", "MEDICAL", ["GI", "ABDOMINAL"], age, "years", sex, "abdominal pain", ["abdominal", "abd pain", "belly"], ["abdominal pain"],
                      story=[f"{side.capitalize()} abdominal pain for {pick(r, ['twelve hours', 'two days', 'since this morning'])}, {pick(r, ['with nausea and vomiting', 'with fever', 'no vomiting', 'with two episodes of vomiting'])}."], vitals=v)
        sc.exam = [pick(r, [f"Tender to palpation {side} with guarding.", "Abdomen distended and tender.", f"Pain is {r.randint(5, 10)} out of 10."])]
        given = []
        if chance(r, 0.5):
            given.append(drug(sc, "Zofran", "4 mg", "IV", ["4"]))
        if chance(r, 0.35):
            given.append(drug(sc, "fentanyl", "50 mcg", "IV", ["50"]))
        if given:
            sc.treat.append(iv_line(r, sc, p=1.0)); sc.treat.append("Administered " + join_list(given) + ".")
        sc.terms += ["guarding", side]
        history(r, sc, ["GERD", "hypertension", "diabetes", "gallstones", "appendectomy"], ["omeprazole", "metformin", "lisinopril"], p_pmh=0.6, p_meds=0.5)
        PMH.setdefault("gallstones", ["gallstone", "cholelith"]); PMH.setdefault("appendectomy", ["appendectomy", "appendix"])
    elif kind == "gi_bleed":
        v.bp = f"{r.randint(84, 112)}/{r.randint(46, 68)}"; v.hr = r.randint(104, 134)
        sc = Scenario("medical", "MEDICAL", ["GI", "GI_BLEED", "HEMORRHAGE"], age, "years", sex, pick(r, ["GI bleed", "rectal bleeding", "vomiting blood"]), ["bleed", "blood", "melena", "hemateme"], ["GI bleed"],
                      story=[pick(r, ["Three episodes of dark tarry stools since yesterday, dizzy on standing.", "Vomited approximately a cup of bright red blood this morning.", "Large amount of bright red blood per rectum, patient on blood thinners."])], vitals=v)
        sc.exam = ["Patient is pale and diaphoretic, skin cool."]
        sc.treat.append(iv_line(r, sc, p=1.0, bore=chance(r, 0.5)))
        if chance(r, 0.6):
            sc.treat.append("500 mL bolus running."); sc.interventions.append(Intervention("500 mL bolus", ["bolus", "fluid", "saline"], ["500"], 3))
        sc.terms += ["GI bleed", "melena", "diaphoretic"]
        history(r, sc, ["AFib", "alcohol use", "GERD", "CAD", "hypertension"], ["Eliquis", "warfarin", "Xarelto", "Plavix", "omeprazole"], p_pmh=0.7, p_meds=0.7)
    elif kind == "afib":
        v.hr = r.randint(140, 178); v.rhythm = ", AFib with RVR"
        sc = Scenario("medical", "MEDICAL", ["CARDIAC"], age, "years", sex, "AFib with RVR", ["afib", "a-fib", "atrial fib", "rvr", "palpitation"], ["AFib", "RVR"],
                      story=[pick(r, ["Palpitations and lightheadedness since waking, patient has a history of AFib.", "Patient felt her heart racing while at work, no chest pain."])], vitals=v)
        sc.exam = [pick(r, ["12-lead shows AFib with RVR, no ST changes.", "Patient alert and oriented, mildly short of breath."])]
        if chance(r, 0.5):
            sc.treat.append(iv_line(r, sc, p=1.0))
            if chance(r, 0.5):
                sc.treat.append("Administered " + drug(sc, "diltiazem", f"{pick(r, [10, 15, 20])} mg", "IV", []) + " with the rate down to the 110s.")
                sc.interventions[-1].numbers = [re.search(r"\d+", sc.interventions[-1].text).group()]
        sc.terms += ["AFib", "RVR", "12-lead"]
        history(r, sc, ["AFib", "hypertension", "hypothyroidism", "CHF"], ["Eliquis", "metoprolol", "Xarelto", "levothyroxine", "diltiazem"], p_pmh=0.8, p_meds=0.7)
        HOME_MEDS.setdefault("diltiazem", ["diltiazem", "cardizem"])
    else:
        age = max(age, 60)
        sc = Scenario("medical", "MEDICAL", [], age, "years", sex, "generalized weakness", ["weakness", "weak"], [],
                      story=[pick(r, ["Generalized weakness for three days, poor oral intake, unable to stand without help.", "Family found the patient on the floor, unable to get up, unknown down time, no injuries."])], vitals=v)
        sc.exam = [pick(r, ["No focal deficits, Cincinnati negative.", "Patient alert and oriented times three, baseline per family is times four."])]
        if chance(r, 0.4):
            v.bg = r.randint(90, 200)
        if chance(r, 0.5):
            sc.treat.append(iv_line(r, sc, p=1.0))
        sc.terms += ["weakness", "Cincinnati"]
        history(r, sc, ["dementia", "hypertension", "CKD", "diabetes", "CHF"], ["donepezil", "lisinopril", "furosemide", "metformin"], p_pmh=0.75, p_meds=0.6, p_code=0.3)
    if sc.exam is None:
        sc.exam = []
    eta_and_priority(r, sc, prio_weights=(0.1, 0.45, 0.2, 0.25))
    return sc


def fam_psych(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Female"]); age = r.randint(16, 68)
    v = pick_vitals(r, bp=(112, 160, 68, 98), hr=(78, 130), rr=(16, 24), spo2=(96, 100)); set_o2(r, v, pick(r, ["ra", None]))
    kind = pick(r, ["si", "agitation", "psychosis"])
    cc = {"si": "suicidal ideation", "agitation": "agitated, altered behavior", "psychosis": "acute psychosis"}[kind]
    sc = Scenario("psychiatric", "PSYCHIATRIC", ["MEDICAL", "BEHAVIORAL"], age, "years", sex, cc, ["suicid", "psych", "agitat", "behavior", "psychosis"], ["psychiatric", "suicidal"],
                  story=[{"si": pick(r, ["Patient called a crisis line stating a plan to overdose; denies having taken anything.", "Family found a note; patient is calm and cooperative, denies ingestion."]),
                          "agitation": pick(r, ["Police on scene, patient combative and yelling, history of schizophrenia, off medications.", "Patient agitated in a store, no weapon, police assisted."]),
                          "psychosis": pick(r, ["Patient reports hearing voices for three days and has not slept.", "Paranoid, states people are following, calm with us now."])}[kind]], vitals=v)
    sc.exam = [pick(r, ["Patient is alert and oriented times four, cooperative.", "No signs of trauma, no medical complaints.", "Pupils equal, speech rapid."])]
    if kind == "agitation" and chance(r, 0.5):
        d = drug(sc, pick(r, ["Haldol", "ketamine", "Versed"]), "", "IM", [])
        name = sc.interventions[-1].text.strip()
        dose = {"Haldol": "5 mg", "ketamine": "250 mg", "Versed": "5 mg"}[name.split()[0]]
        sc.interventions[-1].text = f"{dose} {name}"; sc.interventions[-1].numbers = [dose.split()[0]]
        sc.treat.append(f"We administered {dose} {name} and the patient is calmer, soft restraints in place.")
        sc.interventions.append(Intervention("soft restraints", ["restraint"], [], 2)); sc.terms.append("restraints")
    sc.story.append(pick(r, ["This is an emergency petition.", "Police are following us in.", "Patient is voluntary.", ""]) or "No medical complaints.")
    history(r, sc, ["depression", "bipolar disorder", "schizophrenia", "anxiety", "opioid use"], ["sertraline", "Seroquel", "lithium"], p_pmh=0.7, p_meds=0.5, p_allergy=0.45, p_code=0.0)
    sc.terms += ["psychiatric", "emergency petition"]
    eta_and_priority(r, sc, prio_weights=(0.02, 0.3, 0.45, 0.23))
    return sc


def fam_fragment(r: random.Random) -> Scenario:
    sex = pick(r, ["Male", "Female"]); age = r.randint(20, 90)
    v = Vitals()
    kind = pick(r, ["eta_only", "alert_word", "question", "update"])
    sc = Scenario("fragment", "UNCLEAR", [], age, "years", sex, "", [], [], story=[], vitals=v, fragment=True)
    if kind == "alert_word":
        word, atype, alias = pick(r, [("trauma", "TRAUMA", ["trauma"]), ("stroke alert", "STROKE", ["stroke"]), ("STEMI", "STEMI", ["stemi"]), ("cardiac arrest", "CARDIAC_ARREST", ["arrest"]), ("sepsis alert", "SEPSIS", ["sepsis"])])
        sc.alert_type, sc.alert_words, sc.cc_aliases, sc.cc_short = atype, [word], alias, word
        eta_and_priority(r, sc, p_eta=0.6, prio_weights=(0.3, 0.3, 0.0, 0.4))
        sc.story = [f"{word.capitalize() if word[0].islower() else word}" + (f", priority {PRIORITY_WORD[sc.priority]}" if sc.priority else "") + f", {age_phrase(sc)}" + (f", about {sc.eta} minutes out" if sc.eta else "") + ". How do you copy?"]
        sc.age_known = True
    elif kind == "eta_only":
        eta_and_priority(r, sc, p_eta=1.0, prio_weights=(0.1, 0.4, 0.1, 0.4))
        sc.story = [pick(r, [f"We're about {sc.eta} minutes out" + (f", priority {PRIORITY_WORD[sc.priority]}" if sc.priority else "") + ", how do you copy?",
                             f"Be advised we are {sc.eta} minutes out." , f"Updated ETA {sc.eta} minutes."])]
        sc.age_known = False
    elif kind == "question":
        sc.story = [pick(r, ["Did you copy our last?", "Can you repeat your unit and the chief complaint?", "Is a physician available for orders?", "We're on the back line, go ahead when ready."])]
        sc.age_known = False
    else:
        eta_and_priority(r, sc, p_eta=0.5, prio_weights=(0.0, 0.0, 0.0, 1.0))
        sc.story = [pick(r, ["Be advised the patient's blood pressure is now {bp}, heart rate {hr}.", "Update, patient is now alert and oriented, vitals unchanged.", "Correction, the patient is {age}, not {wrong}."])]
        if "{bp}" in sc.story[0]:
            v.bp = f"{r.randint(90, 160)}/{r.randint(50, 95)}"; v.hr = r.randint(60, 120); sc.story[0] = sc.story[0].format(bp=v.bp, hr=v.hr)
        if "{age}" in sc.story[0]:
            sc.story[0] = sc.story[0].format(age=age, wrong=age + 10); sc.age_known = True
        else:
            sc.age_known = False
        if sc.eta:
            sc.story.append(f"ETA {sc.eta} minutes.")
    return sc


FAMILIES = [(fam_stemi, 12), (fam_chest_pain, 8), (fam_stroke, 12), (fam_trauma, 16), (fam_sepsis, 10), (fam_arrest, 7),
            (fam_pediatric, 7), (fam_obstetric, 5), (fam_endocrine, 7), (fam_medical, 14), (fam_psych, 4)]
FRAGMENT_SHARE = 0.10


# ----------------------------------------------------------------------------- text

def render(r: random.Random, sc: Scenario, unit: str, hosp: str) -> str:
    """The written script, one sentence per line group as the sample scripts are."""
    if sc.fragment:
        return f"{hosp}, {unit}. " + " ".join(sc.story)
    lines: List[str] = []
    notif = pick(r, ["notification only", "patient report", "priority traffic"])
    lines.append(pick(r, [f"{hosp}, {unit}.", f"{hosp} online, this is {unit}.", f"Good {pick(r, TOD)}, {hosp}, this is {unit}.",
                          f"{unit} to {hosp}, how do you copy?", f"{unit} to {hosp} with a {notif}.", f"{hosp}, {unit} on the back line."]))
    prio = f", priority {PRIORITY_WORD[sc.priority]}" if sc.priority else ""
    ap = age_phrase(sc)
    art = article(sc.age)
    intro = pick(r, [f"We're en route to your facility with {art} {ap}, {sc.cc_short}{prio}. How do you copy so far?",
                     f"Inbound with {art} {ap}, chief complaint {sc.cc_short}{prio}.",
                     f"We have {art} {ap}, {sc.cc_short}{prio}.",
                     f"{(prio[2:].capitalize() + ', ') if prio else ''}{ap}, {sc.cc_short}. How do you copy?"])
    if sc.eta and chance(r, 0.3):
        intro = intro.rstrip(".").replace(". How do you copy so far?", "").replace(" How do you copy?", "") + f", approximately {sc.eta} minutes out."
        eta_said_early = True
    else:
        eta_said_early = False
    lines.append(intro)
    lines += sc.story
    v = sc.vitals
    vit = []
    if v.bp:
        vit.append(pick(r, [f"BP {v.bp}", f"blood pressure {v.bp}", f"pressure {v.bp}"]))
    if v.hr is not None:
        vit.append(pick(r, [f"heart rate {v.hr}{v.rhythm}", f"pulse {v.hr}{v.rhythm}", f"rate of {v.hr}{v.rhythm}"]))
    if v.rr is not None and chance(r, 0.8):
        vit.append(pick(r, [f"respiratory rate {v.rr}{v.rr_note}", f"respirations {v.rr}{v.rr_note}", f"resp rate {v.rr}{v.rr_note}"]))
    else:
        v.rr = None
    if v.spo2 is not None:
        o2 = f" {v.o2}" if v.o2 else ""
        vit.append(pick(r, [f"SpO2 {v.spo2}%{o2}", f"satting at {v.spo2}%{o2}", f"oxygen saturation {v.spo2}%{o2}", f"sat of {v.spo2}%{o2}"]))
    if v.temp:
        vit.append(pick(r, [f"temperature {v.temp}", f"temp of {v.temp}"]))
    if v.bg is not None:
        vit.append(pick(r, [f"blood glucose {v.bg}", f"blood sugar {v.bg}", f"sugar is {v.bg}", f"CBG {v.bg}"]))
    if vit:
        lead = pick(r, ["Vitals: ", "Current vitals are as follows: ", "Vitals on scene: ", "Vitals are "])
        lines.append(lead + join_list(vit) + ".")
    if v.gcs is not None:
        if v.gcs_parts:
            e, vb, m = v.gcs_parts
            lines.append(pick(r, [f"GCS is {v.gcs}, eyes {e}, verbal {vb}, motor {m}.", f"GCS {v.gcs}: {e}, {vb}, {m}."]))
        else:
            lines.append(pick(r, [f"GCS is {v.gcs}.", f"GCS of {v.gcs}.", f"GCS {v.gcs}."]))
    lines += sc.exam
    lines += sc.treat
    hist = []
    if sc.pmh_stated:
        hist.append(pick(r, ["Patient has a history of ", "Past medical history includes ", "History of ", "Medical history of "]) + join_list(sc.pmh) + ".")
    elif sc.pmh_stated is False:
        hist.append(pick(r, ["No significant medical history.", "No past medical history per family."]))
    if sc.meds_stated:
        hist.append(pick(r, ["Takes ", "Medications include ", "Home meds are ", "Patient is on "]) + join_list(sc.meds) + ".")
    elif sc.meds_stated is False:
        hist.append("No medications.")
    if sc.allergies:
        hist.append(sc.allergies[0])
    if sc.code:
        hist.append(sc.code[0])
    if hist:
        lines.append(" ".join(hist))
    if sc.request:
        lines.append(sc.request[0])
    if sc.eta and not eta_said_early:
        lines.append(pick(r, [f"ETA {sc.eta} minutes.", f"Our ETA to your facility is approximately {sc.eta} minutes.", f"We're about {sc.eta} minutes out.",
                              f"We'll be at your facility in about {sc.eta} minutes.", f"See you in approximately {sc.eta} minutes."]))
    lines.append(pick(r, [f"{unit} clear.", "How do you copy?", f"Any questions for {unit}?", "If you have no further questions, we'll see you shortly.", ""]))
    text = "\n".join(l for l in lines if l)
    return re.sub(r"[ ]{2,}", " ", text)


# ----------------------------------------------------------------------------- gold

def gold_fields(sc: Scenario) -> List[Dict[str, Any]]:
    F: List[Dict[str, Any]] = []
    add = F.append
    if sc.fragment:
        add({"path": "alert_type", "kind": "enum", "expect": sc.alert_type, "weight": 3, "accept": ["UNCLEAR"] if sc.alert_type != "UNCLEAR" else ["MEDICAL"],
             "note": "a fragment: only the alert word, if any, was spoken"})
        if getattr(sc, "age_known", False):
            add({"path": "patient.age", "kind": "int", "expect": sc.age, "weight": 2})
        else:
            add({"path": "patient.age", "kind": "unknown", "weight": 2})
        add({"path": "eta_minutes", "kind": "int", "expect": sc.eta, "weight": 2} if sc.eta else {"path": "eta_minutes", "kind": "unknown", "weight": 3})
        if sc.priority:
            add({"path": "alert_level", "kind": "enum", "expect": PRIORITY_LEVEL[sc.priority], "weight": 3, "note": "mapping: 1 RED, 2 YELLOW, 3 GREEN"})
        for p, w in (("vital_signs.blood_pressure", 2), ("vital_signs.heart_rate", 2)):
            v = sc.vitals.bp if "blood" in p else sc.vitals.hr
            add({"path": p, "kind": "contains" if "blood" in p else "int", "expect": [v] if "blood" in p else v, "weight": w} if v else {"path": p, "kind": "unknown", "weight": w})
        for p, w in (("history.past_medical", 2), ("history.medications", 3), ("history.allergies", 3), ("history.code_status", 3), ("ems_interventions", 3)):
            add({"path": p, "kind": "unknown", "weight": w})
        return F
    f = {"path": "alert_type", "kind": "enum", "expect": sc.alert_type, "weight": 3}
    if sc.alert_accept:
        f["accept"] = sc.alert_accept
    add(f)
    if sc.priority:
        add({"path": "alert_level", "kind": "enum", "expect": PRIORITY_LEVEL[sc.priority], "weight": 3, "note": "mapping: 1 RED, 2 YELLOW, 3 GREEN"})
    add({"path": "eta_minutes", "kind": "int", "expect": sc.eta, "weight": 2} if sc.eta else
        {"path": "eta_minutes", "kind": "unknown", "weight": 3, "note": "no ETA was spoken"})
    add({"path": "patient.age", "kind": "int", "expect": sc.age, "weight": 2})
    if sc.age_unit == "months":
        add({"path": "patient.age_unit", "kind": "enum", "expect": "months", "weight": 2})
    add({"path": "patient.sex", "kind": "enum", "expect": sc.sex, "weight": 2})
    add({"path": "chief_complaint", "kind": "contains", "expect": sc.cc_aliases, "weight": 3, "search": ["chief_complaint", "handoff_summary", "mechanism_or_onset"]})
    v = sc.vitals
    add({"path": "vital_signs.blood_pressure", "kind": "contains", "expect": [v.bp], "weight": 2} if v.bp else {"path": "vital_signs.blood_pressure", "kind": "unknown", "weight": 2})
    add({"path": "vital_signs.heart_rate", "kind": "int", "expect": v.hr, "weight": 2} if v.hr is not None else {"path": "vital_signs.heart_rate", "kind": "unknown", "weight": 2})
    if v.rhythm.strip(" ,"):
        words = re.findall(r"[A-Za-z]+", v.rhythm.lower())
        key = [w for w in words if w not in ("and", "with", "normal")]
        add({"path": "vital_signs.heart_rate", "kind": "contains", "expect": key, "weight": 1, "search": ["vital_signs", "assessment_findings", "handoff_summary"], "note": "rhythm was stated"})
    add({"path": "vital_signs.respiratory_rate", "kind": "int", "expect": v.rr, "weight": 2} if v.rr is not None else {"path": "vital_signs.respiratory_rate", "kind": "unknown", "weight": 1})
    if v.spo2 is not None:
        add({"path": "vital_signs.spo2", "kind": "contains", "expect": [str(v.spo2)], "weight": 2})
        if v.o2:
            add({"path": "vital_signs.spo2", "kind": "contains", "expect": v.o2_aliases, "weight": 1, "search": ["vital_signs.spo2", "ems_interventions", "assessment_findings.breathing"]})
        else:
            add({"path": "vital_signs.spo2", "kind": "not_contains", "expect": ["room air", "ambient", "nasal", "liters", "non-rebreather", "cpap"], "weight": 1, "note": "oxygen delivery was never stated"})
    else:
        add({"path": "vital_signs.spo2", "kind": "unknown", "weight": 1})
    add({"path": "vital_signs.temperature", "kind": "contains", "expect": [v.temp], "weight": 2} if v.temp else {"path": "vital_signs.temperature", "kind": "unknown", "weight": 1})
    add({"path": "vital_signs.blood_glucose", "kind": "contains", "expect": [str(v.bg)], "weight": 2} if v.bg is not None else {"path": "vital_signs.blood_glucose", "kind": "unknown", "weight": 1})
    if v.gcs is not None:
        add({"path": "vital_signs.gcs", "kind": "gcs", "expect": v.gcs, "breakdown": list(v.gcs_parts) if v.gcs_parts else None, "weight": 3 if v.gcs < 15 else 2})
    else:
        add({"path": "vital_signs.gcs", "kind": "unknown", "weight": 1})
    if sc.pmh_stated:
        add({"path": "history.past_medical", "kind": "list", "expect": [PMH[p] for p in sc.pmh], "weight": 2})
    else:
        add({"path": "history.past_medical", "kind": "unknown", "weight": 2, "note": "no history was stated" if sc.pmh_stated is None else "no significant history, stated"})
    if sc.meds_stated:
        add({"path": "history.medications", "kind": "list", "expect": [HOME_MEDS[m] for m in sc.meds], "weight": 3})
    else:
        add({"path": "history.medications", "kind": "unknown", "weight": 3})
    if sc.allergies:
        add({"path": "history.allergies", "kind": "contains", "expect": sc.allergies[1], "weight": 3})
    else:
        add({"path": "history.allergies", "kind": "unknown", "weight": 3, "note": "allergies never stated"})
    if sc.code:
        add({"path": "history.code_status", "kind": "contains", "expect": sc.code[1], "weight": 3})
    else:
        add({"path": "history.code_status", "kind": "unknown", "weight": 3, "note": "code status never stated"})
    if sc.interventions:
        add({"path": "ems_interventions", "kind": "list", "expect": [i.aliases for i in sc.interventions], "weight": 3})
        for i in sc.interventions:
            for n in i.numbers:
                add({"path": "ems_interventions", "kind": "contains", "expect": [n], "weight": i.weight, "note": f"{i.text}"})
    else:
        add({"path": "ems_interventions", "kind": "unknown", "weight": 3, "note": "no intervention was stated" if not sc.no_interventions else "no interventions, stated"})
    if sc.request:
        add({"path": "hospital_recommendations.activation_requested", "kind": "contains", "expect": sc.request[1], "weight": 2,
             "search": ["hospital_recommendations.activation_requested", "hospital_recommendations.resources_to_prepare", "hospital_recommendations.time_critical_actions"]})
    else:
        add({"path": "hospital_recommendations.activation_requested", "kind": "unknown", "weight": 2})
    if sc.destination:
        add({"path": "hospital_recommendations.suggested_destination", "kind": "enum", "expect": sc.destination, "weight": 3})
    F += sc.extra_fields
    return F


def self_check(text: str, gold: Dict[str, Any]) -> List[str]:
    """Every stated fact in the gold must be findable in the script text (normalised as the scorer does)."""
    norm = " ".join(normalize(text))
    digits = set(re.findall(r"\d+(?:\.\d+)?", text))
    problems = []
    for f in gold["fields"]:
        k = f["kind"]
        if k == "int" and str(f["expect"]) not in digits:
            problems.append(f"{f['path']}: {f['expect']} not in text")
        elif k == "gcs" and str(f["expect"]) not in digits:
            problems.append(f"gcs {f['expect']} not in text")
        elif k == "contains" and f["path"] not in ("hospital_recommendations.activation_requested",):
            alts = [" ".join(normalize(a)) for a in f["expect"]]
            if f["path"] == "history.allergies" and "no known" in f["expect"]:
                continue
            if not any(a and a in norm for a in alts):
                problems.append(f"{f['path']}: none of {f['expect']} in text")
        elif k == "list":
            for alts in f["expect"]:
                na = [" ".join(normalize(a)) for a in alts]
                if not any(a and a in norm for a in na):
                    problems.append(f"{f['path']}: none of {alts} in text")
    return problems


# ----------------------------------------------------------------------------- the set

def make_one(seed: int, i: int) -> Tuple[str, str, Dict[str, Any], Scenario]:
    r = random.Random(f"{seed}:{i}")
    sid = f"SY{i:03d}"
    kind, rng_ = pick(r, UNITS)
    unit = f"{kind} {r.randint(*rng_)}"
    hosp = pick(r, HOSPITALS)
    if r.random() < FRAGMENT_SHARE:
        sc = fam_fragment(r)
    else:
        total = sum(w for _, w in FAMILIES); x = r.uniform(0, total); acc = 0
        for fn, w in FAMILIES:
            acc += w
            if x <= acc:
                sc = fn(r); break
    text = render(r, sc, unit, hosp)
    gold = {
        "case": sid,
        "family": sc.family,
        "marker": unit.lower(),
        "source": f"generated by scripts/corpus/gen_scripts.py seed {seed} index {i}",
        "stated": {"priority": sc.priority, "eta_minutes": sc.eta, "alert_words": sc.alert_words,
                   "unit": unit, "hospital": hosp, "words": len(text.split())},
        "fields": gold_fields(sc),
    }
    problems = self_check(text, gold)
    if problems:
        raise AssertionError(f"{sid} ({sc.family}): " + "; ".join(problems) + "\n" + text)
    return sid, text, gold, sc


def splits(ids_by_family: Dict[str, List[str]], seed: int) -> Dict[str, List[str]]:
    r = random.Random(f"{seed}:splits")
    out = {"train": [], "val": [], "test": []}
    for fam in sorted(ids_by_family):
        ids = sorted(ids_by_family[fam]); r.shuffle(ids)
        n = len(ids); n_test = max(1, round(0.15 * n)) if n >= 3 else (1 if n == 2 else 0); n_val = max(1, round(0.15 * n)) if n >= 4 else 0
        out["test"] += ids[:n_test]; out["val"] += ids[n_test:n_test + n_val]; out["train"] += ids[n_test + n_val:]
    return {k: sorted(v) for k, v in out.items()}


def generate(out: Path, n: int, seed: int) -> Dict[str, Any]:
    (out / "scripts").mkdir(parents=True, exist_ok=True)
    (out / "gold").mkdir(parents=True, exist_ok=True)
    rows, fams, terms = [], {}, set()
    for i in range(1, n + 1):
        sid, text, gold, sc = make_one(seed, i)
        (out / "scripts" / f"{sid}.txt").write_text(text + "\n")
        (out / "scripts" / f"{sid}.spoken.txt").write_text(spoken(text) + "\n")
        (out / "gold" / f"{sid}.json").write_text(json.dumps(gold, indent=1, ensure_ascii=False) + "\n")
        rows.append((sid, sc.family, sc.alert_type, sc.priority or "", sc.eta or "", len(text.split())))
        fams.setdefault(sc.family, []).append(sid)
        terms.update(t.lower() for t in sc.terms if t)
        terms.update(p.lower() for p in sc.pmh); terms.update(m.lower() for m in sc.meds)
    with (out / "index.tsv").open("w") as f:
        f.write("id\tfamily\talert_type\tpriority\teta\twords\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    sp = splits(fams, seed)
    (out / "splits.json").write_text(json.dumps({"seed": seed, "n": n, "by": "script, stratified by family", **sp}, indent=1) + "\n")
    (out / "terms.txt").write_text("\n".join(sorted(terms)) + "\n")
    (out / "README.md").write_text(README.format(n=n, seed=seed, fams=", ".join(f"{k} {len(v)}" for k, v in sorted(fams.items())),
                                                 train=len(sp["train"]), val=len(sp["val"]), test=len(sp["test"])))
    return {"n": n, "families": {k: len(v) for k, v in fams.items()}, "words": sum(r[5] for r in rows), "splits": {k: len(v) for k, v in sp.items()}, "terms": len(terms)}


README = """# Generated radio reports (synth set)

{n} scripts from `scripts/corpus/gen_scripts.py --seed {seed}`: by family {fams}.
Splits (by script, stratified by family): train {train}, val {val}, test {test}; see splits.json.
Voices are split at render time (one voice is test-only); see docs/plans/FINETUNE_PLAN.md section 2.

- `scripts/<id>.txt` is the reference transcript; `<id>.spoken.txt` is what the TTS voice reads.
- `gold/<id>.json` is in the `tests/gold` field format; `scripts/benchmark.py --gold gold/` scores reports
  against it (matched by the unit phrase in the transcript or by the id in the report's file name).
- `terms.txt` is the medical-term list for the term error rate.
- The set is never edited in place after a run has been scored against it (docs/DATA_LAYOUT.md rule 2).
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=paths.datasets() / "synth_v1")
    ap.add_argument("--show", help="print one script (SY017) for this seed instead of writing the set")
    ap.add_argument("--dry", action="store_true", help="generate in memory only and report the counts")
    args = ap.parse_args(argv)
    if args.show:
        i = int(re.sub(r"\D", "", args.show))
        sid, text, gold, sc = make_one(args.seed, i)
        print(f"== {sid} [{sc.family} -> {sc.alert_type}] {len(text.split())} words\n{text}\n\n-- spoken --\n{spoken(text)}\n\n-- gold --")
        print(json.dumps(gold, indent=1))
        return 0
    if args.dry:
        fams, words = {}, 0
        for i in range(1, args.n + 1):
            sid, text, gold, sc = make_one(args.seed, i)
            fams[sc.family] = fams.get(sc.family, 0) + 1; words += len(text.split())
        print(json.dumps({"n": args.n, "families": fams, "words": words}, indent=1))
        return 0
    if (args.out / "index.tsv").exists():
        print(f"{args.out} already holds a set; a new generation is a new folder (synth_v2), see docs/DATA_LAYOUT.md", file=sys.stderr)
        return 2
    summary = generate(args.out, args.n, args.seed)
    print(json.dumps({"out": str(args.out), **summary}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
