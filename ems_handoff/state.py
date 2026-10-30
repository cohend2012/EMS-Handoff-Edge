"""
Patient state: what the system knows about the human in the ambulance, built from the reports it has
written so far, with derived observations that clinicians already use.

The report the writer produces is a snapshot of one transcript. The state is the thing that persists across
bursts: every clinical slot with the burst that first stated it and the bursts that changed it, every set of
vital signs as a timed reading, the quantities derived from them (shock index, mean arterial pressure, pulse
pressure, reverse shock index x GCS, a partial NEWS2), the threshold flags those quantities cross, what two
readings say when there are two, and the documentation the declared alert type still lacks.

Nothing here invents a value. Every number is computed from fields the writer put in the report, and every
flag carries the reading it came from and the published rule it applies. The thresholds and their sources:

- Shock index (HR / SBP) >= 1.0: higher likelihood of massive transfusion and of death in 144,951 trauma
  patients (Jehan et al., Am J Surg 2019); 0.9 separates transfusion amounts in a smaller cohort (Pariente
  Juste et al., Emergencias 2020). Observational; a flag, not a diagnosis.
- Reverse shock index x GCS (rSIG = SBP / HR x GCS) < 12.7: 70 % sensitive and 70 % specific for 7-day
  mortality in severe trauma (Spanish prehospital cohort, 2024).
- Field triage red vital signs (National Guideline for the Field Triage of Injured Patients, 2021, J Trauma
  Acute Care Surg 93(2):e49): respiratory rate < 10 or > 29, room-air SpO2 < 90 %, unable to follow commands
  (motor GCS < 6; total GCS <= 13 is used here as the stand-in because the report carries the total only),
  and the adult systolic threshold, taken as SBP < 90 in the state adaptations.
- Mean arterial pressure < 65 mmHg: the usual perfusion floor (Surviving Sepsis Campaign target).
- NEWS2 (Royal College of Physicians, 2017): computed from whichever of RR, SpO2, SBP, HR, consciousness
  and temperature are present and reported as partial when any is missing; prehospital AUC for 48-hour
  mortality 0.77 to 0.94 (Castilla y Leon cohorts, 2018 to 2021).
- Serial vitals: short-window prehospital trends are not by themselves diagnostic (Chen, Reisner, Gribok,
  Reifman, J Spec Oper Med 2010: no directional difference over 21 minutes, but wider ranges in haemorrhage),
  so two readings are reported as first, last, range and threshold crossings, never as a trajectory claim.
- Stroke documentation: last known well or time of discovery, blood glucose, the deficit or a scale (LAMS,
  RACE, FAST-ED, Cincinnati), anticoagulant use, prenotification (AHA Get With The Guidelines prehospital
  elements; LKW is the field most often missing, 19 % documented in Wisconsin EMS runs in 2018).
- STEMI documentation: a 12-lead with its finding and time (Mission: Lifeline asks for the 12-lead within 10
  minutes of contact and first-medical-contact-to-device of 90 minutes or less), aspirin given, ETA.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Parsing the writer's fields
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _number(v: Any) -> Optional[float]:
    """The first number in a field, or None for an empty or non-numeric one."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    m = _NUM.search(str(v))
    return float(m.group()) if m else None


def parse_bp(v: Any) -> Tuple[Optional[float], Optional[float]]:
    """'119/71' -> (119, 71); '120 systolic' -> (120, None); None -> (None, None)."""
    if v is None:
        return None, None
    if isinstance(v, dict):
        return _number(v.get("systolic")), _number(v.get("diastolic"))
    s = str(v)
    m = re.search(r"(\d{2,3})\s*(?:/|over)\s*(\d{2,3})", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return _number(s), None


def parse_spo2(v: Any) -> Optional[float]:
    """'96% on room air' -> 96."""
    n = _number(v)
    return n if n is not None and 0 < n <= 100 else None


def parse_temp_c(v: Any) -> Optional[float]:
    """A temperature in Celsius; Fahrenheit values (> 45) are converted."""
    n = _number(v)
    if n is None:
        return None
    if n > 45:
        n = (n - 32) * 5 / 9
    return round(n, 1) if 25 <= n <= 45 else None


def _conscious(level: Any) -> Optional[bool]:
    """True when the level-of-consciousness text says alert; False for described altered states; None unknown."""
    if not level:
        return None
    t = str(level).lower()
    if re.search(r"\bunresponsive\b|\bunconscious\b|\bconfus|\bverbal\b|\bpain\b|\bresponds? to\b|\blethargic\b|\bobtunded\b|\bgcs\s*\d", t):
        return False
    if re.search(r"\balert\b|\boriented\b|\ba\s*&\s*o\b|\bao\s*x", t):
        return True
    return None


# ---------------------------------------------------------------------------
# A timed set of vital signs and what follows from it
# ---------------------------------------------------------------------------
@dataclass
class Reading:
    source: str                          # "burst 2" or "report 3"
    time: Optional[float] = None         # epoch seconds, when known
    sbp: Optional[float] = None
    dbp: Optional[float] = None
    hr: Optional[float] = None
    rr: Optional[float] = None
    spo2: Optional[float] = None
    temp_c: Optional[float] = None
    glucose: Optional[float] = None
    gcs: Optional[float] = None
    conscious: Optional[bool] = None
    room_air: Optional[bool] = None

    @property
    def shock_index(self) -> Optional[float]:
        return round(self.hr / self.sbp, 2) if self.hr and self.sbp else None

    @property
    def map(self) -> Optional[float]:
        return round((self.sbp + 2 * self.dbp) / 3) if self.sbp and self.dbp else None

    @property
    def pulse_pressure(self) -> Optional[float]:
        return round(self.sbp - self.dbp) if self.sbp and self.dbp else None

    @property
    def rsig(self) -> Optional[float]:
        return round(self.sbp / self.hr * self.gcs, 1) if self.sbp and self.hr and self.gcs else None

    def has_vitals(self) -> bool:
        return any(v is not None for v in (self.sbp, self.hr, self.rr, self.spo2, self.temp_c, self.glucose, self.gcs))

    def to_dict(self) -> Dict[str, Any]:
        return {"source": self.source, "time": self.time, "sbp": self.sbp, "dbp": self.dbp, "hr": self.hr,
                "rr": self.rr, "spo2": self.spo2, "temp_c": self.temp_c, "glucose": self.glucose, "gcs": self.gcs,
                "conscious": self.conscious, "shock_index": self.shock_index, "map": self.map,
                "pulse_pressure": self.pulse_pressure, "rsig": self.rsig}


def reading_from_report(report: Dict[str, Any], source: str, time: Optional[float] = None) -> Reading:
    vs = report.get("vital_signs") or {}
    af = report.get("assessment_findings") or {}
    sbp, dbp = parse_bp(vs.get("blood_pressure"))
    spo2_text = str(vs.get("spo2") or "")
    r = Reading(source=source, time=time, sbp=sbp, dbp=dbp, hr=_number(vs.get("heart_rate")),
                rr=_number(vs.get("respiratory_rate")), spo2=parse_spo2(vs.get("spo2")),
                temp_c=parse_temp_c(vs.get("temperature")), glucose=_number(vs.get("blood_glucose")),
                gcs=_number(vs.get("gcs")), conscious=_conscious(af.get("level_of_consciousness")),
                room_air=(True if re.search(r"room air|\bRA\b", spo2_text, re.I) else
                          False if re.search(r"liter|L/min|lpm|\bNRB\b|cannula|nasal|CPAP|BVM|oxygen", spo2_text, re.I) else None))
    if r.gcs is not None and r.conscious is None:
        r.conscious = r.gcs >= 15
    return r


# ---------------------------------------------------------------------------
# Flags: published thresholds applied to one reading
# ---------------------------------------------------------------------------
@dataclass
class Flag:
    name: str
    text: str
    source: str          # the reading it came from
    rule: str            # the published rule, short

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "text": self.text, "source": self.source, "rule": self.rule}


FIELD_TRIAGE = "Field triage 2021 red criteria (J Trauma Acute Care Surg 93:e49)"


def flags_for(r: Reading) -> List[Flag]:
    out: List[Flag] = []
    si = r.shock_index
    if si is not None and si >= 1.0:
        out.append(Flag("shock_index", f"shock index {si:.2f} (HR {r.hr:.0f} / SBP {r.sbp:.0f}) at or above 1.0",
                        r.source, "SI >= 1 : higher massive-transfusion and mortality risk, TQIP n=144,951 (Jehan 2019)"))
    elif si is not None and si >= 0.9:
        out.append(Flag("shock_index", f"shock index {si:.2f} (HR {r.hr:.0f} / SBP {r.sbp:.0f}) at or above 0.9",
                        r.source, "SI >= 0.9 : separates transfusion need in polytrauma (Pariente Juste 2020)"))
    if r.sbp is not None and r.sbp < 90:
        out.append(Flag("hypotension", f"SBP {r.sbp:.0f} below 90", r.source, FIELD_TRIAGE))
    if r.rr is not None and (r.rr < 10 or r.rr > 29):
        out.append(Flag("respiratory_rate", f"respiratory rate {r.rr:.0f} outside 10 to 29", r.source, FIELD_TRIAGE))
    if r.spo2 is not None and r.spo2 < 90 and r.room_air is not False:
        out.append(Flag("hypoxia", f"SpO2 {r.spo2:.0f} % below 90 on room air", r.source, FIELD_TRIAGE))
    if r.gcs is not None and r.gcs <= 13:
        out.append(Flag("altered_mental_status", f"GCS {r.gcs:.0f} (13 or less)", r.source,
                        FIELD_TRIAGE + "; stands in for 'unable to follow commands'"))
    if r.map is not None and r.map < 65:
        out.append(Flag("low_map", f"mean arterial pressure {r.map:.0f} below 65", r.source,
                        "MAP < 65 mmHg: perfusion floor (Surviving Sepsis Campaign)"))
    rsig = r.rsig
    if rsig is not None and rsig < 12.7:
        out.append(Flag("rsig", f"reverse shock index x GCS {rsig:.1f} below 12.7", r.source,
                        "rSIG < 12.7: 70 % sensitive / 70 % specific for 7-day mortality in severe trauma (2024)"))
    if r.glucose is not None and (r.glucose < 60 or r.glucose > 400):
        out.append(Flag("glucose", f"blood glucose {r.glucose:.0f} outside 60 to 400", r.source,
                        "hypo- or marked hyperglycaemia; stroke mimic check (AHA prehospital stroke elements)"))
    return out


# ---------------------------------------------------------------------------
# NEWS2, partial when fields are missing
# ---------------------------------------------------------------------------
def news2(r: Reading) -> Dict[str, Any]:
    """Royal College of Physicians NEWS2 (2017), SpO2 scale 1, no supplemental-oxygen point unless stated."""
    parts: Dict[str, int] = {}
    missing: List[str] = []
    if r.rr is not None:
        parts["rr"] = 3 if r.rr <= 8 else 1 if r.rr <= 11 else 0 if r.rr <= 20 else 2 if r.rr <= 24 else 3
    else:
        missing.append("respiratory_rate")
    if r.spo2 is not None:
        parts["spo2"] = 3 if r.spo2 <= 91 else 2 if r.spo2 <= 93 else 1 if r.spo2 <= 95 else 0
        if r.room_air is False:
            parts["oxygen"] = 2
    else:
        missing.append("spo2")
    if r.sbp is not None:
        parts["sbp"] = 3 if r.sbp <= 90 else 2 if r.sbp <= 100 else 1 if r.sbp <= 110 else 0 if r.sbp <= 219 else 3
    else:
        missing.append("blood_pressure")
    if r.hr is not None:
        parts["hr"] = 3 if r.hr <= 40 else 1 if r.hr <= 50 else 0 if r.hr <= 90 else 1 if r.hr <= 110 else 2 if r.hr <= 130 else 3
    else:
        missing.append("heart_rate")
    if r.conscious is not None:
        parts["consciousness"] = 0 if r.conscious else 3
    else:
        missing.append("level_of_consciousness")
    if r.temp_c is not None:
        t = r.temp_c
        parts["temp"] = 3 if t <= 35.0 else 1 if t <= 36.0 else 0 if t <= 38.0 else 1 if t <= 39.0 else 2
    else:
        missing.append("temperature")
    score = sum(parts.values())
    band = ("high" if score >= 7 or any(v == 3 for k, v in parts.items() if k != "oxygen") else
            "medium" if score >= 5 else "low")
    return {"score": score, "parts": parts, "missing": missing, "partial": bool(missing), "band": band,
            "rule": "NEWS2 (RCP 2017); prehospital 48 h mortality AUC 0.77 to 0.94 (Castilla y Leon cohorts)"}


# ---------------------------------------------------------------------------
# Two or more readings
# ---------------------------------------------------------------------------
_SERIES = (("sbp", "SBP", lambda v: v < 90), ("hr", "HR", lambda v: v > 110), ("rr", "RR", lambda v: v < 10 or v > 29),
           ("spo2", "SpO2", lambda v: v < 90), ("shock_index", "shock index", lambda v: v >= 1.0),
           ("gcs", "GCS", lambda v: v <= 13), ("map", "MAP", lambda v: v < 65))


def series(readings: List[Reading]) -> List[Dict[str, Any]]:
    """For each vital with two or more values: first, last, change, range, and any threshold crossing.

    Chen et al. 2010 found no directional signal in short prehospital trends, only wider ranges in haemorrhage,
    so the range and the crossings are the observation; the change is shown, not interpreted.
    """
    out = []
    for key, label, bad in _SERIES:
        pts = [(r.source, getattr(r, key)) for r in readings if getattr(r, key) is not None]
        if len(pts) < 2:
            continue
        values = [v for _, v in pts]
        crossings = [f"{label} crossed the threshold between {pts[i - 1][0]} and {pts[i][0]} ({values[i - 1]:g} to {values[i]:g})"
                     for i in range(1, len(pts)) if bad(values[i]) != bad(values[i - 1])]
        out.append({"vital": key, "label": label, "n": len(pts), "first": values[0], "last": values[-1],
                    "change": round(values[-1] - values[0], 2), "min": min(values), "max": max(values),
                    "sources": [s for s, _ in pts], "crossings": crossings})
    return out


# ---------------------------------------------------------------------------
# What the alert type still lacks
# ---------------------------------------------------------------------------
def _text_of(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_text_of(x) for x in v)
    if isinstance(v, dict):
        return " ".join(_text_of(x) for x in v.values())
    return str(v)


def _get(report: Dict[str, Any], path: str) -> Any:
    cur: Any = report
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _empty(v: Any) -> bool:
    return v is None or v == "" or v == [] or v == {} or (isinstance(v, str) and v.strip().lower() in ("", "null", "none", "unknown", "not stated"))


def _mentions(report: Dict[str, Any], pattern: str, *paths: str) -> bool:
    text = " ".join(_text_of(_get(report, p)) for p in paths) if paths else _text_of(report)
    return re.search(pattern, text, re.I) is not None


# (label, check): check(report) -> True when documented
_COMMON = [
    ("age", lambda r: not _empty(_get(r, "patient.age"))),
    ("sex", lambda r: not _empty(_get(r, "patient.sex"))),
    ("chief complaint", lambda r: not _empty(_get(r, "chief_complaint"))),
    ("ETA", lambda r: not _empty(_get(r, "eta_minutes"))),
]
REQUIRED: Dict[str, List[Tuple[str, Any]]] = {
    "STROKE": _COMMON + [
        ("last known well or onset time", lambda r: _mentions(r, r"last known well|\bLKW\b|onset|last seen normal|time of discovery|\d+\s*(?:min|hour|hr)s?\s*ago|at\s*\d{1,2}[:.]?\d{2}", "mechanism_or_onset", "chief_complaint")),
        ("blood glucose", lambda r: not _empty(_get(r, "vital_signs.blood_glucose"))),
        ("deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati)", lambda r: _mentions(r, r"LAMS|RACE|FAST|Cincinnati|facial|droop|arm drift|weakness|aphasia|slurred|speech|hemipar|gaze|neglect", "assessment_findings.neuro", "assessment_findings.other_findings", "chief_complaint")),
        ("anticoagulant use (yes or no)", lambda r: _mentions(r, r"anticoag|blood thinner|warfarin|coumadin|eliquis|apixaban|xarelto|rivaroxaban|pradaxa|dabigatran|heparin|no (?:blood thinners|anticoagulants)|not on", "history.medications", "history.past_medical")),
    ],
    "STEMI": _COMMON + [
        ("12-lead finding", lambda r: _mentions(r, r"12[- ]lead|\bST\b|STEMI|elevation|ECG|EKG", "assessment_findings", "chief_complaint", "ems_interventions")),
        ("aspirin given or contraindicated", lambda r: _mentions(r, r"aspirin|\bASA\b", "ems_interventions", "history")),
        ("blood pressure", lambda r: not _empty(_get(r, "vital_signs.blood_pressure"))),
        ("heart rate", lambda r: not _empty(_get(r, "vital_signs.heart_rate"))),
    ],
    "TRAUMA": _COMMON + [
        ("mechanism", lambda r: not _empty(_get(r, "mechanism_or_onset"))),
        ("GCS or mental status", lambda r: not _empty(_get(r, "vital_signs.gcs")) or not _empty(_get(r, "assessment_findings.level_of_consciousness"))),
        ("blood pressure", lambda r: not _empty(_get(r, "vital_signs.blood_pressure"))),
        ("respiratory rate", lambda r: not _empty(_get(r, "vital_signs.respiratory_rate"))),
        ("SpO2", lambda r: not _empty(_get(r, "vital_signs.spo2"))),
    ],
    "SEPSIS": _COMMON + [
        ("temperature", lambda r: not _empty(_get(r, "vital_signs.temperature"))),
        ("heart rate", lambda r: not _empty(_get(r, "vital_signs.heart_rate"))),
        ("respiratory rate", lambda r: not _empty(_get(r, "vital_signs.respiratory_rate"))),
        ("blood pressure", lambda r: not _empty(_get(r, "vital_signs.blood_pressure"))),
        ("mental status", lambda r: not _empty(_get(r, "assessment_findings.level_of_consciousness"))),
        ("suspected source", lambda r: _mentions(r, r"infect|pneumonia|\bUTI\b|urinary|cellulitis|wound|sepsis|fever", "chief_complaint", "assessment_findings", "history")),
    ],
    "CARDIAC_ARREST": _COMMON[:3] + [
        ("initial rhythm", lambda r: _mentions(r, r"asystole|\bPEA\b|v[- ]?fib|v[- ]?tach|shockable|rhythm", "assessment_findings", "ems_interventions", "chief_complaint")),
        ("CPR or ROSC status", lambda r: _mentions(r, r"\bCPR\b|compressions|\bROSC\b|pulses? back|working code|arrest", "ems_interventions", "chief_complaint", "assessment_findings")),
        ("ETA", lambda r: not _empty(_get(r, "eta_minutes"))),
    ],
}
REQUIRED["PEDIATRIC"] = _COMMON + [("weight or age for dosing", lambda r: _mentions(r, r"\bkg\b|kilo|pound|\blb", "patient", "ems_interventions") or not _empty(_get(r, "patient.age")))]
for _k in ("OBSTETRIC", "ENDOCRINE", "MEDICAL", "PSYCHIATRIC", "UNCLEAR", ""):
    REQUIRED.setdefault(_k, list(_COMMON))


def missing_documentation(report: Dict[str, Any]) -> List[str]:
    """The required items for the report's alert type that the report does not document."""
    alert = str(report.get("alert_type") or "").upper()
    return [label for label, check in REQUIRED.get(alert, _COMMON) if not check(report)]


# ---------------------------------------------------------------------------
# Slots with provenance, from the report history
# ---------------------------------------------------------------------------
@dataclass
class Slot:
    path: str
    value: Any
    first: str                               # the source that first stated it
    changed: List[str] = field(default_factory=list)   # sources where the value changed
    previous: List[Any] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "value": self.value, "first": self.first, "changed": list(self.changed),
                "previous": list(self.previous)}


_CLINICAL_PREFIXES = ("patient.", "chief_complaint", "mechanism_or_onset", "eta_minutes", "alert_type", "alert_level",
                      "vital_signs.", "assessment_findings.", "ems_interventions", "history.")


def _leaves(report: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        else:
            out[prefix] = value
    walk("", report)
    return {k: v for k, v in out.items() if k.startswith(_CLINICAL_PREFIXES)}


@dataclass
class PatientState:
    slots: Dict[str, Slot] = field(default_factory=dict)
    readings: List[Reading] = field(default_factory=list)
    flags: List[Flag] = field(default_factory=list)
    series: List[Dict[str, Any]] = field(default_factory=list)
    news2: Optional[Dict[str, Any]] = None
    missing: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    alert_type: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"alert_type": self.alert_type,
                "slots": {k: s.to_dict() for k, s in self.slots.items()},
                "readings": [r.to_dict() for r in self.readings],
                "flags": [f.to_dict() for f in self.flags],
                "series": self.series, "news2": self.news2, "missing": self.missing, "conflicts": self.conflicts}


_CONFLICT_PATHS = ("patient.age", "patient.sex", "eta_minutes", "alert_type")


def build_state(history: List[Tuple[str, Optional[float], Dict[str, Any]]]) -> PatientState:
    """The state after a sequence of (source, time, report), oldest first.

    `source` names what produced the report ("burst 2", "report 3"); vitals are read from every report, slots
    from the sequence of values, and the flags, NEWS2 and missing list from the latest report.
    """
    st = PatientState()
    if not history:
        return st
    for source, at, report in history:
        if not isinstance(report, dict):
            continue
        for path, value in _leaves(report).items():
            if _empty(value) or (path == "alert_type" and str(value).upper() == "UNCLEAR"):
                continue        # UNCLEAR means this text stated no type; it is not a change of type
            slot = st.slots.get(path)
            if slot is None:
                st.slots[path] = Slot(path=path, value=value, first=source)
            elif slot.value != value:
                slot.previous.append(slot.value)
                slot.changed.append(source)
                slot.value = value
        r = reading_from_report(report, source, at)
        if r.has_vitals() and (not st.readings or r.to_dict() != {**st.readings[-1].to_dict(), "source": source, "time": at}):
            st.readings.append(r)
    # the merged report: the latest non-empty value of every slot, so a short later burst cannot blank the checklist
    merged: Dict[str, Any] = {}
    for path, slot in st.slots.items():
        cur = merged
        parts = path.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = slot.value
    latest = merged
    st.alert_type = str(latest.get("alert_type") or "")
    if st.readings:
        st.flags = flags_for(st.readings[-1])
        st.news2 = news2(st.readings[-1])
    st.series = series(st.readings)
    st.missing = missing_documentation(latest) if latest else []
    for path in _CONFLICT_PATHS:
        s = st.slots.get(path)
        if s and s.changed:
            st.conflicts.append(f"{path}: {s.previous[0]} ({s.first}) then {s.value} ({s.changed[-1]})")
    return st
