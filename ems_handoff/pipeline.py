"""
pipeline.py — turn a radio-report transcript into the hospital handoff report.

Purpose
    The single path every report takes, whether it was spoken, pasted, or a
    sample case:

        transcript
          -> drop the stop phrase
          -> transcript_corrector  (fix known mis-hearings, flag doubtful words)
          -> lexicon_rag           (EMS term definitions for the prompt)
          -> LLM                   (EXTRACTION_PROMPT -> JSON)
          -> validate              (required fields, alert level, null sections filled)
          -> textbook guidance     (optional, only if database/ exists)

    Every transcript is processed, however short: a radio burst of three words is
    a real input. The model is asked to return the full structure with unstated
    fields null and alert_type UNCLEAR when the words do not identify a problem.
    Only a transcript of zero words skips the model, and even then the result is
    a complete report that says the transcript was empty.

Inputs
    Text, an ``LLMClient``, and optional lexicon and textbook helpers. Stage
    updates go to the ``on_stage`` callback, which the app shows as progress.

Outputs
    A report dict. Pipeline bookkeeping (transcript, corrections, timings,
    model) goes under the ``_meta`` key, so the clinical fields stay clean.
    Failures raise ``PipelineError`` with a message fit to show on screen.

Threads
    Runs in the app's report worker thread. Nothing is printed. Everything is
    logged.
"""

from __future__ import annotations

import copy
import json
import os
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .llm import LLMClient, LLMError
from .phrase import strip_phrase
from .prompt import GUIDANCE_PROMPT, extraction_prompt
from .transcript_corrector import assumption_notes, correct_transcript, format_suspects_for_llm

logger = logging.getLogger(__name__)

MAX_REPORT_CHARS = 8000

REQUIRED_FIELDS = {
    "alert_type", "alert_level", "patient", "chief_complaint",
    "vital_signs", "assessment_findings", "history", "handoff_summary",
}
ALERT_TYPES = {"STEMI", "STROKE", "TRAUMA", "SEPSIS", "CARDIAC_ARREST", "PEDIATRIC",
               "OBSTETRIC", "MEDICAL", "PSYCHIATRIC", "ENDOCRINE", "UNCLEAR"}
ALERT_LEVELS = {"RED", "YELLOW", "GREEN"}
CONFIDENCE_LEVELS = {"HIGH", "MEDIUM", "LOW"}


class PipelineError(RuntimeError):
    """A report could not be produced. The message is shown to the user."""


# Descriptive strings from EXTRACTION_PROMPT that a small model sometimes copies back
# as if they were values ("gcs": "Total and breakdown if mentioned"), and the words a model
# uses for "nothing said". They become null: an unstated field is null, never a word.
_PLACEHOLDERS = {
    "systolic/diastolic or null", "number or null", "percentage and oxygen delivery method",
    "if mentioned", "total and breakdown if mentioned", "brief one-line summary",
    "how/when this started", "alert/verbal/pain/unresponsive and orientation",
    "patent/compromised/secured", "description of respiratory effort and lung sounds",
    "skin signs, pulses, bleeding", "pupils, movement, deficits", "list of other significant findings",
    "what was done", "specifics (dose, route, size, etc.)", "patient response if mentioned",
    "list of conditions", "list of medications", "list or nkda",
    "full code / dnr / dni / other if mentioned",
    "each allergy exactly as the paramedic said it", "the code status exactly as the paramedic said it",
    "what the paramedic requested (cath lab, trauma team, etc.)",
    "list specific resources, teams, equipment", "actions to initiate before patient arrival",
    "single sentence for quick verbal handoff", "top 2-3 clinical concerns",
    "what patient will need immediately on arrival", "important information not provided",
    "any assumptions made from context", "null", "unknown", "not stated", "not provided", "n/a",
}


def scrub_placeholders(value: Any) -> Any:
    """Replace prompt placeholder text with None, recursively. Lists lose placeholder items."""
    if isinstance(value, str):
        return None if value.strip().lower() in _PLACEHOLDERS else value
    if isinstance(value, list):
        return [v for v in (scrub_placeholders(x) for x in value) if v is not None]
    if isinstance(value, dict):
        return {k: scrub_placeholders(v) for k, v in value.items()}
    return value


def blank_report() -> Dict[str, Any]:
    """Every field the dashboard shows, with nothing stated. The zero-word report is exactly
    this; a model reply is laid over it so a null section still has all its keys."""
    return {
        "alert_type": "UNCLEAR", "alert_level": None, "eta_minutes": None,
        "patient": {"age": None, "age_unit": None, "sex": None},
        "chief_complaint": None, "mechanism_or_onset": None,
        "vital_signs": {"blood_pressure": None, "heart_rate": None, "respiratory_rate": None,
                        "spo2": None, "temperature": None, "blood_glucose": None, "gcs": None},
        "assessment_findings": {"level_of_consciousness": None, "airway": None, "breathing": None,
                                "circulation": None, "neuro": None, "other_findings": []},
        "ems_interventions": [],
        "history": {"past_medical": [], "medications": [], "allergies": None, "code_status": None},
        "hospital_recommendations": {"activation_requested": None, "suggested_destination": None,
                                     "resources_to_prepare": [], "time_critical_actions": []},
        "handoff_summary": {"one_liner": None, "key_concerns": [], "pending_needs": []},
        "data_quality": {"confidence": "LOW", "missing_info": [], "assumptions": []},
    }


def fill_blanks(skeleton: Any, value: Any) -> Any:
    """``value`` laid over ``skeleton``: a null or missing part takes the skeleton's empty
    shape, dicts merge key by key (extra keys kept), and a lone value where a list belongs
    becomes a one-item list. A wrong type where a dict belongs is returned as is, so the
    caller can report it."""
    if value is None:
        return copy.deepcopy(skeleton)
    if isinstance(skeleton, dict):
        if not isinstance(value, dict):
            return value
        out = {k: fill_blanks(sub, value.get(k)) for k, sub in skeleton.items()}
        out.update({k: v for k, v in value.items() if k not in skeleton})
        return out
    if isinstance(skeleton, list):
        return value if isinstance(value, list) else [value]
    return value


# (label for missing_info, path in the report) for the facts a handoff normally carries
_FACTS = [
    ("patient age", "patient.age"), ("patient sex", "patient.sex"),
    ("chief complaint", "chief_complaint"), ("onset or mechanism", "mechanism_or_onset"),
    ("alert level", "alert_level"), ("ETA", "eta_minutes"), ("vital signs", "vital_signs"),
    ("assessment findings", "assessment_findings"), ("EMS interventions", "ems_interventions"),
    ("medical history", "history"), ("allergies", "history.allergies"), ("code status", "history.code_status"),
    ("activation request", "hospital_recommendations.activation_requested"),
]


def _is_empty(v: Any) -> bool:
    if isinstance(v, dict):
        return all(_is_empty(x) for x in v.values())
    if isinstance(v, list):
        return all(_is_empty(x) for x in v)
    return v is None or v == ""


def unstated(report: Dict[str, Any]) -> List[str]:
    """Which of the usual handoff facts this report has nothing for, as missing_info labels."""
    out = []
    for label, path in _FACTS:
        v: Any = report
        for part in path.split("."):
            v = v.get(part) if isinstance(v, dict) else None
        if _is_empty(v):
            out.append(label)
    return out


_ALLERGY_WORDS = re.compile(r"allerg|\bnkda\b|\bnka\b|no known drug", re.I)
# the one allergy statement a transcript can make that a small model reads as "nothing to report"
_NO_ALLERGIES = re.compile(r"\bnkda\b|\bnka\b|no known (?:drug |medication )?allerg\w*|no (?:drug |medication )?allergies|allergies[: ]+none\b|denies (?:any )?allergies", re.I)
# an explicit code status spoken as such (the generic words of _CODE_WORDS are not a status by themselves)
_CODE_STATED = re.compile(r"\b(?:dnr|dni|dnar|do not resuscitate|do not intubate|full code|no code|comfort (?:care|measures)(?: only)?|cmo)\b", re.I)
_CODE_WORDS = re.compile(r"\b(?:dnr|dni|dnar|do not (?:resuscitate|intubate|attempt)|full code|no code|code status|"
                         r"polst|molst|full treatment|comfort (?:care|measures)|cmo|advance directive|living will|"
                         r"resuscitat\w*|life[- ]sustaining)\b", re.I)


def ground_history_defaults(report: Dict[str, Any], text: str) -> List[str]:
    """Null out allergies and code status the transcript never mentions; returns notes.

    The 4B model writes "NKDA" and "Full code" into reports about patients whose
    allergies and directives were never spoken (49 and 23 reports in the post-fix
    corpus run). A value is kept only when the transcript talks about allergies or a
    code status at all; what it says is still the model's reading."""
    notes: List[str] = []
    hist = report.get("history")
    if not isinstance(hist, dict):
        return notes
    if isinstance(hist.get("allergies"), str):
        hist["allergies"] = [hist["allergies"]]
    if hist.get("allergies") and not _ALLERGY_WORDS.search(text):
        notes.append(f"Allergies set to null: the transcript never mentions allergies (the model wrote {hist['allergies']!r})")
        hist["allergies"] = None
    if hist.get("code_status") and not _CODE_WORDS.search(text):
        notes.append(f"Code status set to null: the transcript never mentions one (the model wrote {hist['code_status']!r})")
        hist["code_status"] = None
    # the opposite failure: the paramedic said "no known allergies" or "full code" and the model wrote null,
    # reading a stated negative as nothing said. The spoken phrase itself goes in (the schema asks for the
    # words as the paramedic said them), so the report stays grounded.
    dq = report.get("data_quality") or {}
    missing = dq.get("missing_info")
    if not hist.get("allergies"):
        m = _NO_ALLERGIES.search(text)
        if m:
            hist["allergies"] = [m.group(0).strip().rstrip(".,")]
            notes.append(f"Allergies taken from the transcript ('{m.group(0).strip()}'); the model left them null")
            if isinstance(missing, list):
                missing[:] = [x for x in missing if "allerg" not in str(x).lower()]
    if not hist.get("code_status"):
        said = []
        for m in _CODE_STATED.finditer(text):
            w = m.group(0).strip()
            if w.lower() not in (x.lower() for x in said):
                said.append(w)
        if said:
            hist["code_status"] = " ".join(said)
            notes.append(f"Code status taken from the transcript ('{' '.join(said)}'); the model left it null")
            if isinstance(missing, list):
                missing[:] = [x for x in missing if "code" not in str(x).lower()]
    if isinstance(missing, list):
        for label, key in (("allergies", "allergies"), ("code status", "code_status")):
            if not hist.get(key) and not any(label in str(m).lower() for m in missing):
                missing.append(label)
    return notes


# Claims the model adds to what was said, found on the generated set's clean text (SYNTH_V1_2026-10-27.md):
# a GCS breakdown when only the total was spoken, an EMS-given drug filed under home medications, an
# intervention nobody mentioned. Each rule keeps what the transcript supports and notes what it removed.
_GCS_BREAKDOWN_CUES = re.compile(r"\b(?:eyes?|verbal|motor|opening|\bE\s?\d|\bV\s?\d|\bM\s?\d)\b", re.I)
_HOME_MED_CUES = re.compile(r"\b(?:takes|taking|medications?|meds|prescribed|home|daily|history|on)\b", re.I)
_REQUEST_CUES = re.compile(r"\b(?:request\w*|activat\w*|alert|notif\w*|calling (?:a|an|the)|call (?:a|an|the)|have .{0,30}(?:ready|standing by|waiting)|"
                           r"meet us|standing by|be ready|need (?:a|an|the)|physician|orders?)\b", re.I)
_GENERIC_WORDS = {"applied", "established", "administered", "given", "gave", "placed", "started", "initiated", "obtained",
                  "performed", "patient", "with", "via", "and", "the", "of", "to", "in", "on", "at", "for", "per", "times",
                  "dose", "route", "response", "none", "unknown", "not", "mentioned", "stated", "null", "access", "therapy"}


def _tokens(text: str) -> List[str]:
    return [str(_NUMBER_WORDS.get(t, t)) for t in re.split(r"[^\w.]+", text.lower()) if t]


def _supported(words: List[str], transcript_tokens: set) -> bool:
    """Some content word of the claim occurs in the transcript (whole, or by a five-letter stem for long words)."""
    for w in words:
        if w in _GENERIC_WORDS or len(w) < 2:
            continue
        if w in transcript_tokens:
            return True
        if len(w) >= 6 and any(t.startswith(w[:5]) for t in transcript_tokens if len(t) >= 5):
            return True
    return False


def ground_claims(report: Dict[str, Any], text: str) -> List[str]:
    """Remove the three claim shapes above when the transcript does not carry them; returns notes."""
    notes: List[str] = []
    toks = _tokens(text)
    tokset = set(toks)
    # 1. GCS: a breakdown that was never spoken
    vs = report.get("vital_signs")
    if isinstance(vs, dict) and vs.get("gcs") not in (None, "", []):
        g = vs["gcs"]
        gtext = json.dumps(g) if isinstance(g, (dict, list)) else str(g)
        nums = [int(n) for n in re.findall(r"\d+", gtext)]
        if nums and len([n for n in nums[1:] if n <= 6]) >= 3 and not _GCS_BREAKDOWN_CUES.search(text):
            vs["gcs"] = str(nums[0])
            notes.append(f"GCS breakdown removed: only the total {nums[0]} was spoken (the model wrote {gtext[:60]!r})")
    # 2. home medications that are the drugs EMS gave
    hist = report.get("history")
    ivs = report.get("ems_interventions") or []
    given = " ".join(json.dumps(i) if isinstance(i, dict) else str(i) for i in ivs).lower()
    if isinstance(hist, dict) and isinstance(hist.get("medications"), list) and given:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept, dropped = [], []
        for item in hist["medications"]:
            name = re.sub(r"[^a-z]", " ", str(item).lower()).split()
            name = [w for w in name if len(w) > 2]
            drug_words = [w for w in name if w in given or any(w[:5] == gw[:5] for gw in re.findall(r"[a-z]{5,}", given))]
            if not drug_words:
                kept.append(item); continue
            mentions = [sn for sn in sentences if any(w[:5] in sn.lower() for w in drug_words)]
            if mentions and not any(_HOME_MED_CUES.search(sn) for sn in mentions):
                dropped.append(item)
            else:
                kept.append(item)
        if dropped:
            hist["medications"] = kept
            notes.append(f"Home medications removed: {dropped!r} were given by EMS, not stated as the patient's medications")
    # 4. a request nobody made: activation_requested is what the crew asked for, not what the ED should do
    hr = report.get("hospital_recommendations")
    if isinstance(hr, dict) and hr.get("activation_requested") not in (None, "", []) and not _REQUEST_CUES.search(text):
        asked = hr["activation_requested"]
        if not _is_empty(asked) and not any(w in str(asked).lower() for w in ("none", "not ", "no ")):
            notes.append(f"Activation request cleared: the crew asked for nothing (the model wrote {str(asked)[:60]!r})")
            hr["activation_requested"] = None
    # 5. "room air" the crew never said
    if isinstance(vs, dict) and isinstance(vs.get("spo2"), str) and re.search(r"room air|\bRA\b", vs["spo2"], re.I) \
            and not re.search(r"room air|\bR\.?A\.?\b", text, re.I):
        vs["spo2"] = re.sub(r"\s*(?:on|,)?\s*(?:room air|\bRA\b)", "", vs["spo2"], flags=re.I).strip(" ,") or None
        notes.append("Oxygen delivery removed from SpO2: 'room air' was never said")
    # 3. interventions with no word in the transcript
    if isinstance(ivs, list) and ivs:
        kept = []
        for item in ivs:
            claim = " ".join(str(item.get(k) or "") for k in ("intervention", "details")) if isinstance(item, dict) else str(item)
            words = _tokens(claim)
            if _supported(words, tokset):
                kept.append(item)
            else:
                notes.append(f"Intervention removed: {claim.strip()!r} has no word in the transcript")
        report["ems_interventions"] = kept
    return notes


# Alert words a radio fragment can carry. UNCLEAR is for a transcript with none of them.
_ALERT_WORDS = [
    ("CARDIAC_ARREST", re.compile(r"cardiac arrest|\bcpr\b|pulseless|asystole|v-?fib\b|ventricular fibrillation|\brosc\b", re.I)),
    ("STEMI", re.compile(r"\bstemi\b|\bst[- ]elevation", re.I)),
    ("STROKE", re.compile(r"\bstroke\b|\bcva\b|brain bleed|\btia\b|facial droop|hemorrhagic", re.I)),
    ("TRAUMA", re.compile(r"\btrauma\b|\bmvc\b|\bmva\b|\bgsw\b|gunshot|\bstab(?:bed|bing|bed)?\b|fall from|fell (?:from|down|off)|rollover|pedestrian struck|\bfracture", re.I)),
    ("SEPSIS", re.compile(r"\bsepsis\b|\bseptic\b", re.I)),
]


# An alert word that is denied, asked about and answered "no", or named only as history is not a statement
# of the problem: "Did you just use your stroke? No." (S030_01) stays UNCLEAR; "history of stroke" is a
# condition, not the complaint.
_NOT_STATED_BEFORE = re.compile(
    r"\b(?:no|not|negative(?:\s+for)?|denies|denied|deny|without|rul(?:e|ed|ing)\s+out|non|didn't|did\s+not|isn't|is\s+not|"
    r"wasn't|was\s+not|never|any|history\s+of|hx\s+of|prior|previous|past|old)\W+(?:\w+\W+){0,2}$", re.I)
_NOT_STATED_AFTER = re.compile(r"^\W*(?:no|nope|negative|not)\b", re.I)


_DESTINATIONS = ("cath lab", "trauma bay", "resuscitation bay", "critical care", "general ed")
_REQUEST_WORDS = re.compile(r"request|activat|notif|alert|please have|need (?:the|a)|standing by|stand by|be ready|prepar|team|page|call ahead", re.I)
_ETA_PHRASE = re.compile(r"\b(?:minutes?|mins?|see you|how do you copy|copy|eta)\b", re.I)


def ground_activation(report: Dict[str, Any], text: str) -> Optional[str]:
    """The activation request is what the crew asked the hospital for. Two inventions seen on 10-30 are cleared:
    an ETA or closing phrase written as the request ("see you in approximately 20 minutes"), and a bare destination
    name ("Cath Lab", "Resuscitation Bay") when the transcript never asks for it near that term. Returns the note
    added to data_quality.assumptions, or None when nothing changed."""
    hr = report.get("hospital_recommendations") or {}
    val = hr.get("activation_requested")
    if not val or not isinstance(val, str):
        return None
    v = val.strip()
    if _ETA_PHRASE.search(v) and not _REQUEST_WORDS.search(v):
        hr["activation_requested"] = None
        return f"Activation request cleared: '{v[:40]}' is a time or closing phrase, not a request"
    if v.lower().rstrip(".") in _DESTINATIONS:
        low = text.lower()
        term = v.lower().rstrip(".")
        near = False
        for m in re.finditer(re.escape(term.split()[0]), low):
            window = low[max(0, m.start() - 80):m.end() + 80]
            if _REQUEST_WORDS.search(window):
                near = True
                break
        if not near:
            hr["activation_requested"] = None
            return f"Activation request cleared: '{v}' is a destination name and the transcript never asks for it"
    return None


def alert_from_words(text: str) -> Optional[Tuple[str, str]]:
    """(alert_type, the word that names it) when the transcript states an alert word positively, else None.

    "This is Ambulance 721 coming to, what is it, trauma? ... priority 2" (S038_01) must be
    TRAUMA however short. The model is asked the same; this is the check behind it, used
    only when the model answered UNCLEAR. A type the model did name is never overridden.
    A word that is negated within three words before it, answered "no" right after it, or
    introduced as history does not count."""
    for alert_type, pattern in _ALERT_WORDS:
        for m in pattern.finditer(text):
            before = re.split(r"[.!?;]\s", text[max(0, m.start() - 60):m.start()])[-1]   # the same sentence only
            after = text[m.end():m.end() + 20]
            if _NOT_STATED_BEFORE.search(before) or _NOT_STATED_AFTER.match(after):
                continue
            return alert_type, m.group(0)
    return None


_PRIORITY = re.compile(r"\bpriority\s+(?:number\s+)?(one|two|three|1|2|3|to|too|won|tree|free)\b", re.I)
_PRIORITY_WORDS = {"one": 1, "won": 1, "1": 1, "two": 2, "to": 2, "too": 2, "2": 2, "three": 3, "tree": 3, "free": 3, "3": 3}
LEVEL_FOR_PRIORITY = {1: "RED", 2: "YELLOW", 3: "GREEN"}
_PREPOSITION_NEXT = re.compile(r"\s+(?:the|a|an|our|your|this|that|his|her|their|my|its|be|get|go|make)\b", re.I)
_BURST_HEADER = re.compile(r"^\[Burst (\d+)\b", re.M)


def stated_priority(text: str) -> Optional[Tuple[int, str, Optional[int]]]:
    """The priority the crew stated, as (priority, the phrase, the burst it was said in or None).

    "Priority 2" is how the crew states the level on the radio, and Whisper writes the number as
    "to" or "too" about as often as "2". The last statement wins (a crew that upgrades says the
    new priority last). The burst is read from the session's "[Burst N" headers when present."""
    last = None
    for m in _PRIORITY.finditer(text):
        word = m.group(1).lower()
        n = _PRIORITY_WORDS.get(word)
        if word in ("to", "too") and _PREPOSITION_NEXT.match(text, m.end()):
            continue                                   # "priority to the airway": a preposition, not a number
        if n:
            last = (n, m.group(0), m.start())
    if last is None:
        return None
    n, phrase, pos = last
    burst = None
    for h in _BURST_HEADER.finditer(text):
        if h.start() < pos:
            burst = int(h.group(1))
    return n, phrase, burst


def apply_stated_priority(report: Dict[str, Any], text: str) -> Optional[str]:
    """The alert level follows a stated priority (1 RED, 2 YELLOW, 3 GREEN). Returns the note
    added to data_quality.assumptions, or None when no priority was stated. The model's own
    level stays only when nothing was stated; attribution points at the burst of the statement."""
    hit = stated_priority(text)
    if hit is None:
        return None
    n, phrase, burst = hit
    level = LEVEL_FOR_PRIORITY[n]
    before = report.get("alert_level")
    report["alert_level"] = level
    if burst is not None and isinstance(report.get("sources"), dict):
        report["sources"]["alert_level"] = burst
    note = (f"Alert level {level} from the stated '{phrase}'" +
            (f" (the model said {before})" if before and before != level else ""))
    report.setdefault("data_quality", {}).setdefault("assumptions", []).append(note)
    missing = report["data_quality"].get("missing_info")
    if isinstance(missing, list):
        report["data_quality"]["missing_info"] = [m for m in missing if "alert level" not in str(m).lower()]
    return note


_MALE_WORDS = re.compile(r"\b(?:he|his|him|himself)\b", re.I)
_FEMALE_WORDS = re.compile(r"\b(?:she|her|hers|herself)\b", re.I)


def infer_sex(text: str) -> Optional[Tuple[str, str]]:
    """Patient sex from pronouns, when they are decisive: (sex, note) or None.

    The radio side rarely says "male" or "female" but talks about the patient as
    "he" or "she" throughout. The rule is conservative, since a report also mentions
    family members and crew: at least two pronouns of one sex, and more than twice
    as many as of the other. The note goes into data_quality.assumptions so the ED
    can see the sex was inferred, not heard.
    """
    male = len(_MALE_WORDS.findall(text))
    female = len(_FEMALE_WORDS.findall(text))
    if male >= 2 and male > 2 * female:
        return "Male", f"Sex inferred from pronouns ({male} he/his/him, {female} she/her)"
    if female >= 2 and female > 2 * male:
        return "Female", f"Sex inferred from pronouns ({female} she/her, {male} he/his/him)"
    return None


def clean_sources(value: Any) -> Dict[str, int]:
    """The model's "sources" as {field path: burst number}; anything else is dropped.
    An absent or malformed entry is an empty dict, never an error: attribution is additive."""
    if not isinstance(value, dict):
        return {}
    out: Dict[str, int] = {}
    for path, n in value.items():
        if isinstance(n, bool):
            continue
        if isinstance(n, (int, float)) or (isinstance(n, str) and n.strip().isdigit()):
            out[str(path)] = int(n)
    return out


def parse_report(text: str) -> Dict[str, Any]:
    """Parse and validate the LLM's JSON. Null or missing parts take their empty shape, so a
    sparse reply to a short transcript is still a complete report."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PipelineError(f"The AI returned malformed JSON ({e.msg}). Try again.") from e
    if not isinstance(data, dict):
        raise PipelineError("The AI returned JSON that is not an object.")
    data = scrub_placeholders(data)
    required = REQUIRED_FIELDS - ({"handoff_summary"} if facts_only() else set())   # facts-only: the pipeline composes it
    missing = sorted(required - data.keys())
    if missing:
        raise PipelineError(f"The AI left out required fields: {', '.join(missing)}.")
    if not isinstance(data.get("patient"), (dict, type(None))):
        raise PipelineError("The AI's 'patient' field is not an object.")
    if not isinstance(data.get("data_quality"), dict):
        # the model said nothing about its own confidence; MEDIUM, as before
        data["data_quality"] = {"confidence": "MEDIUM"}
    data = fill_blanks(blank_report(), data)
    data["sources"] = clean_sources(data.get("sources"))

    data["alert_type"] = str(data.get("alert_type") or "UNCLEAR").upper().strip()
    if data["alert_type"] not in ALERT_TYPES:
        logger.warning("Unexpected alert_type %r", data["alert_type"])
    level = data.get("alert_level")
    level = str(level).upper().strip() if level not in (None, "") else None
    if level is not None and level not in ALERT_LEVELS:
        raise PipelineError(f"The AI gave an invalid alert level: {data.get('alert_level')!r}.")
    data["alert_level"] = level    # null = no priority was stated and none is invented (any alert type)

    dq = data["data_quality"]
    confidence = str(dq.get("confidence") or "").upper().strip()
    dq["confidence"] = confidence if confidence in CONFIDENCE_LEVELS else "LOW"
    if data["alert_type"] == "UNCLEAR":
        dq["confidence"] = "LOW"
        if not dq["missing_info"]:
            dq["missing_info"] = unstated(data)
    if data["alert_level"] is None and not any("alert level" in str(m).lower() for m in dq["missing_info"]):
        dq["missing_info"].append("alert level")
    return data


_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                 "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
                 "twenty": 20, "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60}
_ETA_CUES = {"eta", "minute", "minutes", "min", "mins", "away", "arrival", "arriving", "landing", "enroute", "route"}


def eta_supported(transcript: str, eta: Any) -> bool:
    """True when the ETA number is spoken as a time somewhere in the transcript: within eight words of
    "ETA", "minutes", "away" or the like, or as "<n> out" (not "6 out of 10"). "12-lead" alone does not
    make an ETA of 12, nor does a dose or an age."""
    try:
        n = int(float(eta))
    except (TypeError, ValueError):
        return False
    tokens = [t for t in re.split(r"[^\w]+", transcript.lower()) if t]
    tokens = [str(_NUMBER_WORDS.get(t, t)) for t in tokens]
    for i, t in enumerate(tokens):
        if t != str(n):
            continue
        if _ETA_CUES & set(tokens[max(0, i - 8): i + 9]):
            return True
        if tokens[i + 1: i + 2] == ["out"] and tokens[i + 2: i + 3] != ["of"]:
            return True
    return False


def parse_guidance(text: str) -> Optional[List[dict]]:
    """Accept {"courses_of_action": [...]} or a bare list. Returns None if unusable."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        data = data.get("courses_of_action") or data.get("actions")
    if not isinstance(data, list):
        return None
    items = [g for g in data if isinstance(g, dict) and g.get("action")]
    return items or None



def compact_output() -> bool:
    """HANDOFF_COMPACT_OUTPUT=1 asks the writer to omit empty fields (fill_blanks restores them); a latency lever under test."""
    return os.environ.get("HANDOFF_COMPACT_OUTPUT", "0").strip().lower() in ("1", "true", "yes", "on")


def facts_only() -> bool:
    """HANDOFF_FACTS_ONLY=1: the writer produces the clinical facts only and render_prose() composes the summary,
    the missing-information list and the preparation lists from them (the prose was 61 % of the output tokens)."""
    return os.environ.get("HANDOFF_FACTS_ONLY", "0").strip().lower() in ("1", "true", "yes", "on")


_PREP = {  # what a receiving facility readies for each alert type, when the crew asked for nothing more specific
    "STEMI": (["Cath team", "12-lead on arrival"], ["Activate cath lab"]),
    "STROKE": (["Stroke team", "CT scanner"], ["Stroke alert", "Confirm last known well"]),
    "TRAUMA": (["Trauma team", "Blood products available"], ["Trauma activation"]),
    "SEPSIS": (["Sepsis bundle", "Lactate and cultures"], ["Early antibiotics"]),
    "CARDIAC_ARREST": (["Resuscitation team", "Airway equipment"], ["Resuscitation bay ready"]),
    "PEDIATRIC": (["Pediatric team", "Weight-based dosing chart"], []),
    "OBSTETRIC": (["OB team"], []),
}


def render_prose(report: Dict[str, Any]) -> None:
    """Compose handoff_summary, data_quality and the preparation lists from the facts the writer stated.

    Nothing here adds a clinical fact: every word comes from a field the writer filled or from the alert type's
    standing preparation table. The rules that run afterwards may append to missing_info and assumptions.
    """
    p = report.get("patient") or {}
    bits = []
    if p.get("age") is not None:
        bits.append(f"{p['age']} {p.get('age_unit') or 'years'}".replace(" years", "-year-old") if (p.get("age_unit") or "years") == "years" else f"{p['age']}-month-old")
    if p.get("sex"):
        bits.append(str(p["sex"]).lower())
    who = " ".join(bits) or "patient"
    parts = [who]
    if report.get("chief_complaint"):
        parts.append(str(report["chief_complaint"]).rstrip("."))
    alert = report.get("alert_type") or "UNCLEAR"
    if alert != "UNCLEAR":
        parts.append(f"{alert.replace('_', ' ').lower()} alert" + (f", {report['alert_level']}" if report.get("alert_level") else ""))
    if report.get("eta_minutes") is not None:
        parts.append(f"ETA {report['eta_minutes']} min")
    hr = report.setdefault("hospital_recommendations", {})
    hs = report.setdefault("handoff_summary", {})
    hs["one_liner"] = ", ".join(parts) + "."
    concerns = [c for c in [report.get("chief_complaint"), hr.get("activation_requested")] if c]
    hs["key_concerns"] = concerns[:3]
    hs["pending_needs"] = [hr["activation_requested"]] if hr.get("activation_requested") else []
    prep = _PREP.get(alert)
    if prep and not hr.get("resources_to_prepare"):
        hr["resources_to_prepare"] = list(prep[0])
    if prep and not hr.get("time_critical_actions"):
        hr["time_critical_actions"] = list(prep[1])
    hr.setdefault("resources_to_prepare", []); hr.setdefault("time_critical_actions", [])
    dq = report.setdefault("data_quality", {})
    vitals_present = any(not _is_empty(v) for v in (report.get("vital_signs") or {}).values())
    if alert != "UNCLEAR" and report.get("chief_complaint") and vitals_present and report.get("eta_minutes") is not None:
        dq["confidence"] = "HIGH"
    elif alert != "UNCLEAR" and report.get("chief_complaint"):
        dq["confidence"] = "MEDIUM"
    else:
        dq["confidence"] = "LOW"
    dq["missing_info"] = unstated(report)
    dq.setdefault("assumptions", []).append("summary, missing-information list and preparation lists composed from the stated facts (facts-only writer)")


def rules_enabled() -> bool:
    """HANDOFF_REPORT_RULES=0 turns the post-model grounding rules off (ablation runs only)."""
    return os.environ.get("HANDOFF_REPORT_RULES", "1").strip().lower() not in ("0", "false", "no", "off")


class ReportPipeline:
    def __init__(self, llm: LLMClient, stop_phrase: str = "",
                 lexicon_store: Any = None, textbook_rag: Any = None,
                 on_stage: Optional[Callable[[str], None]] = None, attribution: bool = False,
                 rules: Optional[bool] = None):
        self.llm = llm
        self.stop_phrase = stop_phrase
        self.attribution = attribution     # ask for burst sources (incident sessions); off on the live path
        # the grounding rules after the model (history defaults, claim check, alert word, stated priority,
        # sex from pronouns, ETA guard); off only for the ablation runs, HANDOFF_REPORT_RULES=0
        self.rules = rules_enabled() if rules is None else bool(rules)
        self.lexicon_store = lexicon_store
        self.textbook_rag = textbook_rag
        self.on_stage = on_stage or (lambda _msg: None)
        # what the last run gave the model, for `cli --show-context` and the tests
        self.last_context = ""        # lexicon terms + flagged suspects block
        self.last_reference = ""      # textbook excerpts ("" when no store / nothing found)

    def run(self, transcript: str, source: str = "text",
            timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        timings = dict(timings or {})
        raw = (transcript or "").strip()
        text = strip_phrase(raw, self.stop_phrase) if self.stop_phrase else raw
        if len(text) > MAX_REPORT_CHARS:
            raise PipelineError(f"The report is too long ({len(text)} characters, limit {MAX_REPORT_CHARS}).")
        if not text.split():
            return self._empty_report(raw, source, timings)

        # 1. clean the transcript
        self.on_stage("Checking medical terms")
        t0 = time.perf_counter()
        fix = correct_transcript(text)
        timings["correct_s"] = round(time.perf_counter() - t0, 3)

        # 2. build the prompt
        context = ""
        if self.lexicon_store is not None:
            context = self.lexicon_store.build_context(fix.text, top_k=25)
        suspects = format_suspects_for_llm(fix.flagged)
        if suspects:
            context = f"{context}\n{suspects}" if context else suspects
        self.last_context, self.last_reference = context, ""
        prompt = extraction_prompt(self.attribution, compact=compact_output(), facts_only=facts_only()).format(ems_report=fix.text, lexicon_context=context)

        # 3. extract
        t0 = time.perf_counter()
        try:
            if hasattr(self.llm, "resolve_model"):
                self.llm.resolve_model()   # name the model actually used, not a retired one
            self.on_stage(f"Writing the handoff ({self.llm.provider} · {self.llm.model})")
            if hasattr(self.llm, "on_wait"):
                self.llm.on_wait = self.on_stage   # "Model server restarting ..." reaches the app as a stage
            reply = self.llm.complete_json(prompt, temperature=0.0)
        except LLMError as e:
            raise PipelineError(str(e)) from e
        report = parse_report(reply)
        timings["extract_s"] = round(time.perf_counter() - t0, 3)
        usage = getattr(self.llm, "last_usage", None)
        if usage:
            timings.update(usage)      # tokens_in, tokens_out, tok_s: says where a slow model spent its time
        if facts_only():
            render_prose(report)
        report["data_quality"]["assumptions"].extend(assumption_notes(fix))
        if self.rules:
            self._apply_rules(report, fix.text)
        else:
            report["data_quality"]["assumptions"].append("Grounding rules off (HANDOFF_REPORT_RULES=0): the model's answer stands as given")

        # 4. optional textbook guidance (never fails the report)
        guidance = self._guidance(report, timings)
        if guidance:
            report["clinical_guidance"] = guidance

        report["_meta"] = self._meta(source, fix.text, raw, timings,
                                     corrections=[{"from": c.original, "to": c.replacement} for c in fix.corrections],
                                     flagged=[{"word": f.token, "maybe": f.best_guess} for f in fix.flagged])
        return report

    def _apply_rules(self, report: Dict[str, Any], text: str) -> None:
        """The grounding rules after the model: what the transcript does not say is cleared or noted."""
        fix_text = text
        report["data_quality"]["assumptions"].extend(ground_history_defaults(report, fix_text))
        report["data_quality"]["assumptions"].extend(ground_claims(report, fix_text))
        note = ground_activation(report, fix_text)
        if note:
            report["data_quality"]["assumptions"].append(note)
        if report["alert_type"] == "UNCLEAR":
            hit = alert_from_words(fix_text)
            if hit:
                report["alert_type"], word = hit
                report["data_quality"]["assumptions"].append(
                    f"Alert type {hit[0]} from the word '{word}' in the transcript; the model answered UNCLEAR")
        apply_stated_priority(report, fix_text)      # a stated priority sets the level, whatever the model chose
        if report["patient"].get("sex") is None:
            # the prompt asks the model to read pronouns; a small model often does not
            guess = infer_sex(fix_text)
            if guess:
                report["patient"]["sex"], note = guess
                report["data_quality"]["assumptions"].append(note)

        eta = report.get("eta_minutes")
        if eta is not None and not eta_supported(fix_text, eta):
            # a number that is not a time (the "12" of "12-lead", a dose, an age) is not an ETA
            logger.info("ETA %r is not stated as a time in the transcript; cleared", eta)
            report["eta_minutes"] = None
            report["data_quality"]["missing_info"].append(f"ETA (the model's {eta} is not stated as a time)")

    def _empty_report(self, raw: str, source: str, timings: Dict[str, float]) -> Dict[str, Any]:
        """Zero words came through: a complete report that says so. The model is not called."""
        self.on_stage("Nothing to extract: the transcript is empty")
        report = blank_report()
        report["handoff_summary"]["one_liner"] = "No report content was received: the transcript is empty."
        report["data_quality"]["missing_info"] = ["Transcript is empty (no words)"] + unstated(report)
        report["_meta"] = self._meta(source, "", raw, timings, corrections=[], flagged=[],
                                     note="empty transcript; the model was not called")
        return report

    def _meta(self, source: str, text: str, raw: str, timings: Dict[str, float], **extra: Any) -> Dict[str, Any]:
        meta = {
            "source": source,
            "transcript": text,
            "raw_transcript": raw,
            "provider": self.llm.provider,
            "model": self.llm.model,
            "timings": timings,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        recovery = getattr(self.llm, "last_recovery", None)
        if recovery:
            meta["model_server"] = dict(recovery)   # the report came after Ollama's model server restarted
        meta.update(extra)
        return meta

    def _guidance(self, report: Dict[str, Any], timings: Dict[str, float]) -> Optional[List[dict]]:
        rag = self.textbook_rag
        if rag is None or not getattr(rag, "available", False):
            return None
        self.on_stage("Finding textbook guidance")
        t0 = time.perf_counter()
        try:
            reference = rag.guidance_context(report, top_k=6)
            self.last_reference = reference or ""
            if not reference:
                return None
            prompt = GUIDANCE_PROMPT.format(handoff_json=json.dumps(report, indent=2),
                                            reference=reference)
            guidance = parse_guidance(self.llm.complete_json(prompt, temperature=0.1))
        except Exception as e:  # noqa: BLE001 - guidance is optional
            logger.warning("Textbook guidance skipped: %s", e)
            guidance = None
        timings["guidance_s"] = round(time.perf_counter() - t0, 3)
        return guidance
