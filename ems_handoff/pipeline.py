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
    dq = report.get("data_quality") or {}
    missing = dq.get("missing_info")
    if isinstance(missing, list):
        for label, key in (("allergies", "allergies"), ("code status", "code_status")):
            if not hist.get(key) and not any(label in str(m).lower() for m in missing):
                missing.append(label)
    return notes


# Alert words a radio fragment can carry. UNCLEAR is for a transcript with none of them.
_ALERT_WORDS = [
    ("CARDIAC_ARREST", re.compile(r"cardiac arrest|\bcpr\b|pulseless|asystole|v-?fib\b|ventricular fibrillation|\brosc\b", re.I)),
    ("STEMI", re.compile(r"\bstemi\b|\bst[- ]elevation", re.I)),
    ("STROKE", re.compile(r"\bstroke\b|\bcva\b|brain bleed|\btia\b|facial droop|hemorrhagic", re.I)),
    ("TRAUMA", re.compile(r"\btrauma\b|\bmvc\b|\bmva\b|\bgsw\b|gunshot|\bstab(?:bed|bing|bed)?\b|fall from|fell (?:from|down|off)|rollover|pedestrian struck|\bfracture", re.I)),
    ("SEPSIS", re.compile(r"\bsepsis\b|\bseptic\b", re.I)),
]


def alert_from_words(text: str) -> Optional[Tuple[str, str]]:
    """(alert_type, the word that names it) when the transcript states an alert word, else None.

    "This is Ambulance 721 coming to, what is it, trauma? ... priority 2" (S038_01) must be
    TRAUMA however short. The model is asked the same; this is the check behind it, used
    only when the model answered UNCLEAR. A type the model did name is never overridden."""
    for alert_type, pattern in _ALERT_WORDS:
        m = pattern.search(text)
        if m:
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
    missing = sorted(REQUIRED_FIELDS - data.keys())
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


class ReportPipeline:
    def __init__(self, llm: LLMClient, stop_phrase: str = "",
                 lexicon_store: Any = None, textbook_rag: Any = None,
                 on_stage: Optional[Callable[[str], None]] = None, attribution: bool = False):
        self.llm = llm
        self.stop_phrase = stop_phrase
        self.attribution = attribution     # ask for burst sources (incident sessions); off on the live path
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
        prompt = extraction_prompt(self.attribution).format(ems_report=fix.text, lexicon_context=context)

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
        report["data_quality"]["assumptions"].extend(assumption_notes(fix))
        report["data_quality"]["assumptions"].extend(ground_history_defaults(report, fix.text))
        if report["alert_type"] == "UNCLEAR":
            hit = alert_from_words(fix.text)
            if hit:
                report["alert_type"], word = hit
                report["data_quality"]["assumptions"].append(
                    f"Alert type {hit[0]} from the word '{word}' in the transcript; the model answered UNCLEAR")
        apply_stated_priority(report, fix.text)      # a stated priority sets the level, whatever the model chose
        if report["patient"].get("sex") is None:
            # the prompt asks the model to read pronouns; a small model often does not
            guess = infer_sex(fix.text)
            if guess:
                report["patient"]["sex"], note = guess
                report["data_quality"]["assumptions"].append(note)

        # 4. optional textbook guidance (never fails the report)
        guidance = self._guidance(report, timings)
        if guidance:
            report["clinical_guidance"] = guidance

        report["_meta"] = self._meta(source, fix.text, raw, timings,
                                     corrections=[{"from": c.original, "to": c.replacement} for c in fix.corrections],
                                     flagged=[{"word": f.token, "maybe": f.best_guess} for f in fix.flagged])
        return report

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
