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
import time
from typing import Any, Callable, Dict, List, Optional

from .llm import LLMClient, LLMError
from .phrase import strip_phrase
from .prompt import EXTRACTION_PROMPT, GUIDANCE_PROMPT
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
        "history": {"past_medical": [], "medications": [], "allergies": [], "code_status": None},
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
    ("ETA", "eta_minutes"), ("vital signs", "vital_signs"),
    ("assessment findings", "assessment_findings"), ("EMS interventions", "ems_interventions"),
    ("medical history", "history"),
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

    data["alert_type"] = str(data.get("alert_type") or "UNCLEAR").upper().strip()
    if data["alert_type"] not in ALERT_TYPES:
        logger.warning("Unexpected alert_type %r", data["alert_type"])
    level = data.get("alert_level")
    level = str(level).upper().strip() if level is not None else None
    if level is None and data["alert_type"] == "UNCLEAR":
        pass                       # nothing said, no priority invented
    elif level not in ALERT_LEVELS:
        raise PipelineError(f"The AI gave an invalid alert level: {data.get('alert_level')!r}.")
    data["alert_level"] = level

    dq = data["data_quality"]
    confidence = str(dq.get("confidence") or "").upper().strip()
    dq["confidence"] = confidence if confidence in CONFIDENCE_LEVELS else "LOW"
    if data["alert_type"] == "UNCLEAR":
        dq["confidence"] = "LOW"
        if not dq["missing_info"]:
            dq["missing_info"] = unstated(data)
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
                 on_stage: Optional[Callable[[str], None]] = None):
        self.llm = llm
        self.stop_phrase = stop_phrase
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
        prompt = EXTRACTION_PROMPT.format(ems_report=fix.text, lexicon_context=context)

        # 3. extract
        t0 = time.perf_counter()
        try:
            if hasattr(self.llm, "resolve_model"):
                self.llm.resolve_model()   # name the model actually used, not a retired one
            self.on_stage(f"Writing the handoff ({self.llm.provider} · {self.llm.model})")
            reply = self.llm.complete_json(prompt, temperature=0.0)
        except LLMError as e:
            raise PipelineError(str(e)) from e
        report = parse_report(reply)
        timings["extract_s"] = round(time.perf_counter() - t0, 3)
        usage = getattr(self.llm, "last_usage", None)
        if usage:
            timings.update(usage)      # tokens_in, tokens_out, tok_s: says where a slow model spent its time
        report["data_quality"]["assumptions"].extend(assumption_notes(fix))

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
