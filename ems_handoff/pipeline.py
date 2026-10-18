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
          -> validate              (required fields, alert level, defaults)
          -> textbook guidance     (optional, only if database/ exists)

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

MIN_REPORT_CHARS = 20
MAX_REPORT_CHARS = 8000

REQUIRED_FIELDS = {
    "alert_type", "alert_level", "patient", "chief_complaint",
    "vital_signs", "assessment_findings", "history", "handoff_summary",
}
OPTIONAL_DEFAULTS: Dict[str, Any] = {
    "eta_minutes": None,
    "mechanism_or_onset": "Unknown",
    "ems_interventions": [],
    "hospital_recommendations": {},
    "data_quality": {"confidence": "MEDIUM", "missing_info": [], "assumptions": []},
}
ALERT_TYPES = {"STEMI", "STROKE", "TRAUMA", "SEPSIS", "CARDIAC_ARREST", "PEDIATRIC",
               "OBSTETRIC", "MEDICAL", "PSYCHIATRIC", "ENDOCRINE"}
ALERT_LEVELS = {"RED", "YELLOW", "GREEN"}


class PipelineError(RuntimeError):
    """A report could not be produced. The message is shown to the user."""


def parse_report(text: str) -> Dict[str, Any]:
    """Parse and validate the LLM's JSON. Fills optional defaults (deep copies)."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise PipelineError(f"The AI returned malformed JSON ({e.msg}). Try again.") from e
    if not isinstance(data, dict):
        raise PipelineError("The AI returned JSON that is not an object.")
    missing = sorted(REQUIRED_FIELDS - data.keys())
    if missing:
        raise PipelineError(f"The AI left out required fields: {', '.join(missing)}.")
    for key, default in OPTIONAL_DEFAULTS.items():
        if key not in data or data[key] is None and default is not None:
            data[key] = copy.deepcopy(default)
    level = str(data.get("alert_level", "")).upper().strip()
    if level not in ALERT_LEVELS:
        raise PipelineError(f"The AI gave an invalid alert level: {data.get('alert_level')!r}.")
    data["alert_level"] = level
    data["alert_type"] = str(data.get("alert_type", "MEDICAL")).upper().strip()
    if data["alert_type"] not in ALERT_TYPES:
        logger.warning("Unexpected alert_type %r", data["alert_type"])
    if not isinstance(data.get("patient"), dict):
        raise PipelineError("The AI's 'patient' field is not an object.")
    if not isinstance(data.get("data_quality"), dict):
        data["data_quality"] = copy.deepcopy(OPTIONAL_DEFAULTS["data_quality"])
    data["data_quality"].setdefault("assumptions", [])
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

    def run(self, transcript: str, source: str = "text",
            timings: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        timings = dict(timings or {})
        raw = (transcript or "").strip()
        text = strip_phrase(raw, self.stop_phrase) if self.stop_phrase else raw
        if len(text) < MIN_REPORT_CHARS:
            raise PipelineError("The report is too short to process. Please say or paste more detail.")
        if len(text) > MAX_REPORT_CHARS:
            raise PipelineError(f"The report is too long ({len(text)} characters, limit {MAX_REPORT_CHARS}).")

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
        prompt = EXTRACTION_PROMPT.format(ems_report=fix.text, lexicon_context=context)

        # 3. extract
        if hasattr(self.llm, "resolve_model"):
            self.llm.resolve_model()   # name the model actually used, not a retired one
        self.on_stage(f"Writing the handoff ({self.llm.provider} · {self.llm.model})")
        t0 = time.perf_counter()
        try:
            reply = self.llm.complete_json(prompt, temperature=0.0)
        except LLMError as e:
            raise PipelineError(str(e)) from e
        report = parse_report(reply)
        timings["extract_s"] = round(time.perf_counter() - t0, 3)
        report["data_quality"]["assumptions"].extend(assumption_notes(fix))

        # 4. optional textbook guidance (never fails the report)
        guidance = self._guidance(report, timings)
        if guidance:
            report["clinical_guidance"] = guidance

        report["_meta"] = {
            "source": source,
            "transcript": fix.text,
            "raw_transcript": raw,
            "corrections": [{"from": c.original, "to": c.replacement} for c in fix.corrections],
            "flagged": [{"word": f.token, "maybe": f.best_guess} for f in fix.flagged],
            "provider": self.llm.provider,
            "model": self.llm.model,
            "timings": timings,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return report

    def _guidance(self, report: Dict[str, Any], timings: Dict[str, float]) -> Optional[List[dict]]:
        rag = self.textbook_rag
        if rag is None or not getattr(rag, "available", False):
            return None
        self.on_stage("Finding textbook guidance")
        t0 = time.perf_counter()
        try:
            reference = rag.guidance_context(report, top_k=6)
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
