"""Report pipeline: text -> validated report, and every way it can fail."""

import json

import pytest

from ems_handoff.lexicon_rag import LexiconStore
from ems_handoff.llm import LLMError
from ems_handoff.pipeline import PipelineError, ReportPipeline, parse_guidance, parse_report
from ems_handoff.samples import SAMPLE_SEPSIS, SAMPLE_STEMI
from fakes import FakeLLM, report_dict

PHRASE = "confirm report ready"


def run(text, llm=None, **kw):
    stages = []
    pipe = ReportPipeline(llm or FakeLLM(), stop_phrase=PHRASE, lexicon_store=LexiconStore(),
                          on_stage=stages.append, **kw)
    return pipe.run(text, source="test"), stages, pipe


def test_sample_report_end_to_end():
    report, stages, pipe = run(SAMPLE_STEMI)
    assert report["alert_type"] == "STEMI" and report["alert_level"] == "RED"
    assert report["_meta"]["provider"] == "groq"
    assert report["_meta"]["source"] == "test"
    assert stages[0] == "Checking medical terms" and stages[1].startswith("Writing the handoff")
    prompt = pipe.llm.prompts[0]
    assert "MEDICAL LEXICON" in prompt and "Unit 41" in prompt


def test_stop_phrase_is_removed_before_the_llm_sees_it():
    report, _, pipe = run(SAMPLE_STEMI.strip() + " Confirm report ready.")
    assert "confirm report ready" not in pipe.llm.prompts[0].lower().split("ems radio report:")[1]
    assert report["_meta"]["raw_transcript"].lower().endswith("confirm report ready.")


def test_corrections_become_visible_assumptions():
    text = SAMPLE_SEPSIS.replace("furosemide", "ferosemide")
    report, _, pipe = run(text)
    assert "furosemide" in pipe.llm.prompts[0]
    assert any("ferosemide" in a for a in report["data_quality"]["assumptions"])
    assert {"from": "ferosemide", "to": "furosemide"} in report["_meta"]["corrections"]


def test_optional_fields_get_independent_defaults():
    minimal = report_dict()
    for key in ("ems_interventions", "hospital_recommendations", "data_quality", "eta_minutes"):
        minimal.pop(key)
    a = parse_report(json.dumps(minimal))
    b = parse_report(json.dumps(minimal))
    a["data_quality"]["assumptions"].append("x")
    assert b["data_quality"]["assumptions"] == []   # no shared mutable defaults
    assert a["ems_interventions"] == [] and a["eta_minutes"] is None


@pytest.mark.parametrize("reply,message", [
    ("not json at all", "malformed JSON"),
    ("[1, 2]", "not an object"),
    (json.dumps({"alert_type": "STEMI"}), "left out required fields"),
    (json.dumps(report_dict(alert_level="PURPLE")), "invalid alert level"),
    (json.dumps(report_dict(patient="67M")), "'patient' field"),
])
def test_bad_llm_output_gives_a_readable_error(reply, message):
    with pytest.raises(PipelineError, match=message):
        run(SAMPLE_STEMI, FakeLLM([reply]))


def test_lowercase_alert_level_is_normalised():
    report, _, _ = run(SAMPLE_STEMI, FakeLLM([json.dumps(report_dict(alert_level="red"))]))
    assert report["alert_level"] == "RED"


def test_llm_failure_becomes_pipeline_error():
    with pytest.raises(PipelineError, match="cannot reach"):
        run(SAMPLE_STEMI, FakeLLM([LLMError("groq failed after 3 tries: cannot reach groq")]))


def test_too_short_report_is_rejected_without_calling_the_llm():
    llm = FakeLLM()
    with pytest.raises(PipelineError, match="too short"):
        run("confirm report ready", llm)
    assert llm.prompts == []


class FakeTextbook:
    available = True

    def guidance_context(self, report, top_k=6):
        return "CLINICAL REFERENCE:\n  [Emergency Care (p. 211)]\n    Give aspirin early."


def test_textbook_guidance_is_added_when_available():
    guidance = json.dumps({"courses_of_action": [{"action": "Activate cath lab", "rationale": "r",
                                                  "source": "Emergency Care (p. 211)"}]})
    llm = FakeLLM([json.dumps(report_dict()), guidance])
    report, stages, _ = run(SAMPLE_STEMI, llm, textbook_rag=FakeTextbook())
    assert report["clinical_guidance"][0]["action"] == "Activate cath lab"
    assert "Finding textbook guidance" in stages


def test_broken_guidance_never_breaks_the_report():
    llm = FakeLLM([json.dumps(report_dict()), "garbage"])
    report, _, _ = run(SAMPLE_STEMI, llm, textbook_rag=FakeTextbook())
    assert "clinical_guidance" not in report and report["alert_type"] == "STEMI"


def test_parse_guidance_accepts_list_or_object():
    item = {"action": "a", "source": "s"}
    assert parse_guidance(json.dumps([item])) == [item]
    assert parse_guidance(json.dumps({"courses_of_action": [item]})) == [item]
    assert parse_guidance(json.dumps({"courses_of_action": []})) is None
    assert parse_guidance("nope") is None


def test_groq_whisper_falls_back_to_an_offered_model():
    from types import SimpleNamespace
    import numpy as np
    from ems_handoff.config import Settings
    from ems_handoff.transcribe import GroqTranscriber
    used = []
    client = SimpleNamespace(
        models=SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id="whisper-large-v3")])),
        audio=SimpleNamespace(transcriptions=SimpleNamespace(
            create=lambda **kw: used.append(kw["model"]) or SimpleNamespace(text="hello"))))
    t = GroqTranscriber(Settings(groq_api_key="k"), client=client)
    assert t.transcribe(np.zeros(16000, np.float32)) == "hello" and used == ["whisper-large-v3"]
