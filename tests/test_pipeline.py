"""Report pipeline: text -> validated report, and every way it can fail."""

import json

import pytest

from ems_handoff.lexicon_rag import LexiconStore
from ems_handoff.llm import LLMError
from ems_handoff.pipeline import REQUIRED_FIELDS, PipelineError, ReportPipeline, parse_guidance, parse_report
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
    (json.dumps(report_dict(alert_level=None)), "invalid alert level"),   # a named alert type needs a priority
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


# What a model honestly returns for a few words: whole sections null, lists null.
SPARSE_REPLY = json.dumps({
    "alert_type": "UNCLEAR", "alert_level": None, "patient": None, "chief_complaint": None,
    "vital_signs": None, "assessment_findings": None,
    "history": {"past_medical": None, "medications": None, "allergies": None, "code_status": None},
    "handoff_summary": None,
    "data_quality": {"confidence": "MEDIUM", "missing_info": None, "assumptions": None},
})


def assert_complete_unclear(report):
    assert REQUIRED_FIELDS <= set(report)
    assert report["alert_type"] == "UNCLEAR" and report["alert_level"] is None
    assert report["data_quality"]["confidence"] == "LOW"
    missing = report["data_quality"]["missing_info"]
    assert "patient age" in missing and "vital signs" in missing and "chief complaint" in missing
    assert report["patient"] == {"age": None, "age_unit": None, "sex": None}
    assert report["vital_signs"]["heart_rate"] is None and report["chief_complaint"] is None
    assert report["history"]["medications"] == [] and report["ems_interventions"] == []
    assert report["handoff_summary"]["key_concerns"] == [] and report["data_quality"]["assumptions"] == []


def test_three_word_transcript_is_processed_into_a_complete_unclear_report():
    report, stages, pipe = run("How old? Sorry.", FakeLLM([SPARSE_REPLY]))      # CS015, verbatim
    assert_complete_unclear(report)
    assert "How old? Sorry." in pipe.llm.prompts[0]        # the model was asked, however short
    assert report["_meta"]["transcript"] == "How old? Sorry."
    assert stages[1].startswith("Writing the handoff")


def test_one_word_transcript_is_processed():
    report, _, pipe = run("Copy.", FakeLLM([SPARSE_REPLY]))
    assert_complete_unclear(report)
    assert len(pipe.llm.prompts) == 1


@pytest.mark.parametrize("text", ["", "   \n", "confirm report ready"])   # the stop phrase alone leaves no words
def test_empty_transcript_gets_a_report_that_says_so_without_calling_the_llm(text):
    llm = FakeLLM()
    report, stages, _ = run(text, llm)
    assert_complete_unclear(report)
    assert report["data_quality"]["missing_info"][0] == "Transcript is empty (no words)"
    assert "empty" in report["handoff_summary"]["one_liner"]
    assert report["_meta"]["transcript"] == "" and report["_meta"]["note"].startswith("empty transcript")
    assert llm.prompts == [] and stages == ["Nothing to extract: the transcript is empty"]


def test_unclear_alert_forces_low_confidence_and_lists_what_is_missing():
    reply = report_dict(alert_type="UNCLEAR", alert_level="GREEN", eta_minutes=None,
                        patient={"age": None, "age_unit": None, "sex": "Male"},
                        data_quality={"confidence": "HIGH", "missing_info": [], "assumptions": []})
    report, _, _ = run(SAMPLE_STEMI, FakeLLM([json.dumps(reply)]))
    assert report["alert_level"] == "GREEN"                    # a level the model did give is kept
    assert report["data_quality"]["confidence"] == "LOW"
    assert report["data_quality"]["missing_info"] == ["patient age", "ETA"]   # filled from the null fields


def test_null_lists_and_sections_from_the_model_become_empty_shapes():
    # the CS022 / S007_01 baseline failure: data_quality.assumptions was null and .extend crashed
    reply = report_dict(ems_interventions=None, hospital_recommendations=None,
                        history={"past_medical": None, "medications": "metformin", "allergies": None, "code_status": None},
                        data_quality={"confidence": "MEDIUM", "missing_info": None, "assumptions": None})
    report, _, _ = run(SAMPLE_SEPSIS.replace("furosemide", "ferosemide"), FakeLLM([json.dumps(reply)]))
    assert report["alert_type"] == "STEMI" and report["data_quality"]["confidence"] == "MEDIUM"
    assert report["data_quality"]["missing_info"] == []
    assert any("ferosemide" in a for a in report["data_quality"]["assumptions"])
    assert report["ems_interventions"] == [] and report["history"]["medications"] == ["metformin"]
    assert report["history"]["allergies"] == [] and report["hospital_recommendations"]["resources_to_prepare"] == []


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


def test_pipeline_exposes_the_context_it_gave_the_model():
    report, _, pipe = run(SAMPLE_STEMI, textbook_rag=FakeTextbook(),
                          llm=FakeLLM([json.dumps(report_dict()), json.dumps({"courses_of_action": []})]))
    assert "MEDICAL LEXICON" in pipe.last_context and "Unit 41" not in pipe.last_context
    assert pipe.last_reference.startswith("CLINICAL REFERENCE")
    # the same lexicon block is inside the prompt the model received
    assert pipe.last_context in pipe.llm.prompts[0]


def test_schema_placeholders_echoed_by_a_small_model_become_null():
    bad = report_dict()
    bad["vital_signs"]["gcs"] = "Total and breakdown if mentioned"
    bad["vital_signs"]["temperature"] = "if mentioned"
    bad["history"]["medications"] = ["List of medications", "aspirin"]
    bad["chief_complaint"] = "Chest pain"
    report = parse_report(json.dumps(bad))
    assert report["vital_signs"]["gcs"] is None and report["vital_signs"]["temperature"] is None
    assert report["history"]["medications"] == ["aspirin"]
    assert report["chief_complaint"] == "Chest pain"


def test_token_stats_are_kept_in_timings():
    llm = FakeLLM()
    llm.last_usage = {"tokens_in": 2100, "tokens_out": 850, "tok_s": 10.7}
    report, _, _ = run(SAMPLE_STEMI, llm)
    t = report["_meta"]["timings"]
    assert t["tokens_out"] == 850 and t["tok_s"] == 10.7 and "extract_s" in t
