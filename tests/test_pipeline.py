"""Report pipeline: text -> validated report, and every way it can fail."""

import json

import pytest

from ems_handoff.lexicon_rag import LexiconStore
from ems_handoff.llm import LLMError
from ems_handoff.pipeline import REQUIRED_FIELDS, PipelineError, ReportPipeline, eta_supported, parse_guidance, parse_report
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


def test_attribution_text_is_off_unless_asked():
    from ems_handoff.prompt import EXTRACTION_PROMPT, extraction_prompt
    _, _, pipe = run(SAMPLE_STEMI)
    assert '"sources"' not in pipe.llm.prompts[0] and "Burst" not in pipe.llm.prompts[0]
    assert extraction_prompt(False) is EXTRACTION_PROMPT
    _, _, pipe = run(SAMPLE_STEMI, attribution=True)
    prompt = pipe.llm.prompts[0]
    assert '"sources": {"field path such as patient.age' in prompt and "[Burst 3" in prompt
    assert prompt.rstrip().endswith("/no_think") and prompt.count("/no_think") == 1


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


def test_null_alert_level_is_kept_for_any_alert_type():
    # gpt-oss-120b on the radio corpus: a clear STROKE report with no priority spoken came back with
    # alert_level null, and the pipeline failed eight reports over it. Nothing said means null.
    report, _, _ = run(SAMPLE_STEMI, FakeLLM([json.dumps(report_dict(alert_type="STROKE", alert_level=None))]))
    assert report["alert_type"] == "STROKE" and report["alert_level"] is None
    assert "alert level" in report["data_quality"]["missing_info"]


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
    assert report["data_quality"]["missing_info"] == ["patient age", "ETA", "code status"]   # filled from the null fields


def test_null_lists_and_sections_from_the_model_become_empty_shapes():
    # the CS022 / S007_01 baseline failure: data_quality.assumptions was null and .extend crashed
    reply = report_dict(eta_minutes=12, ems_interventions=None, hospital_recommendations=None,
                        history={"past_medical": None, "medications": "metformin", "allergies": None, "code_status": None},
                        data_quality={"confidence": "MEDIUM", "missing_info": None, "assumptions": None})
    report, _, _ = run(SAMPLE_SEPSIS.replace("furosemide", "ferosemide"), FakeLLM([json.dumps(reply)]))
    assert report["alert_type"] == "STEMI" and report["data_quality"]["confidence"] == "MEDIUM"
    assert report["data_quality"]["missing_info"] == ["allergies"]   # the null history fact; "DNR" was spoken and fills code status
    assert report["history"]["code_status"] == "DNR"
    assert any("ferosemide" in a for a in report["data_quality"]["assumptions"])
    assert report["ems_interventions"] == [] and report["history"]["medications"] == ["metformin"]
    assert report["history"]["allergies"] is None and report["hospital_recommendations"]["resources_to_prepare"] == []


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


def test_prompt_offers_no_allergy_or_code_status_value_to_copy():
    from ems_handoff.prompt import EXTRACTION_PROMPT
    low = EXTRACTION_PROMPT.lower()
    assert "nkda" not in low and "full code /" not in low
    assert "null unless the paramedic actually spoke them" in EXTRACTION_PROMPT


def test_unspoken_allergies_and_code_status_become_null_and_missing():
    # CS021 bursts: "NKDA" and "Full code" written for a patient whose allergies and directives were never said
    text = "Medic 712, 72-year-old male, blood sugar 38, altered, gave D10, improving, ETA 10."
    reply = json.dumps(report_dict(alert_type="ENDOCRINE", history={"past_medical": ["diabetes"], "medications": [],
                                                                     "allergies": ["NKDA"], "code_status": "Full code"}))
    report, _, _ = run(text, FakeLLM([reply]))
    assert report["history"]["allergies"] is None and report["history"]["code_status"] is None
    missing = [m.lower() for m in report["data_quality"]["missing_info"]]
    assert "allergies" in missing and "code status" in missing
    notes = " ".join(report["data_quality"]["assumptions"])
    assert "Allergies set to null" in notes and "Code status set to null" in notes


def test_spoken_allergies_and_code_status_are_kept():
    reply = json.dumps(report_dict(alert_type="SEPSIS", history={"past_medical": ["CHF"], "medications": ["eliquis"],
                                                                  "allergies": ["penicillin", "sulfa"], "code_status": "DNR, full treatment per POLST"}))
    report, _, _ = run(SAMPLE_SEPSIS, FakeLLM([reply]))
    assert report["history"]["allergies"] == ["penicillin", "sulfa"]
    assert report["history"]["code_status"] == "DNR, full treatment per POLST"
    report, _, _ = run(SAMPLE_STEMI, FakeLLM([json.dumps(report_dict())]))   # "No known allergies" was said
    assert report["history"]["allergies"] == ["NKDA"]
    assert report["history"]["code_status"] is None                        # nothing about a code status was said


def test_blank_report_has_null_allergies_and_code_status():
    from ems_handoff.pipeline import blank_report
    assert blank_report()["history"]["allergies"] is None and blank_report()["history"]["code_status"] is None


def test_fragment_with_an_alert_word_is_not_unclear():
    # S038_01, 40 words, verbatim from the radio corpus
    text = ("Right, I'll do this. Hi, Reverend L9, this is Dirk Coney, do you need a physician? This is Ambulance "
            "721 coming to, what is it, trauma? We're going to consider him priority 2 at this time, how do you copy?")
    report, _, _ = run(text, FakeLLM([SPARSE_REPLY]))
    assert report["alert_type"] == "TRAUMA" and report["alert_level"] == "YELLOW", "the fragment states priority 2"
    assert report["data_quality"]["confidence"] == "LOW"
    assert any("from the word 'trauma'" in a for a in report["data_quality"]["assumptions"])
    report, _, _ = run("CVA, brain bleed.", FakeLLM([SPARSE_REPLY]))            # S009_01, 3 words
    assert report["alert_type"] == "STROKE"


def test_fragment_without_alert_words_stays_unclear():
    for text in ("How old? Sorry.", "So your ETA is 30 minutes, correct? 3-0, yes.", "The number is on the way. Okay, thank you."):
        report, _, _ = run(text, FakeLLM([SPARSE_REPLY]))
        assert report["alert_type"] == "UNCLEAR", text


def test_an_alert_type_the_model_named_is_never_overridden():
    report, _, _ = run("Stroke alert, but the fall was the trauma.", FakeLLM([json.dumps(report_dict(alert_type="STROKE"))]))
    assert report["alert_type"] == "STROKE"
    assert not any("the model answered UNCLEAR" in a for a in report["data_quality"]["assumptions"])


def test_prompt_limits_unclear_to_transcripts_with_no_alert_word():
    from ems_handoff.prompt import EXTRACTION_PROMPT
    assert 'Use "UNCLEAR" only when no chief complaint, no mechanism of injury, and no alert word' in EXTRACTION_PROMPT


def test_model_server_restart_reaches_the_stages_and_the_report_meta():
    llm = FakeLLM()
    llm.restart_notice = "Model server restarting (cannot reach Ollama); retrying in 3 s, 5 s waited"
    report, stages, pipe = run(SAMPLE_STEMI, llm)
    assert llm.on_wait is pipe.on_stage
    assert llm.restart_notice in stages
    assert report["_meta"]["model_server"] == {"model_server": "restarted", "reason": "cannot reach Ollama",
                                               "waited_s": 5.0, "retries": 2}
    report, _, _ = run(SAMPLE_STEMI, FakeLLM())
    assert "model_server" not in report["_meta"]


def test_sex_is_inferred_from_pronouns_when_the_model_left_it_null():
    from ems_handoff.pipeline import infer_sex
    from ems_handoff.prompt import EXTRACTION_PROMPT
    assert "pronouns" in EXTRACTION_PROMPT
    no_sex = report_dict(patient={"age": 72, "age_unit": "years", "sex": None})
    radio = ("Medic 21 inbound with a 72 year old, sudden change in vision in his left eye at 10:30. "
             "He states the vision is blurry. His blood pressure is 160 over 90. ETA 10 minutes.")
    report, _, _ = run(radio, FakeLLM([json.dumps(no_sex)]))
    assert report["patient"]["sex"] == "Male"
    assert "Sex inferred from pronouns (3 he/his/him, 0 she/her)" in report["data_quality"]["assumptions"]
    # nothing to go on: stays null
    report, _, _ = run("Medic 21 inbound with a 72 year old, vision change, ETA 10 minutes.", FakeLLM([json.dumps(no_sex)]))
    assert report["patient"]["sex"] is None
    # mixed pronouns (the daughter is talking): not decisive
    assert infer_sex("she says he fell; her father, his hip hurts, she called us") is None
    assert infer_sex("her daughter found her on the floor; she is alert") == ("Female", "Sex inferred from pronouns (3 she/her, 0 he/his/him)")
    assert infer_sex("he") is None                                   # one pronoun is not enough
    # what the model stated is never overridden
    said_female = report_dict(patient={"age": 72, "age_unit": "years", "sex": "Female"})
    report, _, _ = run(radio, FakeLLM([json.dumps(said_female)]))
    assert report["patient"]["sex"] == "Female" and not any("inferred" in a for a in report["data_quality"]["assumptions"])


# --- a stated priority sets the alert level ---------------------------------------------------
from ems_handoff.pipeline import alert_from_words, apply_stated_priority, stated_priority  # noqa: E402


def test_stated_priority_reads_digits_words_and_whispers_homophones():
    assert stated_priority("It's going to be a priority 2 chest pain.") == (2, "priority 2", None)
    assert stated_priority("It's going to be a priority to chest pain.")[0] == 2, "Whisper writes 'to' for 'two'"
    assert stated_priority("priority too, ten minutes")[0] == 2 and stated_priority("priority won")[0] == 1
    assert stated_priority("trauma, priority three")[0] == 3 and stated_priority("priority number 1")[0] == 1
    assert stated_priority("we'll be there in ten minutes") is None
    assert stated_priority("top priority to the airway") is None, "'priority to the' is not a number"
    # the last statement wins, and the burst comes from the session header before it
    t = "[Burst 1 · 00:00:00 UTC · feed 1] 9 words\npriority 2 chest pain\n\n[Burst 3 · 00:02:00 UTC · feed 1] 5 words\nupgrading to priority 1 now\n"
    assert stated_priority(t) == (1, "priority 1", 3)


def test_apply_stated_priority_overrides_the_model_and_attributes_the_burst():
    report = {"alert_level": "RED", "sources": {"alert_level": 5}, "data_quality": {"assumptions": [], "missing_info": ["alert level"]}}
    text = "[Burst 1 · 00:00:00 UTC · feed 1] 9 words\nMedic 731, priority to chest pain\n"
    note = apply_stated_priority(report, text)
    assert report["alert_level"] == "YELLOW" and report["sources"]["alert_level"] == 1
    assert note == "Alert level YELLOW from the stated 'priority to' (the model said RED)" and note in report["data_quality"]["assumptions"]
    assert report["data_quality"]["missing_info"] == []
    untouched = {"alert_level": "RED", "data_quality": {"assumptions": []}}
    assert apply_stated_priority(untouched, "62-year-old female with chest pain") is None and untouched["alert_level"] == "RED"
    agree = {"alert_level": "YELLOW", "data_quality": {"assumptions": []}}
    assert apply_stated_priority(agree, "priority 2") == "Alert level YELLOW from the stated 'priority 2'"


def test_pipeline_applies_the_stated_priority_to_the_model_report():
    llm = FakeLLM([json.dumps(report_dict(alert_type="MEDICAL", alert_level="RED"))])
    report, _, _ = run("Medic 731 with a 62-year-old female, priority to chest pain, BP 188 over 111, ETA five minutes.", llm)
    assert report["alert_level"] == "YELLOW"
    assert any(a.startswith("Alert level YELLOW from the stated") for a in report["data_quality"]["assumptions"])
    report, _, _ = run("Medic 731 with a 62-year-old female, chest pain, BP 188 over 111.", FakeLLM([json.dumps(report_dict(alert_level="RED"))]))
    assert report["alert_level"] == "RED", "nothing stated: the model's level stands"


def test_eta_is_cleared_when_its_number_is_not_a_time():
    """Over the air on 2026-10-26 Whisper dropped the STEMI script's ETA sentence and the model took the
    12 of "12-lead" as ETA 12. A number that is not spoken as a time is not an ETA."""
    text = ("Unit 41 inbound with a 67-year-old male, chest pain. 12-lead shows ST elevation in leads V1 "
            "through V4, consistent with anterior STEMI. Takes metformin daily. Unit 41 clear.")
    report, _, _ = run(text, FakeLLM([json.dumps(report_dict(eta_minutes=12))]))
    assert report["eta_minutes"] is None
    assert any(m.startswith("ETA") for m in report["data_quality"]["missing_info"])


@pytest.mark.parametrize("text, eta", [
    ("Our ETA to your facility is approximately 8 minutes. Requesting cath lab activation.", 8),
    ("It's going to be a priority two, about a ten minute ETA.", 10),
    ("Priority 2, we are 5 out.", 5),
    ("ETA 12 minutes. 12-lead unremarkable.", 12),
])
def test_eta_stays_when_the_transcript_says_it_as_a_time(text, eta):
    assert eta_supported(text, eta)
    report, _, _ = run(text, FakeLLM([json.dumps(report_dict(eta_minutes=eta))]))
    assert report["eta_minutes"] == eta


def test_eta_support_needs_a_time_cue_near_the_number():
    assert not eta_supported("12-lead shows ST elevation. BP 158 over 92.", 12)
    assert not eta_supported("324 milligrams aspirin chewed. Pain currently 6 out of 10.", 324)
    assert not eta_supported("Pain currently 6 out of 10, down from 9.", 6)
    assert not eta_supported("67-year-old male, BP 158 over 92.", 67)
    assert not eta_supported("", None)


def test_alert_words_count_only_when_stated_positively():
    """Round 3 candidate 2 (ROUND2.md): a denied or questioned alert word, or one named as history, is not the problem."""
    assert alert_from_words("This is Ambulance 721 coming to, what is it, trauma? ... priority 2") == ("TRAUMA", "trauma")
    assert alert_from_words("Priority two, stroke alert, 70-year-old male.") == ("STROKE", "stroke")
    assert alert_from_words("Did you just use your stroke? No.") is None
    assert alert_from_words("Negative for stroke, Cincinnati negative.") is None
    assert alert_from_words("No signs of trauma noted, chest pain.") is None
    assert alert_from_words("Patient has a history of stroke, chief complaint chest pain.") is None
    assert alert_from_words("Patient denies trauma. Stroke symptoms since 8 AM.") == ("STROKE", "Stroke")
    assert alert_from_words("Fell down the stairs, no loss of consciousness") == ("TRAUMA", "Fell down"), "a negation after another phrase does not cancel the mechanism"


def test_stated_no_known_allergies_and_code_status_fill_a_null_from_the_model():
    """Generated set, clean text (2026-10-27): qwen3:4b-instruct read "No known allergies" as nothing said and wrote
    null for both SY007 and SY012, with the assumption "No allergies mentioned". The spoken phrase goes in."""
    text = "Medic 41 to County General. 79-year-old female, motorcycle crash. History of diabetes. No known allergies. Patient is DNR DNI per paperwork. ETA 8 minutes."
    reply = json.dumps(report_dict(alert_type="TRAUMA", history={"past_medical": ["diabetes"], "medications": [], "allergies": None, "code_status": None},
                                   data_quality={"confidence": "HIGH", "missing_info": ["allergies", "code status"], "assumptions": []}))
    report, _, _ = run(text, FakeLLM([reply]))
    assert report["history"]["allergies"] == ["No known allergies"]
    assert report["history"]["code_status"] == "DNR DNI"
    missing = [m.lower() for m in report["data_quality"]["missing_info"]]
    assert "allergies" not in missing and "code status" not in missing
    notes = " ".join(report["data_quality"]["assumptions"])
    assert "Allergies taken from the transcript" in notes and "Code status taken from the transcript" in notes
    # nothing said: still null, still missing
    text2 = "Medic 41 to County General. 79-year-old female, motorcycle crash. ETA 8 minutes."
    report, _, _ = run(text2, FakeLLM([reply]))
    assert report["history"]["allergies"] is None and report["history"]["code_status"] is None


def test_unspoken_gcs_breakdown_ems_drugs_as_home_meds_and_unmentioned_interventions_are_removed():
    """Generated set, clean text (SY055, SY099, SY014 on 2026-10-27): qwen3:4b-instruct wrote "9 (Eye: 3, Verbal: 4, Motor: 2)"
    for "GCS is 9", filed the aspirin and nitro it had just given under home medications, and listed a cervical collar
    nobody mentioned."""
    text = ("Medic 41 to County General. 69-year-old male, chest pain. GCS is 9. Administered 324 mg aspirin chewed and 0.4 mg "
            "nitro sublingual. Takes metformin daily. 18-gauge IV left AC. ETA 8 minutes.")
    reply = json.dumps(report_dict(
        alert_type="STEMI", vital_signs={"blood_pressure": None, "heart_rate": None, "respiratory_rate": None, "spo2": None,
                                         "temperature": None, "blood_glucose": None, "gcs": "9 (Eye: 3, Verbal: 4, Motor: 2)"},
        ems_interventions=[{"intervention": "Aspirin", "details": "324 mg chewed", "response": None},
                           {"intervention": "Nitroglycerin", "details": "0.4 mg sublingual", "response": None},
                           {"intervention": "IV access", "details": "18-gauge left AC", "response": None},
                           {"intervention": "Cervical collar", "details": "Applied", "response": None}],
        history={"past_medical": [], "medications": ["aspirin", "nitroglycerin", "metformin"], "allergies": None, "code_status": None}))
    report, _, _ = run(text, FakeLLM([reply]))
    assert report["vital_signs"]["gcs"] == "9"
    assert report["history"]["medications"] == ["metformin"]
    assert [i["intervention"] for i in report["ems_interventions"]] == ["Aspirin", "Nitroglycerin", "IV access"]
    notes = " ".join(report["data_quality"]["assumptions"])
    assert "GCS breakdown removed" in notes and "Home medications removed" in notes and "Intervention removed: 'Cervical collar Applied'" in notes
    # spoken breakdown and a spoken collar stay
    text2 = "GCS is 9, eyes 3, verbal 4, motor 2. C-collar applied, patient on a backboard. Takes aspirin 81 daily."
    reply2 = json.dumps(report_dict(vital_signs={"gcs": "9 (E3 V4 M2)"}, ems_interventions=[{"intervention": "Cervical collar", "details": "Applied", "response": None}],
                                    history={"past_medical": [], "medications": ["aspirin 81"], "allergies": None, "code_status": None}))
    report, _, _ = run(text2, FakeLLM([reply2]))
    assert report["vital_signs"]["gcs"] == "9 (E3 V4 M2)" and len(report["ems_interventions"]) == 1
    assert report["history"]["medications"] == ["aspirin 81"]


def test_unrequested_activation_and_unspoken_room_air_are_cleared():
    """SY149, SY187, SY188 and SY085, SY140 on the clean text: "endocrine service" / "trauma team" requested by nobody, and
    "on room air" added to an SpO2 whose delivery was never stated."""
    text = "Medic 41 to County General. 30-year-old female, possible DKA. SpO2 97%, blood glucose 612. ETA 10 minutes."
    reply = json.dumps(report_dict(alert_type="ENDOCRINE", vital_signs={"spo2": "97% on room air", "blood_glucose": "612"},
                                   hospital_recommendations={"activation_requested": "endocrine service", "suggested_destination": "Critical Care",
                                                             "resources_to_prepare": [], "time_critical_actions": []}))
    report, _, _ = run(text, FakeLLM([reply]))
    assert report["hospital_recommendations"]["activation_requested"] is None
    assert report["vital_signs"]["spo2"] == "97%"
    text2 = "Medic 41 to County General. 30-year-old female, possible DKA. SpO2 97% on room air. Requesting a physician for orders. ETA 10 minutes."
    report, _, _ = run(text2, FakeLLM([reply]))
    assert report["hospital_recommendations"]["activation_requested"] == "endocrine service" and report["vital_signs"]["spo2"] == "97% on room air"


def test_grounding_rules_can_be_switched_off_for_ablation(monkeypatch):
    # the model claims an ETA the transcript never states
    text = "Medic 21 inbound with a 67 year old male, chest pain, ST elevation. Confirm report ready."
    reply = json.dumps(report_dict(eta_minutes=99))
    report, _, pipe = run(text, llm=FakeLLM([reply]))
    assert pipe.rules and report["eta_minutes"] is None          # the ETA guard cleared it
    report, _, pipe = run(text, llm=FakeLLM([reply]), rules=False)
    assert not pipe.rules and report["eta_minutes"] == 99
    assert any("Grounding rules off" in a for a in report["data_quality"]["assumptions"])
    monkeypatch.setenv("HANDOFF_REPORT_RULES", "0")
    assert run(text, llm=FakeLLM([reply]))[2].rules is False
    monkeypatch.setenv("HANDOFF_REPORT_RULES", "1")
    assert run(text, llm=FakeLLM([reply]))[2].rules is True


def test_compact_output_mode_changes_only_the_guidance(monkeypatch):
    from ems_handoff.prompt import extraction_prompt, COMPACT_GUIDANCE
    from ems_handoff import pipeline as P
    plain = extraction_prompt()
    compact = extraction_prompt(compact=True)
    assert COMPACT_GUIDANCE.strip() not in plain and COMPACT_GUIDANCE.strip() in compact
    assert "Still return the full structure." in plain and "Still return the full structure." not in compact
    assert compact.count("/no_think") == 1 and compact.index(COMPACT_GUIDANCE.strip()) < compact.index("/no_think")
    both = extraction_prompt(attribution=True, compact=True)
    assert "sources" in both and COMPACT_GUIDANCE.strip() in both
    monkeypatch.delenv("HANDOFF_COMPACT_OUTPUT", raising=False)
    assert P.compact_output() is False
    monkeypatch.setenv("HANDOFF_COMPACT_OUTPUT", "1")
    assert P.compact_output() is True


def test_facts_only_prompt_drops_the_prose_sections_and_keeps_the_stated_request(monkeypatch):
    from ems_handoff.prompt import extraction_prompt, FACTS_ONLY_GUIDANCE
    from ems_handoff import pipeline as P
    f = extraction_prompt(facts_only=True)
    for gone in ('"handoff_summary"', '"data_quality"', '"resources_to_prepare"', '"time_critical_actions"', "missing_info", "data_quality.assumptions"):
        assert gone not in f, gone
    assert '"activation_requested"' in f and '"suggested_destination"' in f and '"history"' in f
    assert FACTS_ONLY_GUIDANCE.strip() in f and f.count("/no_think") == 1 and f.rstrip().endswith("/no_think")
    both = extraction_prompt(attribution=True, facts_only=True)
    assert '"sources"' in both and '"handoff_summary"' not in both and "every source is 1" in both
    assert '"handoff_summary"' in extraction_prompt()          # the default is untouched
    monkeypatch.delenv("HANDOFF_FACTS_ONLY", raising=False)
    assert P.facts_only() is False
    monkeypatch.setenv("HANDOFF_FACTS_ONLY", "1")
    assert P.facts_only() is True


def test_render_prose_composes_summary_quality_and_preparation_from_the_facts():
    from ems_handoff.pipeline import blank_report, render_prose
    r = blank_report()
    r.update({"alert_type": "STEMI", "alert_level": "RED", "eta_minutes": 8, "chief_complaint": "Chest pain radiating to left arm"})
    r["patient"].update({"age": 67, "age_unit": "years", "sex": "Male"})
    r["vital_signs"]["heart_rate"] = 98
    r["hospital_recommendations"]["activation_requested"] = "Cath lab"
    render_prose(r)
    assert r["handoff_summary"]["one_liner"] == "67-year-old male, Chest pain radiating to left arm, stemi alert, RED, ETA 8 min."
    assert r["handoff_summary"]["key_concerns"] == ["Chest pain radiating to left arm", "Cath lab"]
    assert r["handoff_summary"]["pending_needs"] == ["Cath lab"]
    assert r["hospital_recommendations"]["resources_to_prepare"] == ["Cath team", "12-lead on arrival"]
    assert r["hospital_recommendations"]["time_critical_actions"] == ["Activate cath lab"]
    assert r["data_quality"]["confidence"] == "HIGH"
    assert "allergies" in r["data_quality"]["missing_info"] and "code status" in r["data_quality"]["missing_info"]
    assert r["data_quality"]["assumptions"][-1].startswith("summary, missing-information list")
    bare = blank_report()
    render_prose(bare)
    assert bare["handoff_summary"]["one_liner"] == "patient." and bare["data_quality"]["confidence"] == "LOW"
    assert bare["hospital_recommendations"]["resources_to_prepare"] == []
