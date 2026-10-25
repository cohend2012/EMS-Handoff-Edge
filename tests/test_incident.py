"""Incident sessions: bursts -> growing transcript -> re-extraction under the trigger rules."""

import json
import threading
import time

import pytest

from conftest import wait_for
from ems_handoff.incident import (Burst, IncidentManager, describe_change, is_closing_cue, leaf_fields,
                                  session_transcript, value_in_text, verify_attribution)
from ems_handoff.pipeline import ReportPipeline, parse_report
from fakes import FakeLLM, report_dict

# CS014 as the design note scripts it: times are seconds from the first burst, durations real.
B1 = ("Shady Grove, Medics 729, requesting a pediatric physician. Three-month-old female, trouble breathing.", 0.0, 22.9, "4390", "S054_01")
B2 = ("How old? Sorry.", 10.0, 3.7, "4430", "S055_01")
B3 = ("729, you have my doctor yet?", 134.0, 3.8, "4390", "S013_01")
B4 = ("Shady Grove, the number is on the way.", 136.0, 5.5, "4390", "S014_01")
B5 = ("Three-month-old female in respiratory distress, drooling, SpO2 low 90s on oxygen, "
      "gave dexamethasone 2 milligrams, ETA 10 minutes, see you when you get here.", 155.0, 67.6, "4390", "S015_01")


def first_report():
    return json.dumps(report_dict(
        alert_type="PEDIATRIC", alert_level=None, eta_minutes=None,
        patient={"age": 3, "age_unit": "months", "sex": "Female"},
        chief_complaint="Trouble breathing", mechanism_or_onset=None,
        vital_signs={}, assessment_findings={}, ems_interventions=[],
        history={"past_medical": [], "medications": [], "allergies": None, "code_status": None},
        sources={"patient.age": 1, "patient.sex": 1, "chief_complaint": 1}))


def full_report():
    return json.dumps(report_dict(
        alert_type="PEDIATRIC", alert_level="RED", eta_minutes=10,
        patient={"age": 3, "age_unit": "months", "sex": "Female"},
        chief_complaint="Respiratory distress", mechanism_or_onset=None,
        vital_signs={"spo2": "low 90s on oxygen"}, assessment_findings={"breathing": "drooling, distress"},
        ems_interventions=[{"intervention": "Oxygen", "details": "", "response": ""},
                           {"intervention": "Dexamethasone", "details": "2 mg", "response": ""}],
        history={"past_medical": [], "medications": [], "allergies": ["NKDA"], "code_status": None},
        sources={"patient.age": 1, "patient.sex": 5, "chief_complaint": 5, "vital_signs.spo2": 5,
                 "assessment_findings.breathing": 5, "ems_interventions": 5, "eta_minutes": 5, "history.allergies": 5}))


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now


def manager(replies, quiet_s=25.0, worker=False, delay=0.0, **kw):
    llm = FakeLLM(replies, delay=delay)
    pipe = ReportPipeline(llm, attribution=True)
    events = []
    clock = Clock()
    m = IncidentManager(lambda text, meta: pipe.run(text, source="replay"), quiet_s=quiet_s, clock=clock,
                        emit=events.append, start_worker=worker, **kw)
    m.events, m.llm, m.clock_obj = events, llm, clock
    return m


def play(m, *bursts):
    out = []
    for text, at, dur, feed, sid in bursts:
        m.clock_obj.now = at + dur
        out.append(m.add_burst(text, at=at, duration_s=dur, feed=feed, source_id=sid))
    return out


# -- transcript and cues ------------------------------------------------------
def test_transcript_grows_with_burst_headers():
    m = manager([first_report()])
    play(m, B1, B2)
    s = m.open_session
    assert [b.number for b in s.bursts] == [1, 2]
    text = s.transcript
    assert text.startswith("[Burst 1 · 00:00:00 UTC · feed 4390]\nShady Grove")
    assert "[Burst 2 · 00:00:10 UTC · feed 4430]\nHow old? Sorry." in text
    assert s.snapshot()["of"] == 2 and s.snapshot()["report"] is None


def test_closing_cues():
    assert is_closing_cue("ETA 10, see you when you get here.")
    assert is_closing_cue("Copy, see you on arrival")
    assert is_closing_cue("Medic 14 is clear.")
    assert not is_closing_cue("the road is clear of traffic but we are still twenty out")
    assert not is_closing_cue("How old? Sorry.")


# -- trigger rules ----------------------------------------------------------------
def test_quiet_period_triggers_one_run_over_everything_received():
    m = manager([first_report()])
    play(m, B1, B2)
    s = m.open_session
    m.poll(now=40.0)                       # burst 1 ended at 22.9: 25 s of quiet end at 47.9
    assert not s.updating and m._jobs.empty()
    m.poll(now=48.0)
    assert s.updating and s.snapshot()["updating"]
    m.run_pending()
    assert s.current["alert_type"] == "PEDIATRIC" and s.covered == 2 and not s.updating
    assert s.change_lines == ["burst 2 added: patient (age, age_unit, sex), complaint; alert PEDIATRIC"]
    assert s.current["_meta"]["session"] == {"id": "P1", "covers": 2, "burst_ids": ["S054_01", "S055_01"]}
    assert "[Burst 2" in m.llm.prompts[0], "the model read the whole session transcript"
    assert '"sources"' in m.llm.prompts[0], "sessions ask for burst attribution"
    kinds = [e["event"] for e in m.events]
    assert kinds == ["burst", "burst", "extracting", "report"]


def test_closing_cue_extracts_at_once_and_online_quiet_is_shorter():
    m = manager([first_report()], quiet_s=5.0)
    play(m, B1)
    assert not m.open_session.updating
    m.poll(now=27.0)
    assert not m.open_session.updating
    m.poll(now=28.0)
    assert m.open_session.updating
    m.run_pending()
    play(m, B5)                            # "see you when you get here"
    assert m.open_session.updating, "the closing cue did not wait for the quiet period"


def test_previous_report_stays_current_with_the_updating_marker_and_runs_coalesce():
    m = manager([first_report(), full_report()], worker=True, delay=0.4)
    try:
        play(m, B1)
        m.poll(now=48.0)
        assert wait_for(lambda: m.open_session.current is not None)
        s = m.open_session
        first = s.current
        play(m, B5)                        # a closing cue: the second run starts at once
        assert s.updating and s.current is first and s.covered == 1
        snap = s.snapshot()
        assert snap["updating"] and snap["covered"] == 1 and snap["of"] == 2 and snap["report"] is first
        m.add_burst("Copy, Medics 729 clear.", at=230.0, duration_s=2.0, feed="4390", source_id="S099_01")
        m.add_burst("Shady Grove clear.", at=233.0, duration_s=1.0, feed="4430", source_id="S100_01")
        assert s.pending and s.updating and s.current is first, "two more bursts during the run: one run queued"
        assert wait_for(lambda: s.covered == 4 and not s.updating, timeout=5)
        time.sleep(0.5)
        assert len(s.reports) == 3 and len(m.llm.prompts) == 3, "bursts 3 and 4 coalesced into one run"
        assert s.current["eta_minutes"] == 10 and s.current["alert_level"] == "RED"
        assert s.change_lines[1].startswith("burst 2 added:") and s.change_lines[2] == "burst 4: no new facts"
    finally:
        m.stop()


def test_cs014_offline_schedule_makes_three_reports():
    """The design note's table: a run after the quiet period past burst 2, one after burst 4,
    and one at the closing cue of burst 5 (coalesced behind the burst-4 run)."""
    m = manager([first_report(), first_report(), full_report()])
    s = play(m, B1, B2)[0][0]
    m.poll(now=47.9); m.run_pending()
    assert s.covered == 2
    play(m, B3, B4)
    m.poll(now=141.5 + 25 - 0.1)
    assert not s.updating
    m.poll(now=141.5 + 25)
    assert s.updating and s.queued_through == 4
    play(m, B5)                            # arrives while the burst-4 run is in flight
    assert s.pending
    m.run_pending()
    assert s.covered == 5 and len(s.reports) == 3 and not s.updating
    assert s.change_lines[1] == "burst 4: no new facts"
    assert s.change_lines[2] == ("burst 5 added: ETA 10 min, vitals (spo2), assessment (breathing), interventions (2); "
                                 "changed: chief_complaint Trouble breathing -> Respiratory distress; alert PEDIATRIC RED")
    att = s.current["attribution"]
    assert att["verified"]["patient.age"] == 1 and att["verified"]["eta_minutes"] == 5
    assert att["verified"]["ems_interventions"] == 5
    # the model's NKDA was nulled by the pipeline's grounding rule before attribution ran
    assert s.current["history"]["allergies"] is None and att["suspect"] == []


# -- boundary rule and the operator buttons ---------------------------------------
def test_long_gap_starts_a_new_session():
    m = manager([first_report()])
    play(m, B1)
    m.add_burst("Medic 12 with a 70 year old male chest pain", at=22.9 + 301, duration_s=5)
    assert [s.id for s in m.sessions] == ["P1", "P2"]
    assert m.sessions[0].closed and m.open_session.id == "P2"
    assert m.open_session.bursts[0].number == 1
    assert m.sessions[0].events == ["session started: first burst"]
    assert "gap of 301 s" in m.sessions[1].events[0]


def test_new_patient_closes_the_session_and_merge_restores_it_with_one_run():
    m = manager([first_report(), first_report(), full_report()])
    s1 = play(m, B1, B2, B3)[0][0]
    m.new_patient()                        # pressed at burst 3, with its quiet period still pending
    assert s1.closed and s1.updating, "the pending text is extracted on New patient"
    m.run_pending()
    assert s1.covered == 3 and m.open_session is None
    play(m, B4, B5)
    s2 = m.open_session
    assert s2.id == "P2" and [b.number for b in s2.bursts] == [1, 2]
    assert s2.updating                     # B5's closing cue
    m.run_pending()
    prompts_before = len(m.llm.prompts)
    merged = m.merge_with_previous()
    assert merged is s1 and m.sessions == [s1] and not s1.closed
    assert [b.source_id for b in s1.bursts] == ["S054_01", "S055_01", "S013_01", "S014_01", "S015_01"]
    assert [b.number for b in s1.bursts] == [1, 2, 3, 4, 5]
    assert s1.updating
    m.run_pending()
    assert len(m.llm.prompts) == prompts_before + 1, "one re-extraction on the merged transcript"
    assert s1.covered == 5 and "[Burst 5" in m.llm.prompts[-1]
    assert any(e.startswith("New patient pressed after burst 3") for e in s1.events)
    assert any(e.startswith("Merge with previous: 2 burst(s) of P2") for e in s1.events)
    assert m.merge_with_previous() is None


def test_extraction_failure_keeps_the_session_and_the_old_report():
    from ems_handoff.llm import LLMError
    m = manager([first_report(), LLMError("Ollama is down")])
    s = play(m, B1)[0][0]
    m.poll(now=48.0); m.run_pending()
    first = s.current
    play(m, B5)
    m.run_pending()
    assert s.current is first and s.last_error.startswith("PipelineError: Ollama is down")
    assert not s.updating and m.events[-1]["event"] == "error"
    assert s.snapshot()["error"] == s.last_error


# -- attribution ---------------------------------------------------------------------
def test_value_in_text_handles_numbers_words_and_phrases():
    assert value_in_text(3, "three-month-old female")
    assert value_in_text("low 90s on oxygen", "SpO2 low 90s, we have her on oxygen")
    assert value_in_text("Dexamethasone 2 mg", "gave dexamethasone 2 milligrams")
    assert value_in_text([{"intervention": "Oxygen"}], "put him on oxygen")
    assert not value_in_text("NKDA", "three-month-old female, trouble breathing")
    assert not value_in_text(None, "anything")


def test_verify_attribution_marks_unattributed_and_wrong_bursts():
    bursts = [Burst(1, "67 year old male chest pain", 0), Burst(2, "BP 158 over 92, gave aspirin 324", 30)]
    report = parse_report(json.dumps(report_dict(sources={"patient.age": 1, "vital_signs.blood_pressure": 1,
                                                         "ems_interventions": 9, "history": 2})))
    att = verify_attribution(report, bursts)
    assert att["verified"]["patient.age"] == 1
    assert att["verified"]["vital_signs.blood_pressure"] == 2, "wrong burst cited: corrected to the one that states it"
    assert att["verified"]["ems_interventions"] == 2, "burst 9 does not exist, but burst 2 states aspirin"
    assert "history.past_medical" in att["suspect"] and "history.medications" in att["suspect"]
    assert "chief_complaint" in att["unattributed"] and "vital_signs.heart_rate" in att["unattributed"]
    assert report["attribution"] is att


def test_parse_report_keeps_sources_as_ints_and_drops_junk():
    r = parse_report(json.dumps(report_dict(sources={"patient.age": "2", "chief_complaint": 1.0, "x": "burst one", "y": None})))
    assert r["sources"] == {"patient.age": 2, "chief_complaint": 1}
    assert parse_report(json.dumps(report_dict()))["sources"] == {}
    assert parse_report(json.dumps(report_dict(sources="n/a")))["sources"] == {}


def test_describe_change_wording():
    a = {"patient": {"age": 3}, "alert_type": "PEDIATRIC", "alert_level": None}
    b = {"patient": {"age": 4}, "eta_minutes": 10, "alert_type": "PEDIATRIC", "alert_level": "RED"}
    assert describe_change(None, a, 1) == "burst 1 added: patient (age); alert PEDIATRIC"
    assert describe_change(a, b, 5) == "burst 5 added: ETA 10 min; changed: patient.age 3 -> 4; alert PEDIATRIC RED"
    assert describe_change(b, b, 6) == "burst 6: no new facts"
    assert leaf_fields({"vital_signs": {"spo2": None, "gcs": "15"}, "history": {"medications": []}}) == {"vital_signs.gcs": "15"}
