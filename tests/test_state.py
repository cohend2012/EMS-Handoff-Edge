"""Patient state: derived observations, flags, series, missing documentation, provenance."""
from ems_handoff import state as S


def _report(**over):
    r = {"alert_type": "TRAUMA", "alert_level": "RED", "eta_minutes": 8,
         "patient": {"age": 34, "age_unit": "years", "sex": "Male"},
         "chief_complaint": "MVC rollover", "mechanism_or_onset": "rollover, ejected",
         "vital_signs": {"blood_pressure": "88/50", "heart_rate": 124, "respiratory_rate": 32,
                         "spo2": "89% on room air", "temperature": None, "blood_glucose": None, "gcs": 12},
         "assessment_findings": {"level_of_consciousness": "Responds to verbal", "airway": "Patent", "breathing": None,
                                 "circulation": None, "neuro": None, "other_findings": []},
         "ems_interventions": [], "history": {"past_medical": [], "medications": [], "allergies": [], "code_status": None}}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k] = {**r[k], **v}
        else:
            r[k] = v
    return r


def test_parsing_the_writers_fields():
    assert S.parse_bp("119/71") == (119, 71)
    assert S.parse_bp("120 over 80") == (120, 80)
    assert S.parse_bp("systolic 100") == (100, None)
    assert S.parse_bp(None) == (None, None)
    assert S.parse_spo2("96% on room air") == 96
    assert S.parse_temp_c("98.6 F") == 37.0
    assert S.parse_temp_c(38.2) == 38.2


def test_derived_quantities_and_published_flags():
    r = S.reading_from_report(_report(), "burst 1")
    assert r.shock_index == round(124 / 88, 2)
    assert r.map == round((88 + 100) / 3)
    assert r.pulse_pressure == 38
    assert r.rsig == round(88 / 124 * 12, 1)
    assert r.room_air is True and r.conscious is False
    names = {f.name for f in S.flags_for(r)}
    assert {"shock_index", "hypotension", "respiratory_rate", "hypoxia", "altered_mental_status", "low_map", "rsig"} <= names
    si = next(f for f in S.flags_for(r) if f.name == "shock_index")
    assert "1.41" in si.text and "Jehan" in si.rule and si.source == "burst 1"


def test_normal_vitals_raise_no_flag_and_a_partial_news2_says_what_is_missing():
    r = S.reading_from_report(_report(vital_signs={"blood_pressure": "119/71", "heart_rate": 72, "respiratory_rate": 19,
                                                   "spo2": "96% on room air", "gcs": 15},
                                      assessment_findings={"level_of_consciousness": "Alert and oriented x4"}), "burst 1")
    assert S.flags_for(r) == []
    n = S.news2(r)
    assert n["score"] == 0 and n["partial"] is True and n["missing"] == ["temperature"] and n["band"] == "low"


def test_news2_scores_the_bands_and_marks_high_on_any_three():
    r = S.reading_from_report(_report(vital_signs={"blood_pressure": "88/50", "heart_rate": 124, "respiratory_rate": 32,
                                                   "spo2": "89% on 4 liters", "temperature": "35.0", "gcs": 12}), "burst 1")
    n = S.news2(r)
    assert n["parts"] == {"rr": 3, "spo2": 3, "oxygen": 2, "sbp": 3, "hr": 2, "consciousness": 3, "temp": 3}
    assert n["score"] == 19 and n["partial"] is False and n["band"] == "high"


def test_two_readings_give_first_last_range_and_threshold_crossings_not_a_trajectory():
    h = [("burst 1", 0.0, _report(vital_signs={"blood_pressure": "118/76", "heart_rate": 96, "respiratory_rate": 20,
                                                "spo2": "97% on room air", "gcs": 15})),
         ("burst 3", 300.0, _report(vital_signs={"blood_pressure": "84/48", "heart_rate": 128, "respiratory_rate": 30,
                                                  "spo2": "97% on room air", "gcs": 15}))]
    st = S.build_state(h)
    assert len(st.readings) == 2
    sbp = next(s for s in st.series if s["vital"] == "sbp")
    assert sbp["first"] == 118 and sbp["last"] == 84 and sbp["change"] == -34 and sbp["min"] == 84 and sbp["max"] == 118
    assert sbp["crossings"] == ["SBP crossed the threshold between burst 1 and burst 3 (118 to 84)"]
    si = next(s for s in st.series if s["vital"] == "shock_index")
    assert si["crossings"] and "burst 1 and burst 3" in si["crossings"][0]
    assert all("trend" not in k for s in st.series for k in s)


def test_slots_carry_provenance_and_conflicts_are_named():
    h = [("burst 1", 0.0, _report(eta_minutes=10, patient={"age": 34})),
         ("burst 2", 60.0, _report(eta_minutes=10, patient={"age": 34})),
         ("burst 3", 120.0, _report(eta_minutes=5, patient={"age": 43}))]
    st = S.build_state(h)
    assert st.slots["eta_minutes"].first == "burst 1" and st.slots["eta_minutes"].value == 5
    assert st.slots["eta_minutes"].changed == ["burst 3"] and st.slots["eta_minutes"].previous == [10]
    assert st.slots["chief_complaint"].changed == []
    assert any(c.startswith("eta_minutes: 10 (burst 1) then 5 (burst 3)") for c in st.conflicts)
    assert any(c.startswith("patient.age: 34") for c in st.conflicts)


def test_stroke_documentation_checklist_names_what_is_missing():
    r = _report(alert_type="STROKE", chief_complaint="left-sided weakness", mechanism_or_onset=None,
                vital_signs={"blood_pressure": "168/94", "heart_rate": 88, "respiratory_rate": 18, "spo2": "97% on room air",
                             "blood_glucose": None, "gcs": 15},
                assessment_findings={"level_of_consciousness": "Alert", "neuro": "left arm drift, facial droop"},
                history={"medications": [], "past_medical": ["hypertension"], "allergies": [], "code_status": None})
    missing = S.missing_documentation(r)
    assert "last known well or onset time" in missing
    assert "blood glucose" in missing
    assert "anticoagulant use (yes or no)" in missing
    assert "deficit or stroke scale (LAMS, RACE, FAST-ED, Cincinnati)" not in missing
    done = _report(alert_type="STROKE", mechanism_or_onset="last known well 14:10",
                   vital_signs={"blood_pressure": "168/94", "heart_rate": 88, "respiratory_rate": 18, "spo2": "97%", "blood_glucose": 112, "gcs": 15},
                   assessment_findings={"level_of_consciousness": "Alert", "neuro": "LAMS 4"},
                   history={"medications": ["Eliquis"], "past_medical": [], "allergies": [], "code_status": None})
    assert S.missing_documentation(done) == []


def test_stemi_and_trauma_checklists_and_the_common_floor():
    stemi = _report(alert_type="STEMI", chief_complaint="chest pain",
                    assessment_findings={"level_of_consciousness": "Alert", "other_findings": ["12-lead shows inferior ST elevation"]},
                    ems_interventions=[{"intervention": "Aspirin administration", "details": "324 mg", "response": None}])
    assert S.missing_documentation(stemi) == []
    stemi_no_ecg = _report(alert_type="STEMI", chief_complaint="chest pain")
    assert "12-lead finding" in S.missing_documentation(stemi_no_ecg)
    assert "aspirin given or contraindicated" in S.missing_documentation(stemi_no_ecg)
    trauma = _report(vital_signs={"blood_pressure": None, "heart_rate": 124, "respiratory_rate": None, "spo2": None, "gcs": None},
                     assessment_findings={"level_of_consciousness": None})
    assert S.missing_documentation(trauma) == ["GCS or mental status", "blood pressure", "respiratory rate", "SpO2"]
    bare = {"alert_type": "MEDICAL"}
    assert S.missing_documentation(bare) == ["age", "sex", "chief complaint", "ETA"]


def test_state_from_an_empty_history_or_fragments_is_calm():
    assert S.build_state([]).to_dict()["slots"] == {}
    st = S.build_state([("burst 1", None, {"alert_type": "UNCLEAR", "vital_signs": {}})])
    assert st.readings == [] and st.flags == [] and st.news2 is None and st.series == []
    assert st.missing == ["age", "sex", "chief complaint", "ETA"]
    d = st.to_dict()
    assert set(d) == {"alert_type", "slots", "readings", "flags", "series", "news2", "missing", "conflicts"}


def test_incident_session_snapshot_carries_the_state_with_burst_provenance():
    from ems_handoff.incident import IncidentSession, Burst
    s = IncidentSession(id="s1")
    s.bursts = [Burst(number=1, text="Medic 14, 34 male, MVC", time=1000.0, duration_s=20.0),
                Burst(number=2, text="BP 84 over 48, pulse 128, ETA 5", time=1100.0, duration_s=15.0)]
    r1 = _report(eta_minutes=10, vital_signs={"blood_pressure": None, "heart_rate": None, "respiratory_rate": None, "spo2": None, "gcs": None})
    r1["_meta"] = {"session": {"covers": 1}}
    r2 = _report(eta_minutes=5)
    r2["_meta"] = {"session": {"covers": 2}}
    s.reports = [r1, r2]
    st = s.snapshot()["state"]
    assert st["slots"]["eta_minutes"] == {"path": "eta_minutes", "value": 5, "first": "burst 1", "changed": ["burst 2"], "previous": [10]}
    assert st["readings"][0]["source"] == "burst 2" and st["readings"][0]["time"] == 1115.0
    assert any(f["name"] == "shock_index" for f in st["flags"])
    assert st["missing"] == []
    assert "eta_minutes: 10 (burst 1) then 5 (burst 2)" in st["conflicts"]


def test_an_unclear_alert_after_a_stated_one_is_not_a_conflict():
    h = [("burst 1", None, _report(alert_type="STROKE")), ("burst 2", None, _report(alert_type="UNCLEAR"))]
    st = S.build_state(h)
    assert st.slots["alert_type"].value == "STROKE" and st.slots["alert_type"].changed == []
    assert st.conflicts == []


def test_the_checklist_runs_over_the_merged_slots_so_a_short_later_burst_does_not_blank_it():
    full = _report(alert_type="STEMI", chief_complaint="chest pain",
                   assessment_findings={"level_of_consciousness": "Alert", "other_findings": ["12-lead: inferior ST elevation"]},
                   ems_interventions=[{"intervention": "Aspirin", "details": "324 mg", "response": None}])
    reply = {"alert_type": "UNCLEAR", "eta_minutes": None, "patient": {}, "vital_signs": {}, "assessment_findings": {}, "history": {}}
    st = S.build_state([("burst 1", None, full), ("burst 2", None, reply)])
    assert st.alert_type == "STEMI" and st.missing == []


def test_the_ask_rule_fires_only_when_the_call_closes_with_a_required_slot_empty():
    from ems_handoff.incident import IncidentManager
    stroke = _report(alert_type="STROKE", chief_complaint="left-sided weakness", mechanism_or_onset=None,
                     vital_signs={"blood_pressure": "168/94", "heart_rate": 88, "respiratory_rate": 18, "spo2": "97%", "blood_glucose": 112, "gcs": 15},
                     assessment_findings={"level_of_consciousness": "Alert", "neuro": "left arm drift"},
                     history={"medications": ["Eliquis"], "past_medical": [], "allergies": [], "code_status": None})
    events = []
    m = IncidentManager(lambda text, meta: dict(stroke), quiet_s=25.0, clock=lambda: 1000.0, emit=events.append, start_worker=False)
    m.add_burst("Medic 14, 64 male, left-sided weakness, BP 168 over 94", at=1000.0, duration_s=20.0)
    m.run_pending(timeout=5)                                       # nothing fired yet: the quiet period has not passed
    assert m.sessions[0].ask is None and m.sessions[0].closing is None
    m.add_burst("ETA 6 minutes, see you on arrival", at=1040.0, duration_s=5.0)      # closing cue
    m.run_pending(timeout=5)
    s = m.sessions[0]
    assert s.closing == "closing cue in burst 2"
    assert s.ask == "Not stated for STROKE: last known well or onset time"
    assert [e for e in events if e.get("event") == "ask"][-1]["ask"] == s.ask
    assert s.snapshot()["ask"] == s.ask and s.snapshot()["closing"] == s.closing
    assert any(ev.startswith("ask after burst 2 (closing cue in burst 2): Not stated") for ev in s.events)


def test_no_ask_when_the_closing_report_has_everything_or_the_call_is_still_open():
    from ems_handoff.incident import IncidentManager
    full = _report(alert_type="STEMI", chief_complaint="chest pain",
                   assessment_findings={"level_of_consciousness": "Alert", "other_findings": ["12-lead: inferior ST elevation"]},
                   ems_interventions=[{"intervention": "Aspirin", "details": "324 mg", "response": None}])
    m = IncidentManager(lambda text, meta: dict(full), quiet_s=25.0, clock=lambda: 1000.0, start_worker=False)
    m.add_burst("Medic 2, 67 male, chest pain, ST elevation, aspirin given, ETA 8, Medic 2 clear", at=1000.0, duration_s=30.0)
    m.run_pending(timeout=5)
    assert m.sessions[0].closing == "closing cue in burst 1" and m.sessions[0].ask is None
    assert any(ev.endswith("nothing missing") for ev in m.sessions[0].events)
    m2 = IncidentManager(lambda text, meta: dict(_report(eta_minutes=None)), quiet_s=25.0, clock=lambda: 1000.0, start_worker=False)
    m2.add_burst("Medic 3, 34 male, rollover", at=1000.0, duration_s=20.0)
    m2.poll(now=2000.0)                                            # quiet period, not a closing cue
    m2.run_pending(timeout=5)
    assert m2.sessions[0].reports and m2.sessions[0].ask is None and m2.sessions[0].closing is None


def test_a_call_that_ends_in_silence_still_gets_its_closing_pass_and_ask():
    from ems_handoff.incident import IncidentManager
    seen = []
    def extract(text, meta):
        seen.append(dict(meta)); return dict(_report(eta_minutes=None))
    now = {"t": 1000.0}
    m = IncidentManager(extract, quiet_s=25.0, clock=lambda: now["t"], start_worker=False, close_quiet_s=90.0)
    m.add_burst("Medic 3, 34 male, rollover, BP 88 over 50", at=1000.0, duration_s=20.0)
    now["t"] = 1046.0; m.poll(); m.run_pending(timeout=5)                   # the quiet-period report: the call is open
    assert len(seen) == 1 and seen[0]["closing"] is False and m.sessions[0].ask is None
    assert m.sessions[0].close_due_at == 1020.0 + 90.0
    now["t"] = 1100.0; m.poll(); m.run_pending(timeout=5)                   # not yet
    assert len(seen) == 1
    now["t"] = 1111.0; m.poll(); m.run_pending(timeout=5)                   # 90 s of silence after the last burst: closing
    assert len(seen) == 2 and seen[1]["closing"] is True
    s = m.sessions[0]
    assert s.closing.startswith("quiet for 90 s") and s.ask == "Not stated for TRAUMA: ETA" and s.close_due_at is None
    assert any(e.startswith("closing by silence") for e in s.events)
    now["t"] = 1300.0; m.poll(); m.run_pending(timeout=5)                   # nothing more fires
    assert len(seen) == 2


def test_a_new_burst_cancels_the_close_after_quiet_deadline():
    from ems_handoff.incident import IncidentManager
    seen = []
    now = {"t": 1000.0}
    m = IncidentManager(lambda text, meta: (seen.append(dict(meta)) or dict(_report())), quiet_s=25.0, clock=lambda: now["t"], start_worker=False, close_quiet_s=90.0)
    m.add_burst("first burst", at=1000.0, duration_s=20.0)
    now["t"] = 1046.0; m.poll(); m.run_pending(timeout=5)
    assert m.sessions[0].close_due_at is not None
    m.add_burst("second burst", at=1080.0, duration_s=10.0)
    assert m.sessions[0].close_due_at is None                                   # the call went on
    now["t"] = 1116.0; m.poll(); m.run_pending(timeout=5)                   # quiet report for the second burst
    assert len(seen) == 2 and seen[1]["closing"] is False and m.sessions[0].close_due_at == 1090.0 + 90.0
    m2 = IncidentManager(lambda text, meta: dict(_report()), quiet_s=25.0, clock=lambda: 1000.0, start_worker=False)   # the default: off
    m2.add_burst("x", at=1000.0, duration_s=5.0); m2.poll(now=1031.0); m2.run_pending(timeout=5)
    assert m2.sessions[0].close_due_at is None
