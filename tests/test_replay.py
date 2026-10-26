"""Replay: call-set loading, sink audio, the driver's timing, and the server path behind the flag."""

import dataclasses
import json
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from conftest import wait_for
from ems_handoff.llm import LLMError
from ems_handoff.replay import (BurstClip, CallSet, ReplayDriver, click, list_callsets, load_callset,
                                parse_source_name, prepare_sink_wav)
from ems_handoff.server import Engines, create_app
from fakes import RATE, FakeLLM, FakeTranscriber, ScriptedCaptioner, report_dict, tone


# -- fixtures -------------------------------------------------------------------
def write_wav(path, seconds, rate=RATE):
    pcm = (tone(seconds) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm.tobytes())
    return path


def make_callset(root, cs_id="CS900", bursts=None, enriched=True, feed="4390", words=True):
    """A call-set folder like ~/callsets/CSxxx. bursts: [(segment_id, seconds, epoch, feed)]."""
    bursts = bursts or [("S901_01", 0.3, 1792827662, feed), ("S902_01", 0.2, 1792827672, "4430")]
    folder = root / cs_id
    folder.mkdir(parents=True)
    detail, prev_end = [], None
    for sid, secs, epoch, fd in bursts:
        write_wav(folder / f"{sid}.wav", secs)
        if words:
            (folder / f"{sid}.txt").write_text("some words here for " + sid)
        gap = None if prev_end is None else round(epoch - prev_end, 1)
        detail.append({"segment_id": sid, "source_file": f"mocomdps-{fd}-{epoch}.m4a", "feed": fd, "epoch": epoch,
                       "duration_sec": secs, "gap_from_previous_sec": gap, "words": 4})
        prev_end = max(prev_end or 0, epoch + secs)
    info = {"callset_id": cs_id, "feed": feed, "hospital": ["shady grove"], "units": ["729"],
            "patient": {"ages": ["3m"], "sex": ["female"]}, "complaint_keywords": ["respiratory"],
            "first_burst_utc": "2026-10-24 07:41:02", "segments": [b[0] for b in bursts]}
    if enriched:
        info["bursts"] = detail
    (folder / "callset_info.json").write_text(json.dumps(info))
    return folder


class FakeSink:
    sink = "fake_sink"
    available = True

    def __init__(self):
        self.played = []
        self.last_started = None
        self.stopped = False

    def play(self, path):
        self.played.append((time.time(), path))
        self.last_started = time.time()

    def stop(self):
        self.stopped = True


class BurstTranscriber(FakeTranscriber):
    """One text per call, in order."""

    def __init__(self, texts):
        super().__init__(texts[0])
        self.texts = list(texts)

    def transcribe(self, audio):
        self.calls += 1
        self.last_audio_len = len(audio)
        return self.texts[min(self.calls, len(self.texts)) - 1]


# -- loading --------------------------------------------------------------------
def test_parse_source_name():
    assert parse_source_name("mocomdps-4390-1792827662.m4a") == ("4390", 1792827662.0)
    assert parse_source_name("S054_01.wav") is None


def test_enriched_callset_info_is_the_first_timing_source(tmp_path):
    folder = make_callset(tmp_path, bursts=[("A", 0.3, 100, "4390"), ("B", 0.2, 90, "4430"), ("C", 0.1, 200, "4390")])
    cs = load_callset(folder)
    assert cs.timing_source == "callset_info"
    assert [b.segment_id for b in cs.bursts] == ["B", "A", "C"], "epoch order, not listed order"
    assert [b.gap_s for b in cs.bursts] == [None, 9.8, 99.7]
    assert cs.bursts[1].feed == "4390" and cs.bursts[1].duration_s == 0.3 and cs.bursts[1].words == 4
    s = cs.summary()
    assert s["bursts"] == 3 and s["hospital"] == "shady grove" and s["ages"] == "3m"


def test_filenames_give_timing_when_the_info_has_none(tmp_path):
    folder = tmp_path / "CS901"
    folder.mkdir()
    write_wav(folder / "mocomdps-4390-1792827000.wav", 0.5)
    write_wav(folder / "mocomdps-4430-1792827003.wav", 0.2)
    (folder / "callset_info.json").write_text(json.dumps(
        {"callset_id": "CS901", "segments": ["mocomdps-4390-1792827000", "mocomdps-4430-1792827003"]}))
    cs = load_callset(folder)
    assert cs.timing_source == "filenames"
    assert [(b.feed, b.epoch, b.gap_s) for b in cs.bursts] == [("4390", 1792827000.0, None), ("4430", 1792827003.0, 2.5)]
    assert cs.bursts[0].duration_s == 0.5


def test_index_is_the_optional_third_source_and_none_plays_back_to_back(tmp_path):
    folder = make_callset(tmp_path, enriched=False)
    index = tmp_path / "callsets.csv"
    index.write_text("callset_id,segment_id,feed,epoch,source_file\n"
                     "CS900,S901_01,4390,1792827662,mocomdps-4390-1792827662.m4a\n"
                     "CS900,S902_01,4430,1792827672,mocomdps-4430-1792827672.m4a\n")
    cs = load_callset(folder, index)
    assert cs.timing_source == "index" and cs.bursts[1].gap_s == 9.7
    cs = load_callset(folder)
    assert cs.timing_source == "none" and [b.gap_s for b in cs.bursts] == [None, 0.0]
    assert cs.bursts[0].feed == "4390", "the call-set's feed when a burst has none"


def test_list_callsets_reports_bad_folders_without_hiding_the_rest(tmp_path):
    make_callset(tmp_path, "CS001")
    (tmp_path / "CS002").mkdir()
    (tmp_path / "CS002" / "callset_info.json").write_text(json.dumps({"callset_id": "CS002", "segments": ["gone"]}))
    (tmp_path / "notes").mkdir()
    listed = list_callsets(tmp_path)
    assert [c["id"] for c in listed] == ["CS001", "CS002"]
    assert listed[0]["bursts"] == 2 and "gone.wav is missing" in listed[1]["error"]
    assert list_callsets(tmp_path / "missing") == []
    with pytest.raises(FileNotFoundError, match="no callset_info.json"):
        load_callset(tmp_path / "notes")


# -- sink audio -------------------------------------------------------------------
def test_sink_wav_carries_the_click_over_the_first_50_ms(tmp_path):
    src = write_wav(tmp_path / "burst.wav", 0.5)
    out = prepare_sink_wav(src, tmp_path / "out" / "burst.sink.wav")
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 8000)
        audio = np.frombuffer(w.readframes(8000), dtype=np.int16).astype(np.float32) / 32768
    first, rest = audio[:800], audio[800:]
    assert np.sqrt(np.mean(first ** 2)) > 4 * np.sqrt(np.mean(rest ** 2)), "the click stands out"
    spectrum = np.abs(np.fft.rfft(first))
    assert abs(np.argmax(spectrum) * 16000 / 800 - 1000) < 40, "a 1 kHz click"
    assert len(click()) == 800 and click().max() <= 0.9


# -- the driver ---------------------------------------------------------------------
class FakeHost:
    def __init__(self, play_s=0.05):
        self.started = []
        self.play_s = play_s
        self.settled_after = 0
        self._settle_calls = 0

    def start_burst(self, clip, number):
        ev = threading.Event()
        self.started.append((time.time(), number, clip.segment_id))
        threading.Timer(self.play_s, ev.set).start()
        return ev

    def settled(self):
        self._settle_calls += 1
        return self._settle_calls > self.settled_after


def test_driver_plays_gaps_at_speed_clamps_overlap_and_logs_every_start(tmp_path):
    folder = make_callset(tmp_path, bursts=[("A", 0.1, 1000, "4390"), ("B", 0.1, 1000.05, "4430"), ("C", 0.1, 1010.1, "4390")])
    cs = load_callset(folder)
    assert [b.gap_s for b in cs.bursts] == [None, -0.1, 10.0]
    host, sink, events = FakeHost(), FakeSink(), []
    d = ReplayDriver(cs, host, sink=sink, speed=20.0, emit=events.append, work_dir=tmp_path / "work")
    t0 = time.time()
    d.start()
    assert wait_for(lambda: d.phase == "done", timeout=5)
    assert [n for _, n, _ in host.started] == [1, 2, 3]
    assert len(sink.played) == 3 and all(p.name.endswith(".sink.wav") for _, p in sink.played)
    for (t_host, _, _), (t_sink, _) in zip(host.started, sink.played):
        assert abs(t_sink - t_host) < 0.05, "sink audio starts with the file source"
    log = d.log
    assert [e["burst"] for e in log] == [1, 2, 3]
    assert log[1]["played_gap_s"] == 0 and "overlapped the previous burst by 0 s on feed 4430" in log[1]["note"]
    assert log[2]["played_gap_s"] == 0.5, "10 s gap at 20x"
    assert host.started[2][0] - host.started[1][0] >= 0.5
    assert all(e["start_iso"].count(":") == 2 and e["sink_lead_ms"] is not None for e in log)
    phases = [e["phase"] for e in events]
    assert phases[0] == "gap" and "burst" in phases and "waiting" in phases and phases[-1] == "done"
    assert sum(1 for e in events if e.get("flash")) == 3
    assert d.elapsed_s < time.time() - t0 + 0.01 and d.status()["of"] == 3


def test_sink_player_starts_a_fixed_lead_before_the_file_source(tmp_path):
    folder = make_callset(tmp_path, bursts=[("A", 0.1, 1000, "4390"), ("B", 0.1, 1001, "4390")])
    cs = load_callset(folder)
    host, sink, before = FakeHost(), FakeSink(), []
    d = ReplayDriver(cs, host, sink=sink, speed=10.0, work_dir=tmp_path / "work", sink_lead_s=0.12,
                     before_first_burst=lambda: before.append(time.time()))
    d.start()
    assert wait_for(lambda: d.phase == "done", timeout=5)
    for (t_host, _, _), (t_sink, _) in zip(host.started, sink.played):
        assert 0.10 <= t_host - t_sink <= 0.2, "the sink audio started the lead ahead of the file source"
    assert all(100 <= e["sink_lead_ms"] <= 200 for e in d.log)
    assert len(before) == 1 and before[0] <= sink.played[0][0], "the recorder hook ran once, before burst 1"
    host2, sink2 = FakeHost(), FakeSink()
    d = ReplayDriver(cs, host2, sink=sink2, speed=10.0, work_dir=tmp_path / "work", sink_lead_s=-0.12)
    d.start()
    assert wait_for(lambda: d.phase == "done", timeout=5)
    for (t_host, _, _), (t_sink, _) in zip(host2.started, sink2.played):
        assert 0.10 <= t_sink - t_host <= 0.2, "a negative lead starts the sink audio after the file source"
    assert all(-200 <= e["sink_lead_ms"] <= -100 for e in d.log)


def test_driver_stop_ends_the_run_early(tmp_path):
    folder = make_callset(tmp_path, bursts=[("A", 0.1, 1000, "4390"), ("B", 0.1, 1100, "4390")])
    cs = load_callset(folder)
    host, sink = FakeHost(), FakeSink()
    d = ReplayDriver(cs, host, sink=sink, speed=1.0, work_dir=tmp_path / "work")
    d.start()
    assert wait_for(lambda: d.phase == "gap" and d.burst_number == 2)
    d.stop()
    assert len(host.started) == 1 and sink.stopped and d.phase == "gap"


# -- the server path ------------------------------------------------------------------
SCRIPT = "medic seven two nine three month old female trouble breathing"


def build(settings, tmp_path, llms=None, texts=None, enabled=True, sink=None):
    callsets = tmp_path / "callsets"
    make_callset(callsets, "CS900")
    settings = dataclasses.replace(settings, replay_mode_enabled=enabled, replay_callsets_dir=callsets,
                                   replay_speed=50.0, output_dir=tmp_path / "out")
    llms = llms or {}
    default_llm = FakeLLM([json.dumps(report_dict(alert_type="PEDIATRIC",
                                                  patient={"age": 3, "age_unit": "months", "sex": "Female"},
                                                  sources={"patient.age": 1}))])
    made = []

    def make_llm(s):
        made.append(s)
        return llms.get(s.provider, default_llm)

    transcriber = BurstTranscriber(texts or ["Three month old female, trouble breathing.", "How old? Sorry."])
    engines = Engines(make_llm=make_llm, make_transcriber=lambda s: transcriber, make_fallback_transcriber=None,
                      make_listeners=lambda s: ((lambda: ScriptedCaptioner(SCRIPT, 1)), None, "scripted captions"),
                      make_capture=lambda s: (_ for _ in ()).throw(AssertionError("the mic must not be opened")),
                      make_sink=lambda s: sink or FakeSink(), list_mics=lambda: [])
    app = create_app(settings, engines)
    ctl = app.config["controller"]
    ctl.made_llm_settings = made
    return app.test_client(), ctl, transcriber


def events_of(client):
    resp = client.get("/api/events", buffered=False)
    return resp, iter(resp.response)


def test_replay_endpoints_do_not_exist_with_the_flag_off(settings, tmp_path):
    client, ctl, _ = build(settings, tmp_path, enabled=False)
    assert client.get("/api/replay/callsets").status_code == 404
    assert client.post("/api/replay/start", json={"callset": "CS900"}).status_code == 404
    status = client.get("/api/status").get_json()
    assert "replay" not in status and status["settings"]["replay_mode"] is False


def test_replay_runs_a_callset_through_captions_transcript_and_session(settings, tmp_path):
    sink = FakeSink()
    client, ctl, transcriber = build(settings, tmp_path, sink=sink)
    listed = client.get("/api/replay/callsets").get_json()
    assert [c["id"] for c in listed["callsets"]] == ["CS900"] and listed["callsets"][0]["bursts"] == 2
    assert client.post("/api/replay/start", json={"callset": "nope"}).status_code == 400
    r = client.post("/api/replay/start", json={"callset": "CS900"}).get_json()
    assert r["replay"]["provider"] == "groq" and r["replay"]["driver"]["of"] == 2
    assert ctl.state == "replaying"
    assert client.post("/api/live/start").status_code == 409, "the mic path is busy while a replay runs"
    assert ctl.made_llm_settings[-1].provider == "groq" and ctl.made_llm_settings[-1].groq_llm_model == "openai/gpt-oss-120b"
    assert wait_for(lambda: ctl.state == "idle", timeout=15)
    run = ctl._replay
    st = run.status()
    assert st["driver"]["phase"] == "done" and len(st["driver"]["log"]) == 2
    assert len(sink.played) == 2
    assert transcriber.calls == 2 and [t["words"] for t in st["transcriptions"]] == [6, 3]
    session = st["session"]
    assert session["of"] == 2 and session["covered"] == 2 and not session["updating"]
    assert session["bursts"][1]["source_id"] == "S902_01" and session["bursts"][1]["feed"] == "4430"
    assert "[Burst 2" in session["transcript"] and "How old? Sorry." in session["transcript"]
    assert session["report"]["alert_type"] == "PEDIATRIC"
    att = session["report"]["attribution"]
    assert att["verified"] == {"patient.age": 1, "patient.age_unit": 1} and "patient.sex" in att["unattributed"]
    assert "vital_signs.blood_pressure" in att["unattributed"], "the fake's vitals were never said and have no source"
    assert session["report"]["_meta"]["source"] == "replay:CS900" and "sources" in session["report"]
    assert ctl.last_result is session["report"], "the final report is the last result"
    assert client.get("/api/replay/status").get_json()["replay"]["commit"]
    assert client.post("/api/replay/merge").get_json()["session"] is None, "one session: nothing to merge"
    closed = client.post("/api/replay/new_patient").get_json()["session"]
    assert closed["closed"] and closed["id"] == "P1", "New patient closes the session that stayed open after done"
    assert client.post("/api/replay/new_patient").get_json()["session"] is None


def test_replay_events_carry_bursts_captions_and_the_flash(settings, tmp_path):
    client, ctl, _ = build(settings, tmp_path)
    resp, chunks = events_of(client)
    next(chunks)                                             # status
    client.post("/api/replay/start", json={"callset": "CS900"})
    seen, deadline = [], time.time() + 15
    while time.time() < deadline:
        e = json.loads(next(chunks).decode().removeprefix("data: ").strip())
        seen.append(e)
        if e.get("type") == "state" and e["state"] == "idle":
            break
    kinds = [e["type"] for e in seen]
    assert "replay" in kinds and "caption" in kinds and "transcript" in kinds and "incident" in kinds
    flashes = [e for e in seen if e.get("flash")]
    assert [e["burst"] for e in flashes] == [1, 2] and flashes[0]["provider"] == "groq"
    assert any(e["type"] == "caption" and e.get("replay_burst") == 1 for e in seen)
    assert any(e["type"] == "transcript" and e.get("replay_burst") == 2 and e["segment_id"] == "S902_01" for e in seen)
    assert any(e["type"] == "incident" and e["event"] == "report" for e in seen)
    assert [e["state"] for e in seen if e["type"] == "state"] == ["replaying", "idle"]
    assert seen[-2]["type"] == "replay" and seen[-2]["phase"] == "done", "done, then idle"
    resp.close()


def test_groq_failure_falls_back_to_local_and_is_marked(settings, tmp_path):
    groq = FakeLLM([LLMError("groq rate limit reached (HTTP 429)")], provider="groq")
    local = FakeLLM([json.dumps(report_dict(alert_type="PEDIATRIC"))], provider="ollama", model="qwen3:4b-instruct")
    client, ctl, _ = build(settings, tmp_path, llms={"groq": groq, "ollama": local})
    client.post("/api/replay/start", json={"callset": "CS900"})
    assert wait_for(lambda: ctl.state == "idle", timeout=15)
    st = ctl._replay.status()
    assert st["fallback"]["from"] == "groq" and st["fallback"]["to"] == "ollama"
    assert "rate limit" in st["fallback"]["reason"]
    report = st["session"]["report"]
    assert report["_meta"]["fallback"]["from"] == "groq" and report["_meta"]["provider"] == "ollama"
    assert len(local.prompts) == 1 and len(groq.prompts) == 1


def test_replay_can_be_stopped_and_refuses_a_missing_key(settings, tmp_path):
    client, ctl, _ = build(settings, tmp_path)
    r = client.post("/api/replay/start", json={"callset": "CS900", "provider": "openrouter"})
    assert r.status_code == 400 and "No API key" in r.get_json()["message"]
    client.post("/api/replay/start", json={"callset": "CS900", "speed": 0.01})
    assert ctl.state == "replaying"
    assert client.post("/api/replay/start", json={"callset": "CS900"}).status_code == 409
    client.post("/api/replay/stop")
    assert ctl.state == "idle" and ctl._replay.driver.phase != "done"



def test_silent_burst_is_no_speech_and_still_joins_the_session(settings, tmp_path):
    client, ctl, transcriber = build(settings, tmp_path)
    silent = tmp_path / "callsets" / "CS900" / "S902_01.wav"
    with wave.open(str(silent), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(np.zeros(RATE // 5, np.int16).tobytes())
    client.post("/api/replay/start", json={"callset": "CS900"})
    assert wait_for(lambda: ctl.state == "idle", timeout=15)
    st = ctl._replay.status()
    second = st["transcriptions"][1]
    assert second["no_speech"] and second["words"] == 0 and second["transcriber"] == "none"
    assert transcriber.calls == 1, "Whisper was not called for the silent burst"
    assert st["session"]["of"] == 2 and st["session"]["bursts"][1]["text"] == ""
    assert st["session"]["report"]["_meta"]["transcription"][1]["no_speech"]


class FakeRecorder:
    def __init__(self, out, geo):
        self.out = Path(out); self.geo = geo; self.started_wall = None; self.events = []

    def start(self, warm_up_s=0.0):
        self.started_wall = time.time(); self.events.append(("start", time.time()))
        self.out.parent.mkdir(parents=True, exist_ok=True); self.out.write_bytes(b"mp4" * 100)

    def stop(self, tail_s=0.0):
        self.events.append(("stop", time.time())); self.tail_s = tail_s
        return {"path": str(self.out), "bytes": 300, "wall_s": 1.0, "ffmpeg_exit": 0, "duration_s": 2.5}


class FakeWindow:
    def __init__(self):
        from ems_handoff.recording import Geometry
        self.geometry = Geometry(0, 69, 1280, 720); self.wid = "0x1"


def test_record_option_runs_ffmpeg_around_the_bursts_and_checks_sync(settings, tmp_path, monkeypatch):
    made = {}
    checks = []
    sink = FakeSink()
    client, ctl, _ = build(settings, tmp_path, sink=sink)
    ctl.settings = dataclasses.replace(ctl.settings, replay_record_dir=tmp_path / "rec", replay_sink_lead_ms=50.0,
                                       replay_record_tail_s=0.25)
    ctl.engines.make_browser = lambda s, url: made.setdefault("window", FakeWindow())
    ctl.engines.make_recorder = lambda s, out, geo: made.setdefault("recorder", FakeRecorder(out, geo))
    ctl.engines.check_sync = lambda path, expected: (checks.append((path, expected)) or
                                                    {"pass": True, "flashes": expected, "clicks": expected, "expected_bursts": expected,
                                                     "max_abs_offset_ms": 12.0, "bursts": []})
    r = client.post("/api/replay/start", json={"callset": "CS900", "record": True}).get_json()
    assert r["replay"]["recording"]["state"] == "prepared" and r["replay"]["recording"]["geometry"] == [0, 69, 1280, 720]
    assert wait_for(lambda: ctl.state == "idle", timeout=15)
    assert wait_for(lambda: (ctl._replay.recording or {}).get("sync") is not None, timeout=10)
    rec = ctl._replay.recording
    recorder = made["recorder"]
    assert [e[0] for e in recorder.events] == ["start", "stop"]
    assert recorder.events[0][1] <= sink.played[0][0], "ffmpeg was running before burst 1's audio"
    assert recorder.events[1][1] >= ctl._replay.driver.finished_wall, "and stopped after the replay finished"
    assert recorder.tail_s == 0.25, "the recording holds the final report for the configured tail"
    assert rec["state"] == "done" and rec["bytes"] == 300 and rec["duration_s"] == 2.5 and rec["sync"]["pass"]
    assert checks == [(Path(rec["path"]), 2)]
    folder = Path(rec["folder"])
    prov = ctl._replay.status()["provider"]
    assert folder.name.endswith(ctl._replay.status()["commit"]) and (folder / f"CS900.{prov}.summary.json").exists()
    assert Path(rec["path"]).name == f"CS900.{prov}.mp4", "the provider is in the name: both providers' recordings share a folder"
    assert json.loads((folder / f"CS900.{prov}.sync.json").read_text())["max_abs_offset_ms"] == 12.0
    index = (tmp_path / "rec" / "INDEX.md").read_text()
    assert "| CS900 | groq |" in index and "pass (2/2 bursts, worst 12 ms) | pending |" in index
    assert all(50 <= e["sink_lead_ms"] <= 200 for e in ctl._replay.driver.log), "the sink lead was applied"


def test_a_recorder_that_fails_to_start_returns_the_app_to_idle(settings, tmp_path):
    client, ctl, _ = build(settings, tmp_path)
    ctl.engines.make_browser = lambda s, url: (_ for _ in ()).throw(RuntimeError("no display"))
    r = client.post("/api/replay/start", json={"callset": "CS900", "record": True})
    assert r.status_code == 409 and "no display" in r.get_json()["message"]
    assert ctl.state == "idle" and "Replay could not start" in ctl.last_error
    assert client.post("/api/replay/start", json={"callset": "CS900"}).status_code == 200, "a plain replay still starts"
    assert wait_for(lambda: ctl.state == "idle", timeout=15)
