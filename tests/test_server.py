"""The app server end to end through its HTTP API, with fake engines."""

import json

import pytest

from conftest import wait_for
from ems_handoff.llm import LLMError
from ems_handoff.server import Engines, create_app, load_user_settings
from fakes import AutoFeedCapture, FakeLLM, FakeTranscriber, ScriptedCaptioner, TriggerSpotter

SCRIPT = "Medic 21 to General. Inbound 45 year old male found down. GCS 14. ETA 5 minutes."


def build(settings, llm=None, transcriber=None, fallback=None, spotter_after=None, captions=True,
          capture=None):
    llm = llm or FakeLLM()
    transcriber = transcriber or FakeTranscriber(SCRIPT)

    def listeners(_s):
        cap_f = (lambda: ScriptedCaptioner(SCRIPT, 1)) if captions else None
        spot_f = (lambda: TriggerSpotter(spotter_after)) if spotter_after else None
        return cap_f, spot_f, "scripted captions"

    engines = Engines(make_llm=lambda s: llm, make_transcriber=lambda s: transcriber,
                      make_fallback_transcriber=(lambda s: fallback) if fallback else None,
                      make_capture=lambda s: capture or AutoFeedCapture(),
                      make_listeners=listeners,
                      list_mics=lambda: [(2, "USB PnP Audio Device", 48000.0)])
    app = create_app(settings, engines)
    return app, app.test_client(), app.config["controller"]


def test_status_samples_and_mics(settings):
    _, client, _ = build(settings)
    st = client.get("/api/status").get_json()
    assert st["state"] == "idle" and st["settings"]["provider"] == "groq"
    assert "test-key" not in json.dumps(st)
    assert [s["id"] for s in client.get("/api/samples").get_json()] == ["quick", "stemi", "trauma", "sepsis"]
    assert client.get("/api/mics").get_json()["devices"][0]["name"] == "USB PnP Audio Device"
    assert client.get("/").status_code == 200
    assert b"HANDOFF.AI" in client.get("/web/app.js").data or client.get("/web/app.js").status_code == 200


def test_text_report_flow(settings):
    _, client, ctl = build(settings)
    assert client.post("/api/report/text", json={"text": SCRIPT}).get_json()["ok"]
    assert wait_for(lambda: ctl.state == "done")
    report = client.get("/api/result").get_json()["report"]
    assert report["alert_type"] == "STEMI" and report["_meta"]["source"] == "text"
    assert (settings.output_dir / "handoff_report.json").exists()
    assert len(list((settings.output_dir / "reports").glob("handoff-*.json"))) == 1


def test_empty_text_is_refused(settings):
    _, client, _ = build(settings)
    r = client.post("/api/report/text", json={"text": "  "})
    assert r.status_code == 400 and "Paste or choose" in r.get_json()["message"]


def test_live_done_button_flow(settings):
    transcriber = FakeTranscriber(SCRIPT)
    _, client, ctl = build(settings, transcriber=transcriber)
    assert client.post("/api/live/start").get_json()["ok"]
    assert ctl.state == "recording"
    assert wait_for(lambda: ctl.session is not None and ctl.session.capture.seconds >= 0.5)
    assert client.post("/api/live/done").get_json()["ok"]
    assert wait_for(lambda: ctl.state == "done")
    assert transcriber.calls == 1 and transcriber.last_audio_len >= 8000
    report = ctl.last_result
    assert report["_meta"]["source"] == "live" and report["_meta"]["timings"]["audio_s"] >= 0.5
    assert (settings.output_dir / "recording.wav").exists()


def test_live_stop_phrase_flow(settings):
    _, client, ctl = build(settings, spotter_after=5)
    client.post("/api/live/start")
    assert wait_for(lambda: ctl.state == "done", timeout=8)
    assert ctl.last_result["_meta"]["source"] == "live"


def test_cancel_discards_and_returns_to_idle(settings):
    transcriber = FakeTranscriber()
    _, client, ctl = build(settings, transcriber=transcriber)
    client.post("/api/live/start")
    assert wait_for(lambda: ctl.session.capture.seconds > 0.2)
    client.post("/api/live/cancel")
    assert ctl.state == "idle" and ctl.session is None
    import time
    time.sleep(0.3)
    assert transcriber.calls == 0 and ctl.last_result is None


def test_cannot_start_twice_or_finish_when_idle(settings):
    _, client, ctl = build(settings)
    assert client.post("/api/live/done").status_code == 409
    client.post("/api/live/start")
    r = client.post("/api/live/start")
    assert r.status_code == 409 and "Busy" in r.get_json()["message"]
    client.post("/api/live/cancel")


def test_groq_down_falls_back_to_local_transcriber(settings):
    primary = FakeTranscriber(fail=True, name="groq")
    local = FakeTranscriber(SCRIPT, name="local")
    _, client, ctl = build(settings, transcriber=primary, fallback=local)
    client.post("/api/live/start")
    assert wait_for(lambda: ctl.session.capture.seconds >= 0.7)
    client.post("/api/live/done")
    assert wait_for(lambda: ctl.state == "done")
    assert primary.calls == 1 and local.calls == 1


def test_all_transcribers_down_uses_live_captions(settings):
    _, client, ctl = build(settings, transcriber=FakeTranscriber(fail=True))
    client.post("/api/live/start")
    assert wait_for(lambda: len(ctl.session.final_text.split()) >= 6, timeout=8)
    client.post("/api/live/done")
    assert wait_for(lambda: ctl.state == "done")
    assert ctl.last_result["_meta"]["raw_transcript"].startswith("Medic 21")


def test_no_transcript_and_no_captions_is_a_clear_error(settings):
    _, client, ctl = build(settings, transcriber=FakeTranscriber(fail=True), captions=False)
    client.post("/api/live/start")
    assert wait_for(lambda: ctl.session.capture.seconds >= 0.7)
    client.post("/api/live/done")
    assert wait_for(lambda: ctl.state == "error")
    assert "Could not transcribe" in ctl.last_error


def test_llm_failure_is_reported_and_app_recovers(settings):
    llm = FakeLLM([LLMError("groq rejected the API key (HTTP 401). Create a new key and update .env.")])
    _, client, ctl = build(settings, llm=llm)
    client.post("/api/report/text", json={"text": SCRIPT})
    assert wait_for(lambda: ctl.state == "error")
    assert "rejected the API key" in client.get("/api/status").get_json()["error"]
    llm.replies = [json.dumps(__import__("fakes").report_dict())]
    client.post("/api/report/text", json={"text": SCRIPT})     # can try again right away
    assert wait_for(lambda: ctl.state == "done")


def test_settings_change_is_saved_and_reloaded(settings, tmp_path):
    _, client, ctl = build(settings)
    r = client.post("/api/settings", json={"provider": "ollama", "mic_device": "PnP"}).get_json()
    assert r["status"]["settings"]["provider"] == "ollama"
    assert client.post("/api/settings", json={"provider": "skynet"}).status_code == 400
    reloaded = load_user_settings(settings)
    assert reloaded.provider == "ollama" and reloaded.mic_device == "PnP"


def test_provider_check_endpoint(settings):
    _, client, _ = build(settings)
    assert client.post("/api/check").get_json() == {"ok": True, "message": "groq ready (fake-model)"}


def test_event_stream_sends_status_then_live_events(settings):
    app, client, ctl = build(settings)
    resp = client.get("/api/events", buffered=False)
    chunks = iter(resp.response)
    first = json.loads(next(chunks).decode().removeprefix("data: ").strip())
    assert first["type"] == "status" and first["status"]["state"] == "idle"
    client.post("/api/report/text", json={"text": SCRIPT})
    seen = []
    for raw in chunks:
        line = raw.decode().strip()
        if line.startswith("data: "):
            seen.append(json.loads(line[6:])["type"])
        if "result" in seen:
            break
    assert seen[0] == "state" and "stage" in seen and seen[-1] == "result"
    resp.close()


def test_quit_only_from_launcher(settings):
    _, client, _ = build(settings)
    assert client.post("/api/quit").status_code == 409


def test_unexpected_errors_come_back_as_json(settings):
    app, client, ctl = build(settings)

    def boom():
        raise ZeroDivisionError("bug")
    ctl.start_live = boom
    r = client.post("/api/live/start")
    assert r.status_code == 500 and r.get_json()["message"].startswith("Unexpected error: ZeroDivisionError")


def test_warm_up_loads_listeners_once(settings):
    calls = []
    app, client, ctl = build(settings)
    original = ctl.engines.make_listeners
    ctl.engines.make_listeners = lambda s: calls.append(1) or original(s)
    ctl.warm_up()
    client.post("/api/live/start")
    client.post("/api/live/cancel")
    assert calls == [1] and ctl.captions_label == "scripted captions"
