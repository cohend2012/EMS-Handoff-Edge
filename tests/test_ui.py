"""
Drive the real app screens in headless Chromium, with fake engines behind the server.

Checks what a user sees: sample -> report, live captions appearing while
"talking", the transcript box being at least as big as the script box, DONE ->
report, and a readable error. Screenshots go to tests/artifacts/.
"""

import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from ems_handoff.llm import LLMError
from ems_handoff.server import Engines, create_app
from fakes import AutoFeedCapture, FakeLLM, FakeTranscriber, ScriptedCaptioner, report_dict

pytestmark = pytest.mark.ui
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
SCRIPT = ("Medic 21 to General Hospital. We are inbound with a 45-year-old male found down. "
          "GCS is 14. History of seizures. Vitals BP 140 over 90, heart rate 110, SpO2 98 percent. "
          "ETA is 5 minutes.")


def _chromium():
    for path in ("/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
                 shutil.which("chromium") or "", shutil.which("chromium-browser") or ""):
        if path and os.path.exists(path):
            return path
    return None


@pytest.fixture
def app_url(settings):
    from werkzeug.serving import make_server
    llm = FakeLLM(delay=0.8)
    engines = Engines(make_llm=lambda s: llm, make_transcriber=lambda s: FakeTranscriber(SCRIPT),
                      make_fallback_transcriber=None, make_capture=lambda s: AutoFeedCapture(),
                      make_listeners=lambda s: ((lambda: ScriptedCaptioner(SCRIPT, 1)), None, "Vosk (test)"),
                      list_mics=lambda: [(24, "USB PnP Audio Device: Audio (hw:2,0)", 48000.0)])
    app = create_app(settings, engines)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/", llm
    server.shutdown()


@pytest.fixture
def page():
    exe = _chromium()
    if not exe:
        pytest.skip("Chromium not available")
    from playwright.sync_api import sync_playwright
    ARTIFACTS.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        pg = browser.new_page(viewport={"width": 1366, "height": 860})
        errors = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        yield pg
        browser.close()
        assert not errors, f"JavaScript errors: {errors}"


def test_sample_case_to_dashboard(app_url, page):
    url, _ = app_url
    page.goto(url)
    page.wait_for_selector("#sampleRow button")
    page.screenshot(path=str(ARTIFACTS / "1-home.png"))
    page.click("#sampleRow button[data-sample='stemi']")
    assert "Unit 41" in page.input_value("#reportText")
    page.click("#textBtn")
    page.wait_for_selector("#screen-working:not([hidden])")
    page.wait_for_selector(".dash-banner", timeout=10000)
    assert "STEMI ALERT" in page.inner_text(".dash-banner")
    assert "Activate cath lab" in page.inner_text("#dashboard")
    page.screenshot(path=str(ARTIFACTS / "4-report.png"), full_page=True)


def test_live_recording_captions_done_and_report(app_url, page):
    url, _ = app_url
    page.goto(url)
    page.wait_for_selector("#scriptSelect option[value='quick']", state="attached")
    page.click("#startBtn")
    page.wait_for_selector("#screen-live:not([hidden])")
    # words appear while "talking"
    page.wait_for_function("document.querySelector('#liveTranscript').innerText.includes('inbound')",
                           timeout=8000)
    page.wait_for_function("document.querySelector('#recTime').innerText !== '0:00'", timeout=5000)
    assert "Fake USB mic" in page.inner_text("#micLine")
    assert "Vosk (test)" in page.inner_text("#captionEngine")
    script = page.locator("#scriptPanel").bounding_box()
    live = page.locator(".transcript-panel").bounding_box()
    assert live["width"] >= script["width"] * 0.98 and live["height"] >= script["height"] * 0.98
    assert live["height"] >= 400, "transcript box should be large"
    page.screenshot(path=str(ARTIFACTS / "2-live.png"))
    page.click("#doneBtn")
    page.wait_for_selector("#screen-working:not([hidden])")
    page.wait_for_selector("#stageList li")
    assert page.inner_text("#stageList li:first-child").startswith("Done pressed")
    page.screenshot(path=str(ARTIFACTS / "3-working.png"))
    page.wait_for_selector(".dash-banner", timeout=10000)
    meta = page.inner_text(".meta")
    assert "transcribe" in meta and "report" in meta


def test_error_is_shown_in_plain_words(app_url, page):
    url, llm = app_url
    llm.replies = [LLMError("groq rejected the API key (HTTP 401). Create a new key and update .env.")]
    page.goto(url)
    page.wait_for_selector("#sampleRow button")
    page.click("#sampleRow button[data-sample='quick']")
    page.click("#textBtn")
    page.wait_for_selector("#screen-error:not([hidden])", timeout=10000)
    assert "rejected the API key" in page.inner_text("#errorText")
    page.screenshot(path=str(ARTIFACTS / "5-error.png"))
    page.click("#errorBackBtn")
    assert page.is_visible("#screen-home")
    llm.replies = [json.dumps(report_dict())]


def test_settings_dialog_lists_mics_and_providers(app_url, page):
    url, _ = app_url
    page.goto(url)
    page.click("#settingsBtn")
    page.wait_for_selector("#settingsDialog[open]")
    assert "USB PnP Audio Device" in page.inner_text("#micSelect")
    assert page.locator("input[name=provider]").count() == 3
    page.check("input[name=provider][value=ollama]")
    page.click("#saveSettingsBtn")
    page.wait_for_function("document.querySelector('#providerText').innerText.startsWith('ollama')")
    page.screenshot(path=str(ARTIFACTS / "6-settings-saved.png"))
