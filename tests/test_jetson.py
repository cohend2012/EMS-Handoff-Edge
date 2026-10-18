"""
Real-device tests. Run them on vox-ai with the USB mic plugged in:

    pytest -m jetson -s

They use the real .env, mic, models and network, so they are skipped elsewhere.
Each test prints what it measured, so the output doubles as a setup report.
"""

import time

import numpy as np
import pytest

from ems_handoff.config import PROJECT_ROOT, load_settings

pytestmark = pytest.mark.jetson


@pytest.fixture(scope="module")
def real_settings():
    from ems_handoff.config import load_env_files
    load_env_files()
    return load_settings()


def test_mic_records_three_seconds(real_settings):
    from ems_handoff.audio.capture import AudioCapture
    cap = AudioCapture(real_settings.mic_device)
    cap.start()
    print(f"\nRecording 3 s from '{cap.device_name}' at {cap.native_rate} Hz. Say something...")
    time.sleep(3)
    audio = cap.stop()
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    print(f"Captured {len(audio) / 16000:.2f} s, RMS {rms:.4f}, overflows {cap.overflows}")
    assert 2.5 < len(audio) / 16000 < 3.6
    assert rms > 0.0005, "The mic delivered silence. Check the gain in alsamixer, or pick another device."


def test_caption_engine_runs_faster_than_real_time(real_settings):
    from ems_handoff.audio.listeners import make_listeners
    cap_f, spot_f, label = make_listeners(real_settings)
    assert cap_f is not None, "No caption engine available"
    captioner = cap_f()
    noise = (np.random.default_rng(0).standard_normal(16000 * 5) * 0.01).astype(np.float32)
    t0 = time.perf_counter()
    for i in range(0, len(noise), 1600):
        captioner.feed(noise[i:i + 1600])
    took = time.perf_counter() - t0
    print(f"\n{label}: 5 s of audio processed in {took:.2f} s ({5 / took:.1f}x real time)")
    assert took < 5, "Captions cannot keep up with live speech on this device"


def test_ai_provider_is_ready(real_settings):
    from ems_handoff.llm import LLMClient
    ok, msg = LLMClient(real_settings).check()
    print(f"\n{msg}")
    assert ok, msg


def test_text_report_end_to_end(real_settings, tmp_path):
    from ems_handoff.cli import main
    t0 = time.perf_counter()
    assert main(["--case", "quick"]) == 0
    print(f"Quick demo report in {time.perf_counter() - t0:.1f} s with {real_settings.provider}")


def test_final_transcription_speed(real_settings):
    from ems_handoff.transcribe import make_transcriber
    from ems_handoff.samples import SAMPLE_STEMI  # noqa: F401  (documentation: read this aloud for a real test)
    t = make_transcriber(real_settings)
    tone = (0.05 * np.sin(2 * np.pi * 220 * np.arange(16000 * 10) / 16000)).astype(np.float32)
    t0 = time.perf_counter()
    text = t.transcribe(tone)
    took = time.perf_counter() - t0
    print(f"\n{t.describe()}: 10 s of audio in {took:.1f} s -> {text!r}")
    assert took < 60
