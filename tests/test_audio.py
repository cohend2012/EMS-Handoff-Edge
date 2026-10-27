"""Mic capture, listeners and the live session, with a fake mic (no hardware)."""

import numpy as np
import pytest

from conftest import wait_for
from ems_handoff.audio.capture import MicError, find_input_device, resample
from ems_handoff.audio.listeners import WhisperCaptioner
from ems_handoff.audio.session import LiveSession
from fakes import RATE, AutoFeedCapture, ScriptedCaptioner, TriggerSpotter, fake_capture, tone

DEVICES = [
    {"name": "HDMI 0", "max_input_channels": 0, "default_samplerate": 48000},
    {"name": "APE in", "max_input_channels": 16, "default_samplerate": 48000},
    {"name": "USB PnP Audio Device: Audio (hw:2,0)", "max_input_channels": 1, "default_samplerate": 48000},
    {"name": "default", "max_input_channels": 128, "default_samplerate": 44100},
]


# -- device selection -------------------------------------------------------
def test_auto_prefers_the_usb_mic():
    assert find_input_device("", DEVICES)[:2] == (2, "USB PnP Audio Device: Audio (hw:2,0)")


def test_select_by_index_or_name():
    assert find_input_device("3", DEVICES)[0] == 3
    assert find_input_device("pnp", DEVICES)[0] == 2


def test_bad_selection_explains_the_choices():
    with pytest.raises(MicError, match="Inputs: 1: APE in"):
        find_input_device("0", DEVICES)        # HDMI has no input
    with pytest.raises(MicError, match="No input device name contains"):
        find_input_device("blue yeti", DEVICES)
    with pytest.raises(MicError, match="No microphone found"):
        find_input_device("", DEVICES[:1])


def test_resample_48k_to_16k_keeps_duration():
    x = np.zeros(4800, np.float32)
    assert len(resample(x, 48000)) == 1600
    assert len(resample(x, 16000)) == 4800


# -- capture --------------------------------------------------------------------
def test_capture_fans_out_and_keeps_full_recording():
    cap = fake_capture()
    q1, q2 = cap.subscribe(), cap.subscribe()
    cap.start()
    for _ in range(5):
        cap.feed(tone(0.1))
    assert wait_for(lambda: q1.qsize() == 5 and q2.qsize() == 5)
    audio = cap.stop()
    assert len(audio) == 5 * 1600
    assert cap.level > 0.01
    # listeners are told the stream ended
    assert [q1.get() for _ in range(6)][-1] is None


def test_capture_high_pass_removes_mains_hum_but_keeps_speech_band():
    """The Jetson's USB mic delivers 60 Hz hum well above speech; the capture cuts it before anyone
    sees the audio, block by block without a seam, and leaves a 1 kHz tone alone. 0 switches it off."""
    from fakes import FakeStream
    from ems_handoff.audio.capture import AudioCapture

    def band(x, lo, hi):
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        f = np.fft.rfftfreq(len(x), 1 / 16000)
        return float(sp[(f >= lo) & (f <= hi)].sum())

    t = np.arange(16000) / 16000
    signal = (0.3 * np.sin(2 * np.pi * 60 * t) + 0.03 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    out = {}
    for hz in (150.0, 0.0):
        cap = AudioCapture("", stream_factory=lambda *a: FakeStream(), device_info=(0, "USB", 16000.0), highpass_hz=hz)
        cap.start()
        for i in range(10):                      # ten 100 ms blocks, as the mic delivers them
            cap.feed(signal[i * 1600:(i + 1) * 1600])
        assert wait_for(lambda: cap.seconds >= 1.0)
        out[hz] = cap.stop()
    filtered, raw = out[150.0][1600:], out[0.0][1600:]        # skip the filter's settling block
    assert band(raw, 55, 65) / band(raw, 990, 1010) > 50, "the hum dominates the unfiltered audio"
    assert band(filtered, 55, 65) / band(raw, 55, 65) < 1e-3, "hum down by more than 30 dB"
    assert 0.7 < band(filtered, 990, 1010) / band(raw, 990, 1010) < 1.3, "the speech-band tone is kept"
    assert np.abs(np.diff(filtered)).max() < 0.02, "no seam between blocks"


def test_capture_opens_native_rate_first_and_resamples():
    opened = []

    def factory(dev, rate, block, cb):
        opened.append(rate)
        if rate == 16000:
            raise RuntimeError("Invalid sample rate")
        from fakes import FakeStream
        return FakeStream()

    from ems_handoff.audio.capture import AudioCapture
    cap = AudioCapture("", stream_factory=factory, device_info=(2, "USB", 48000.0))
    cap.start()
    cap.feed(np.zeros(4800, np.float32))          # 100 ms at 48 kHz
    assert wait_for(lambda: cap.seconds > 0)
    assert opened == [48000] and cap.native_rate == 48000   # no failed 16 kHz attempt
    assert len(cap.stop()) == 1600


def test_capture_tries_16k_when_native_rate_fails():
    opened = []

    def factory(dev, rate, block, cb):
        opened.append(rate)
        if rate == 44100:
            raise RuntimeError("busy")
        from fakes import FakeStream
        return FakeStream()

    from ems_handoff.audio.capture import AudioCapture
    cap = AudioCapture("", stream_factory=factory, device_info=(2, "Mic", 44100.0))
    cap.start()
    assert opened == [44100, 16000] and cap.native_rate == 16000
    cap.stop()


# -- live session -------------------------------------------------------------------
def make_session(captioner=None, spotter=None, max_seconds=30):
    events, finished = [], []
    cap = fake_capture()
    s = LiveSession(cap, (lambda: captioner) if captioner else None,
                    (lambda: spotter) if spotter else None,
                    emit=events.append, on_finish=finished.append, max_seconds=max_seconds,
                    captions_label="test")
    return s, cap, events, finished


def test_captions_stream_to_the_screen():
    s, cap, events, finished = make_session(ScriptedCaptioner("Medic 21 inbound with male", 1))
    s.start()
    for _ in range(5):
        cap.feed(tone(0.1))
    assert wait_for(lambda: any(e["type"] == "caption" and "inbound" in e["final"] + e["partial"]
                                for e in events))
    assert any(e["type"] == "listening" and e["mic"] == "Fake USB mic" for e in events)
    s.finish("button")
    assert finished == ["button"]
    assert len(s.stop()) == 5 * 1600


def test_spotter_finishes_once_even_if_it_fires_again():
    s, cap, events, finished = make_session(spotter=TriggerSpotter(after_blocks=3))
    s.start()
    for _ in range(6):
        cap.feed(tone(0.1))
    assert wait_for(lambda: finished == ["phrase"])
    assert s.finish("button") is False and finished == ["phrase"]
    s.stop()


def test_phrase_in_captions_also_finishes():
    s, cap, _, finished = make_session(ScriptedCaptioner("ETA five confirm report ready", 1,
                                                         phrase="confirm report ready"))
    s.start()
    for _ in range(8):
        cap.feed(tone(0.1))
    assert wait_for(lambda: finished == ["phrase"])
    s.stop()


def test_timeout_finishes_and_keeps_the_audio():
    s, cap, events, finished = make_session(max_seconds=0.3)
    s.start()
    cap.feed(tone(0.2))
    assert wait_for(lambda: finished == ["timeout"], timeout=3)
    assert len(s.stop()) == int(0.2 * RATE)
    assert any(e["type"] == "level" for e in events)


class ExplodingCaptioner:
    def feed(self, block):
        raise ValueError("model crashed")


def test_caption_crash_does_not_stop_recording():
    s, cap, _, finished = make_session(ExplodingCaptioner(), TriggerSpotter(after_blocks=4))
    s.start()
    for _ in range(5):
        cap.feed(tone(0.1))
    assert wait_for(lambda: finished == ["phrase"])     # spotter still works
    assert len(s.stop()) == 5 * 1600                    # nothing lost


def test_real_time_fake_mic_session():
    cap = AutoFeedCapture()
    finished = []
    s = LiveSession(cap, None, None, on_finish=finished.append)
    s.start()
    assert wait_for(lambda: cap.seconds >= 0.5)
    s.finish("button")
    assert len(s.stop()) >= int(0.5 * RATE)


# -- whisper fallback captions ---------------------------------------------------
def test_whisper_fallback_covers_every_sample_and_commits():
    seen = []

    def fake_whisper(audio):
        seen.append(len(audio))
        return f"chunk{len(seen)}"

    c = WhisperCaptioner(None, phrase="confirm report ready", step_s=1.0, commit_s=3.0,
                         transcribe_fn=fake_whisper)
    updates = [c.feed(tone(0.5)) for _ in range(10)]          # 5 s of audio
    # each pass transcribes everything since the last commit: no blind spots
    assert seen[0] == 1 * RATE and seen[1] == 2 * RATE and seen[2] == 3 * RATE
    assert updates[-1].final.startswith("chunk3")             # committed after 3 s
    assert seen[3] == RATE // 2 + RATE                        # 0.5 s overlap + new second


def test_whisper_fallback_hears_the_phrase():
    c = WhisperCaptioner(None, phrase="confirm report ready", step_s=0.5,
                         transcribe_fn=lambda a: "ETA five minutes confirm report ready")
    assert c.feed(tone(0.5)).heard_phrase


def test_missing_caption_model_records_anyway():
    def broken():
        raise OSError("model not downloaded and no internet")

    events, finished = [], []
    cap = fake_capture()
    s = LiveSession(cap, broken, broken, emit=events.append, on_finish=finished.append)
    s.start()                                            # must not raise
    cap.feed(tone(0.6))
    listening = next(e for e in events if e["type"] == "listening")
    assert listening["captions"] == "off" and not listening["spotter"]
    assert "Live captions unavailable" in listening["problems"][0]
    s.finish("button")
    assert len(s.stop()) == int(0.6 * RATE)
