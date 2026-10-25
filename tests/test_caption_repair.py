"""Captions with repair: Vosk's words now, Whisper's words a few seconds later, in place."""

import dataclasses
import threading
import time

import numpy as np

from conftest import wait_for
from ems_handoff.audio.listeners import RepairingCaptioner, make_listeners
from ems_handoff.audio.session import LiveSession
from ems_handoff.audio.sources import FileAudioSource
from fakes import RATE, ScriptedCaptioner, tone

BLOCK = tone(0.1)


class SlowWhisper:
    """Answers each chunk after a delay, from a queue of texts; raises when told to."""

    def __init__(self, texts, delay=0.05, fail=None):
        self.texts = list(texts)
        self.delay = delay
        self.fail = fail
        self.calls = []
        self.gate = threading.Event()
        self.gate.set()

    def __call__(self, audio):
        self.calls.append(len(audio) / RATE)
        self.gate.wait(timeout=5)
        time.sleep(self.delay)
        if self.fail and len(self.calls) == self.fail[0]:
            raise RuntimeError(self.fail[1])
        return self.texts.pop(0) if self.texts else ""


def feed(cap, seconds):
    upd = None
    for _ in range(int(seconds * 10)):
        upd = cap.feed(BLOCK)
    return upd


def test_rough_words_appear_at_once_and_are_replaced_in_place():
    base = ScriptedCaptioner("many seven thirty one sixty two year old female chest pain", blocks_per_word=3)
    whisper = SlowWhisper(["Medic 731,", "62-year-old female, chest pain."])
    cap = RepairingCaptioner(base, whisper, min_chunk_s=1.0, max_chunk_s=3.0)
    upd = feed(cap, 1.2)
    assert upd.final.startswith("many seven"), "Vosk's words are on screen immediately"
    assert upd.segments and not upd.segments[0]["fixed"]
    # the first cut happens at the first new final after 1.0 s; Whisper answers; the next feed shows it
    assert wait_for(lambda: cap.segments and cap.segments[0].status == "fixed", timeout=3)
    upd = cap.feed(BLOCK)
    assert upd.changed and upd.segments[0]["fixed"] and upd.segments[0]["text"] == "Medic 731,"
    assert upd.final.startswith("Medic 731,")
    assert whisper.calls[0] >= 1.0, "the chunk sent carries the audio of that stretch"
    feed(cap, 4.0)
    assert wait_for(lambda: len(cap.segments) >= 2 and cap.segments[1].status == "fixed", timeout=3)
    upd = cap.feed(BLOCK)
    assert "62-year-old female, chest pain." in upd.final
    assert all(s["fixed"] for s in upd.segments[:2])
    cap.close()


def test_forced_cut_at_max_chunk_and_a_failure_keeps_the_rough_words():
    base = ScriptedCaptioner("one two three four five six seven eight nine ten eleven twelve", blocks_per_word=100)
    whisper = SlowWhisper(["fixed words"], fail=(1, "groq rate limit reached (HTTP 429)"))
    cap = RepairingCaptioner(base, whisper, min_chunk_s=1.0, max_chunk_s=2.0, backoff_s=0.2)
    feed(cap, 2.0)                                       # no new final yet: the cut comes from max_chunk_s
    assert len(cap.segments) == 1 and cap.segments[0].seconds == 2.0
    assert wait_for(lambda: cap.segments[0].status == "failed", timeout=3)
    upd = cap.feed(BLOCK)
    assert upd.segments[0]["fixed"] is False, "a failed repair leaves the rough words"
    feed(cap, 2.0)                                       # inside the back-off window: skipped, not sent
    assert wait_for(lambda: len(cap.segments) == 2 and cap.segments[1].status == "skipped", timeout=3)
    time.sleep(0.3)
    feed(cap, 2.0)
    assert wait_for(lambda: len(cap.segments) == 3 and cap.segments[2].status == "fixed", timeout=3)
    assert len(whisper.calls) == 2, "the skipped chunk was never sent"
    cap.close()


def test_session_event_carries_the_segments_and_closes_the_captioner(tmp_path):
    import wave
    path = tmp_path / "clip.wav"
    pcm = (tone(2.5) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(RATE); w.writeframes(pcm.tobytes())
    whisper = SlowWhisper(["Medic 731 inbound."])
    caps = []
    closed = []

    def factory():
        c = RepairingCaptioner(ScriptedCaptioner("medic seven thirty one inbound now", 2), whisper, min_chunk_s=1.0)
        orig = c.close
        c.close = lambda: (closed.append(True), orig())
        caps.append(c)
        return c

    events = []
    s = LiveSession(FileAudioSource(path, pace=0), captioner_factory=factory, emit=events.append)
    s.start()
    assert wait_for(lambda: s.is_finished)
    assert any(e.get("segments") for e in events if e["type"] == "caption"), "rough words were shown during the clip"
    s.stop()                                              # ends the caption loop: flush, then close
    assert wait_for(lambda: closed, timeout=3), "the caption loop closes the repair worker"
    assert any(e.get("segments") and e["segments"][0]["fixed"] for e in events if e["type"] == "caption")
    last = [e for e in events if e["type"] == "caption"][-1]
    assert last["segments"][0]["text"] == "Medic 731 inbound." and last["segments"][0]["fixed"]
    assert last["partial"] == "" and s.final_text.startswith("Medic 731 inbound.")
    assert len(whisper.calls) == 2, "the first cut and the flush of the remaining 1.5 s"


def test_flush_sends_the_remaining_audio_and_waits_for_repairs():
    base = ScriptedCaptioner("alpha bravo charlie delta", blocks_per_word=3)
    whisper = SlowWhisper(["Alpha bravo", "charlie delta."], delay=0.2)
    cap = RepairingCaptioner(base, whisper, min_chunk_s=1.0, max_chunk_s=8.0)
    feed(cap, 1.3)
    feed(cap, 0.7)
    upd = cap.flush(timeout=3.0)
    assert len(cap.segments) == 2 and [s.status for s in cap.segments] == ["fixed", "fixed"]
    assert upd.final == "Alpha bravo charlie delta." and upd.partial == "" and all(x["fixed"] for x in upd.segments)
    cap.close()


def test_make_listeners_adds_repair_only_with_a_groq_key(settings, monkeypatch):
    from ems_handoff.audio import listeners as mod
    monkeypatch.setattr(mod, "vosk_available", lambda d: True)
    monkeypatch.setattr(mod, "load_vosk_model", lambda d: object())
    monkeypatch.setattr(mod, "VoskCaptioner", lambda model, phrase: ScriptedCaptioner("x"))
    monkeypatch.setattr(mod, "VoskPhraseSpotter", lambda model, phrase: object())
    cap_f, spot_f, label = make_listeners(settings)                       # the fixture has a Groq key
    assert label.endswith("+ Groq Whisper repair") and isinstance(cap_f(), RepairingCaptioner)
    cap_f().close()
    cap_f, _, label = make_listeners(dataclasses.replace(settings, groq_api_key=""))
    assert "repair" not in label and not isinstance(cap_f(), RepairingCaptioner)
    cap_f, _, label = make_listeners(dataclasses.replace(settings, caption_repair=False))
    assert "repair" not in label
    assert settings.public_dict()["caption_repair"] is True
