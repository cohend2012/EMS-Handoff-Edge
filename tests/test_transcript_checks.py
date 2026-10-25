"""The silence gate and the words-per-second rule around the final transcript."""

import numpy as np

from ems_handoff.transcript_checks import (NOISE_FLOOR_RMS, has_speech, plausible_rate, speech_stats,
                                           transcribe_checked)
from ems_handoff.transcribe import TranscriptionError
from fakes import RATE, FakeTranscriber, tone


class Scripted(FakeTranscriber):
    """Returns the queued texts in order, or raises a queued exception."""

    def __init__(self, replies, name="groq"):
        super().__init__("", name=name)
        self.replies = list(replies)

    def transcribe(self, audio):
        self.calls += 1
        r = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(r, Exception):
            raise r
        return r


def long_audio(seconds=15.0):
    return tone(seconds)


def words(n):
    return " ".join(["word"] * n)


def test_level_statistics_and_the_silence_gate():
    silent = np.zeros(RATE * 3, np.float32)
    assert not has_speech(silent) and speech_stats(silent)["peak_rms"] == 0.0
    assert has_speech(tone(1.0))
    stats = speech_stats(np.concatenate([np.zeros(RATE, np.float32), tone(1.0)]))
    assert stats["seconds"] == 2.0 and 0.4 < stats["active_fraction"] < 0.6
    assert has_speech(tone(0.5, amp=NOISE_FLOOR_RMS * 3))


def test_silence_skips_the_transcriber():
    t = Scripted(["Thank you."])
    r = transcribe_checked(t, None, np.zeros(RATE * 8, np.float32), caption_text="")
    assert r.no_speech and r.text == "" and t.calls == 0
    assert r.notes[0].startswith("no speech detected") and r.meta()["no_speech"]


def test_plausible_rate():
    assert plausible_rate(2, 8.0), "short clips are never judged by rate"
    assert plausible_rate(44, 15.2) and not plausible_rate(2, 15.2)
    assert plausible_rate(6, 12.0) and not plausible_rate(5, 12.0)


def test_implausible_result_is_retried_then_local_is_tried():
    stages = []
    groq = Scripted(["... ...", "..."], name="groq")
    local = Scripted([words(40)], name="local")
    r = transcribe_checked(groq, local, long_audio(15.2), on_stage=stages.append)
    assert groq.calls == 2 and local.calls == 1
    assert r.who == "fake local" and len(r.text.split()) == 40
    assert [a["role"] for a in r.attempts] == ["primary", "retry", "fallback"]
    assert [a["plausible"] for a in r.attempts] == [False, False, True]
    assert stages[0].startswith("Transcript looks incomplete (2 words in 15 s)")
    assert any("implausible" in n for n in r.notes)


def test_plausible_primary_is_not_retried_and_fallback_is_not_called():
    groq, local = Scripted([words(40)]), Scripted([words(50)], name="local")
    r = transcribe_checked(groq, local, long_audio(15.0))
    assert groq.calls == 1 and local.calls == 0 and r.who == "fake groq"
    assert len(r.attempts) == 1


def test_short_clip_with_few_words_is_accepted_as_is():
    groq = Scripted(["Copy, clear."])
    r = transcribe_checked(groq, Scripted([words(9)], name="local"), tone(3.0))
    assert groq.calls == 1 and r.text == "Copy, clear." and r.attempts[0]["plausible"]


def test_everything_implausible_keeps_the_longest():
    groq = Scripted(["one two", "one two three"], name="groq")
    local = Scripted(["one"], name="local")
    r = transcribe_checked(groq, local, long_audio(20.0))
    assert r.text == "one two three" and r.who == "fake groq"
    assert any(n.startswith("kept the longest transcript (3 words") for n in r.notes)


def test_errors_and_empty_results_fall_through_to_captions():
    groq = Scripted([TranscriptionError("groq is down")], name="groq")
    local = Scripted([""], name="local")
    stages = []
    r = transcribe_checked(groq, local, tone(2.0), caption_text="medic twenty one inbound now", on_stage=stages.append,
                           caption_min_words=4)
    assert r.who == "live captions" and r.text.startswith("medic")
    assert r.attempts[0]["error"] == "groq is down" and r.attempts[1]["words"] == 0
    assert groq.calls == 1, "an error is not retried"
    r = transcribe_checked(groq, local, tone(2.0), caption_text="hi", caption_min_words=4)
    assert not r.ok and r.notes[-1] == "no transcript"


def test_same_transcriber_as_fallback_is_not_tried_twice():
    local = Scripted(["one two"], name="local")
    r = transcribe_checked(local, local, long_audio(15.0))
    assert local.calls == 2, "primary and its retry only"
    assert [a["role"] for a in r.attempts] == ["primary", "retry"]
