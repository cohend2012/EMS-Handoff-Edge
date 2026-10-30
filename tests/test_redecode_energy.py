"""Idea 4 of the Lyapunov notes: the span re-decode has a discrete energy, the number of windows it would flag, and one
pass drives that energy to zero: a second pass over its own output flags nothing and decodes nothing. This is the
termination and fixed-point property the paper may claim for the re-decode; it is a property of our algorithm, not of
Whisper."""
import dataclasses

import numpy as np

from ems_handoff.config import load_settings
from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE


class Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


def _loud_audio(seconds=60, quiet_from=40):
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(seconds * SAMPLE_RATE)).astype(np.float32)
    audio[quiet_from * SAMPLE_RATE:] *= 0.01
    return audio


def test_one_pass_reaches_a_fixed_point_when_the_span_comes_back_whole():
    class Model:
        def __init__(self):
            self.calls = 0
        def transcribe(self, audio, **kw):
            self.calls += 1
            if len(audio) > 50 * SAMPLE_RATE:       # the first pass: a looped window then a real tail
                return iter([Seg(2.0 * i, 2.0 * i + 2.0, "Priority 1, protocol.") for i in range(14)] + [Seg(30.0, 40.0, "blurry vision, LAMS zero")]), None
            return iter([Seg(0.0, 28.0, "Shady Grove, this is Medic 715, 57-year-old male, stroke alert by protocol.")]), None
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    m = Model(); t._model = m
    x = _loud_audio()
    first = list(t._decode(x, dict(temperature=0.0)))
    out = t._redecode_gaps(x, first, dict(temperature=0.0))
    assert m.calls == 2, "exactly one span was flagged and decoded"       # energy 1 -> 0 in one pass
    calls_after_pass = m.calls
    again = t._redecode_gaps(x, out, dict(temperature=0.0))
    assert m.calls == calls_after_pass, "a second pass over the output flags no span and decodes nothing"
    assert [s.text for s in again] == [s.text for s in out], "the output is a fixed point"


def test_one_pass_reaches_a_fixed_point_when_the_span_loops_again_and_one_copy_is_kept():
    class Model:
        def __init__(self):
            self.calls = 0
        def transcribe(self, audio, **kw):
            self.calls += 1
            if len(audio) > 50 * SAMPLE_RATE:
                return iter([Seg(2.0 * i, 2.0 * i + 2.0, "Priority 1, protocol.") for i in range(14)] + [Seg(30.0, 40.0, "blurry vision, LAMS zero")]), None
            return iter([Seg(0.0, 28.0, "Priority 1, protocol. " * 14)]), None     # the span loops on its own too
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    m = Model(); t._model = m
    x = _loud_audio()
    out = t._redecode_gaps(x, list(t._decode(x, dict(temperature=0.0))), dict(temperature=0.0))
    assert m.calls == 2
    assert " ".join(s.text.strip() for s in out) == "Priority 1, protocol. blurry vision, LAMS zero"
    again = t._redecode_gaps(x, out, dict(temperature=0.0))
    assert m.calls == 2 and [s.text for s in again] == [s.text for s in out]
