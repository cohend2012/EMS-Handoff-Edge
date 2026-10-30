"""Settings parsing and stop-phrase matching."""

from ems_handoff.config import Settings, load_settings
from ems_handoff.phrase import contains_phrase, strip_phrase

PHRASE = "confirm report ready"


def test_defaults_without_env():
    s = load_settings({})
    assert s.provider == "groq"
    assert s.ollama_model == "qwen3:8b"
    assert s.stop_phrase == PHRASE
    assert s.mic_device == ""
    assert s.effective_transcriber() == "local"  # no Groq key -> local Whisper


def test_env_overrides_and_bad_values_fall_back():
    s = load_settings({"HANDOFF_PROVIDER": "OLLAMA", "OLLAMA_MODEL": "qwen3:4b", "GROQ_API_KEY": " gsk_x ",
                       "HANDOFF_MIC_DEVICE": "PnP", "HANDOFF_PORT": "not-a-number",
                       "HANDOFF_CAPTIONS": "nonsense", "HANDOFF_USE_LEXICON": "0"})
    assert s.provider == "ollama" and s.llm_model() == "qwen3:4b"
    assert s.ollama_model_pinned is True and s.public_dict()["model_pinned"] is True
    assert load_settings({}).ollama_model_pinned is False
    assert load_settings({"OLLAMA_MODEL": "  "}).ollama_model == "qwen3:8b"   # blank = unset
    assert s.groq_api_key == "gsk_x"
    assert s.mic_device == "PnP"
    assert s.port == 5050
    assert s.caption_engine == "auto"
    assert s.use_lexicon is False
    assert s.effective_transcriber() == "groq"   # auto + key -> Groq Whisper


def test_public_dict_never_leaks_keys():
    s = Settings(groq_api_key="gsk_secret", openrouter_api_key="or_secret")
    text = repr(s) + str(s.public_dict())
    assert "secret" not in text
    assert s.public_dict()["keys"] == {"groq": True, "ollama": True, "openrouter": True}


def test_phrase_matches_exact_and_misheard():
    assert contains_phrase("ETA five minutes. Confirm report ready.", PHRASE)
    assert contains_phrase("confirm the port ready", PHRASE)
    assert contains_phrase("CONFIRM, REPORT READY!", PHRASE)


def test_phrase_does_not_fire_on_ordinary_report_language():
    for text in ["report is ready for you", "we confirmed the report", "patient ready for transport",
                 "requesting cath lab activation", ""]:
        assert not contains_phrase(text, PHRASE), text


def test_strip_phrase_only_at_the_end():
    assert strip_phrase("ETA is 5 minutes. Confirm report ready.", PHRASE) == "ETA is 5 minutes."
    assert strip_phrase("ETA is 5 minutes. confirm report ready uh", PHRASE) == "ETA is 5 minutes."
    long = "confirm report ready " + "word " * 30 + "end"
    assert strip_phrase(long, PHRASE).startswith("confirm report ready")   # not in the tail -> kept
    assert strip_phrase("no phrase here at all", PHRASE) == "no phrase here at all"


def test_shared_secrets_file_beats_blank_key_in_project_env(tmp_path, monkeypatch):
    import ems_handoff.config as config
    secrets = tmp_path / "secrets.env"
    secrets.write_text("GROQ_API_KEY=gsk_shared123\n")
    project = tmp_path / "proj"
    project.mkdir()
    (project / ".env").write_text("GROQ_API_KEY=\nHANDOFF_PROVIDER=ollama\n")
    monkeypatch.setattr(config, "SECRETS_FILE", secrets)
    monkeypatch.setattr(config, "PROJECT_ROOT", project)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("HANDOFF_PROVIDER", raising=False)
    config.load_env_files()
    s = config.load_settings()
    assert s.groq_api_key == "gsk_shared123" and s.provider == "ollama"


def test_local_whisper_fallback_env(monkeypatch):
    from ems_handoff.config import load_settings
    monkeypatch.delenv("LOCAL_WHISPER_FALLBACK", raising=False)
    assert load_settings().local_whisper_fallback is False
    monkeypatch.setenv("LOCAL_WHISPER_FALLBACK", "1")
    assert load_settings().local_whisper_fallback is True


def test_local_transcriber_fallback_decode_options(monkeypatch):
    import dataclasses
    import numpy as np
    from ems_handoff.config import load_settings
    from ems_handoff.transcribe import LocalTranscriber

    class Seg:
        start, end, text = 0.0, 1.0, "hello"

    calls = []

    class FakeModel:
        def transcribe(self, audio, **kw):
            calls.append(kw)
            return iter([Seg()]), None

    audio = np.zeros(16000, dtype=np.float32)
    for fb in (False, True):
        t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_fallback=fb), model_size="tiny.en")
        t._model = FakeModel()
        assert t.transcribe(audio) == "hello"
    assert calls[0]["temperature"] == 0.0 and "no_speech_threshold" not in calls[0]
    assert calls[1]["temperature"] == [0.0, 0.2, 0.4] and calls[1]["no_speech_threshold"] is None
    assert "fallback" in LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_fallback=True)).describe()


def test_local_transcriber_redecodes_a_long_wordless_gap():
    import dataclasses
    import numpy as np
    from ems_handoff.config import load_settings
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE

    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    calls = []

    class FakeModel:
        def transcribe(self, audio, **kw):
            calls.append(len(audio) / SAMPLE_RATE)
            if len(audio) > 50 * SAMPLE_RATE:          # the whole clip: the middle window came out empty
                return iter([Seg(0.0, 10.0, "vitals are"), Seg(40.0, 45.0, "any questions")]), None
            return iter([Seg(0.0, 29.0, "pressure 80 over 55 sugar 285 sepsis alert")]), None

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(60 * SAMPLE_RATE)).astype(np.float32)   # speech-like level throughout
    audio[45 * SAMPLE_RATE:] *= 0.01                                            # ...and trailing silence
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    t._model = FakeModel()
    text = t.transcribe(audio)
    assert text == "vitals are pressure 80 over 55 sugar 285 sepsis alert any questions"
    assert calls == [60.0, 33.0] and t.last_notes and "10.0-40.0" in t.last_notes[0]   # the span starts 3 s early
    # a quiet gap (a pause) is left alone
    audio[10 * SAMPLE_RATE:40 * SAMPLE_RATE] *= 0.01
    calls.clear(); t.last_notes = []
    assert t.transcribe(audio) == "vitals are any questions" and calls == [60.0]


def test_local_whisper_redecode_defaults_on_and_env_turns_it_off():
    assert load_settings({}).local_whisper_redecode is True
    assert load_settings({"LOCAL_WHISPER_REDECODE_GAPS": "0"}).local_whisper_redecode is False


def test_local_whisper_redecode_uses_beam_5_on_the_span_and_greedy_on_the_clip():
    import dataclasses
    import numpy as np
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE

    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    beams = []

    class FakeModel:
        def transcribe(self, audio, **kw):
            beams.append((round(len(audio) / SAMPLE_RATE), kw["beam_size"]))
            if len(audio) > 50 * SAMPLE_RATE:
                return iter([Seg(0.0, 10.0, "vitals are"), Seg(40.0, 45.0, "any questions")]), None
            return iter([Seg(0.0, 29.0, "pressure 80 over 55")]), None

    assert load_settings({}).local_whisper_redecode_beam == 5
    assert load_settings({"LOCAL_WHISPER_REDECODE_BEAM": "3"}).local_whisper_redecode_beam == 3
    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(60 * SAMPLE_RATE)).astype(np.float32)
    audio[45 * SAMPLE_RATE:] *= 0.01
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    t._model = FakeModel()
    assert t.transcribe(audio) == "vitals are pressure 80 over 55 any questions"
    assert beams == [(60, 1), (33, 5)]


def test_local_transcriber_discards_a_re_decode_that_loops():
    import dataclasses
    import numpy as np
    from ems_handoff.config import load_settings
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE

    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    span_reply = {"text": "ETA ten minutes. " * 25}       # what large-v3-turbo gave for SY074's collapsed span

    class FakeModel:
        def transcribe(self, audio, **kw):
            if len(audio) > 50 * SAMPLE_RATE:
                return iter([Seg(0.0, 10.0, "vitals are"), Seg(40.0, 45.0, "thank you")]), None
            return iter([Seg(0.0, 29.0, span_reply["text"])]), None

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(60 * SAMPLE_RATE)).astype(np.float32)
    audio[45 * SAMPLE_RATE:] *= 0.01
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    t._model = FakeModel()
    assert t.transcribe(audio) == "vitals are thank you"                 # the loop is not spliced in
    assert len(t.last_notes) == 1 and "discarded" in t.last_notes[0] and "repeated sentence" in t.last_notes[0]
    # an impossible word rate is refused too, loop or not
    span_reply["text"] = " ".join(f"word{i}" for i in range(200))      # 200 words for 30 s
    t.last_notes = []
    assert t.transcribe(audio) == "vitals are thank you" and "200 words in 30 s" in t.last_notes[0]
    # a plausible re-decode still goes in
    span_reply["text"] = "pressure 80 over 55 sugar 285 sepsis alert"
    t.last_notes = []
    assert t.transcribe(audio) == "vitals are pressure 80 over 55 sugar 285 sepsis alert thank you"


def test_local_transcriber_redecodes_a_sparse_or_repeating_long_segment():
    import dataclasses
    import numpy as np
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE

    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    first = {"text": "We'll see you next time."}                 # 5 words for 30 s: a collapsed window
    span = {"text": "GCS is 9. Two large-bore IVs. ETA 8 minutes."}

    class FakeModel:
        def transcribe(self, audio, **kw):
            if len(audio) > 50 * SAMPLE_RATE:
                return iter([Seg(0.0, 30.0, " ".join(f"w{i}" for i in range(60))), Seg(30.0, 60.0, first["text"])]), None
            return iter([Seg(0.0, 29.0, span["text"])]), None

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(60 * SAMPLE_RATE)).astype(np.float32)
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    t._model = FakeModel()
    assert t.transcribe(audio).endswith("w59 GCS is 9. Two large-bore IVs. ETA 8 minutes.")
    assert "30.0-60.0" in t.last_notes[0] and "9 words where the first pass had 5" in t.last_notes[0]
    # a repeating window (120 words for 30 s) is decoded again and replaced only by something shorter
    first["text"] = " ".join("13." for _ in range(120))
    t.last_notes = []
    assert t.transcribe(audio).endswith("w59 GCS is 9. Two large-bore IVs. ETA 8 minutes.")
    span["text"] = " ".join("12." for _ in range(130))
    t.last_notes = []
    assert t.transcribe(audio).endswith("w59 " + first["text"]) and "discarded" in t.last_notes[0]
    # a normal window (60 words for 30 s) is left alone
    first["text"] = " ".join(f"v{i}" for i in range(60))
    t.last_notes = []
    assert t.transcribe(audio).endswith("w59 " + first["text"]) and not t.last_notes


def test_local_transcriber_trims_the_lead_in_seam_of_a_re_decoded_span():
    import dataclasses
    import numpy as np
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE, _trim_seam

    assert _trim_seam("temp of 96.1 and", "96.1, and blood glucose 231", 8) == "blood glucose 231"
    assert _trim_seam("vitals are", "pressure 80 over 55", 8) == "pressure 80 over 55"
    assert _trim_seam("satting at 94% on 4 liters nasal cannula temp of", "24% on 4 liters, nasal cannula, temp of 96.1, and blood glucose 231", 8) == "96.1, and blood glucose 231"
    assert _trim_seam("rate of 127", "the patient is on 4 liters", 8) == "the patient is on 4 liters"   # no run, nothing dropped

    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    class FakeModel:
        def transcribe(self, audio, **kw):
            if len(audio) > 50 * SAMPLE_RATE:
                return iter([Seg(0.0, 10.0, "satting at 94 percent, temp of"), Seg(40.0, 45.0, "any questions")]), None
            # the span 7-40 s decoded alone: its first 3 s re-say the previous tail, then the missing middle
            return iter([Seg(0.0, 2.5, "Temp of"), Seg(2.5, 30.0, "temp of 96.1 and blood glucose 231")]), None

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(60 * SAMPLE_RATE)).astype(np.float32)
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    t._model = FakeModel()
    assert t.transcribe(audio) == "satting at 94 percent, temp of 96.1 and blood glucose 231 any questions"


def test_local_transcriber_redecodes_speech_after_the_last_worded_segment():
    import dataclasses
    import numpy as np
    from ems_handoff.transcribe import LocalTranscriber, SAMPLE_RATE

    class Seg:
        def __init__(self, start, end, text):
            self.start, self.end, self.text = start, end, text

    calls = []

    class FakeModel:
        def transcribe(self, audio, **kw):
            calls.append(round(len(audio) / SAMPLE_RATE, 1))
            if len(audio) > 50 * SAMPLE_RATE:      # the first pass stops at 36 s of a 60 s clip
                return iter([Seg(0.0, 30.0, " ".join(f"w{i}" for i in range(60))), Seg(30.0, 36.0, "Paramedic 700 to clear.")]), None
            return iter([Seg(0.0, 26.0, "GCS 12. Two large-bore IVs. ETA seven minutes. Paramedic 700 to clear.")]), None

    rng = np.random.default_rng(0)
    audio = (0.1 * rng.standard_normal(60 * SAMPLE_RATE)).astype(np.float32)
    t = LocalTranscriber(dataclasses.replace(load_settings(), local_whisper_redecode=True), model_size="tiny.en")
    t._model = FakeModel()
    text = t.transcribe(audio)
    assert text.endswith("w59 Paramedic 700 to clear. GCS 12. Two large-bore IVs. ETA seven minutes. Paramedic 700 to clear.")
    assert calls == [60.0, 27.0] and "36.0-60.0" in t.last_notes[0]
    # a quiet tail (the clip's trailing silence) is left alone
    audio[36 * SAMPLE_RATE:] *= 0.01
    calls.clear(); t.last_notes = []
    t.transcribe(audio)
    assert calls == [60.0] and not t.last_notes
