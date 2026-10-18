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
    assert s.groq_api_key == "gsk_x"
    assert s.mic_device == "PnP"
    assert s.port == 5000
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
