"""LLM client: JSON cleanup, request options per provider, retries, and errors."""

import json
from types import SimpleNamespace

import openai
import pytest

from ems_handoff.config import Settings
from ems_handoff.llm import LLMClient, LLMError, strip_to_json


def _reply(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _exc(cls, status=None, message="boom"):
    e = cls.__new__(cls)
    Exception.__init__(e, message)
    if status is not None:
        e.status_code = status
    return e


class FakeOpenAI:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return _reply(out)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr("ems_handoff.llm.time.sleep", lambda s: None)


@pytest.mark.parametrize("raw,expected", [
    ('{"a": 1}', '{"a": 1}'),
    ('```json\n{"a": 1}\n```', '{"a": 1}'),
    ('<think>long reasoning {not json}</think>\n{"a": 1}', '{"a": 1}'),
    ('Sure! Here is the JSON: {"a": {"b": 2}} Hope this helps.', '{"a": {"b": 2}}'),
    ('[{"x": 1}, {"x": 2}]', '[{"x": 1}, {"x": 2}]'),
    ("", ""),
    (None, ""),
])
def test_strip_to_json(raw, expected):
    assert strip_to_json(raw) == expected


def test_groq_request_uses_json_mode_and_configured_model():
    fake = FakeOpenAI(['{"ok": true}'])
    client = LLMClient(Settings(provider="groq", groq_api_key="k"), client=fake)
    assert json.loads(client.complete_json("prompt")) == {"ok": True}
    call = fake.calls[0]
    assert call["model"] == "llama-3.3-70b-versatile"
    assert call["response_format"] == {"type": "json_object"}
    assert "extra_body" not in call


def test_ollama_request_turns_thinking_off():
    fake = FakeOpenAI(['{"ok": true}'])
    client = LLMClient(Settings(provider="ollama", ollama_model="qwen3:4b"), client=fake)
    client.complete_json("prompt")
    assert fake.calls[0]["model"] == "qwen3:4b"
    assert fake.calls[0]["extra_body"] == {"think": False}


def test_openrouter_has_no_json_mode():
    fake = FakeOpenAI(['{"ok": true}'])
    LLMClient(Settings(provider="openrouter", openrouter_api_key="k"), client=fake).complete_json("p")
    assert "response_format" not in fake.calls[0]


def test_empty_reply_is_retried_then_succeeds():
    fake = FakeOpenAI(["", "<think>only thinking</think>", '{"ok": 1}'])
    assert LLMClient(Settings(), client=fake).complete_json("p") == '{"ok": 1}'
    assert len(fake.calls) == 3


def test_connection_errors_exhaust_attempts_with_a_clear_message():
    fake = FakeOpenAI([_exc(openai.APIConnectionError)] * 3)
    with pytest.raises(LLMError, match="after 3 tries"):
        LLMClient(Settings(provider="ollama"), client=fake).complete_json("p")


def test_bad_key_is_not_retried():
    fake = FakeOpenAI([_exc(openai.APIStatusError, status=401)])
    with pytest.raises(LLMError, match="rejected the API key"):
        LLMClient(Settings(groq_api_key="bad"), client=fake).complete_json("p")
    assert len(fake.calls) == 1


def test_missing_key_is_reported_before_any_call():
    with pytest.raises(LLMError, match="No API key"):
        LLMClient(Settings(provider="groq", groq_api_key=""))


# -- retired models ------------------------------------------------------------
from ems_handoff.llm import _RESOLVED, pick_model


def _models(*ids):
    return SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=i) for i in ids]))


@pytest.fixture(autouse=True)
def clear_model_cache():
    _RESOLVED.clear()
    yield
    _RESOLVED.clear()


def test_pick_model_keeps_configured_when_offered():
    assert pick_model({"llama-3.3-70b-versatile", "x"}, "llama-3.3-70b-versatile") == ("llama-3.3-70b-versatile", "")


def test_pick_model_uses_preference_list_then_heuristic():
    model, note = pick_model({"openai/gpt-oss-120b", "llama-3.1-8b-instant", "whisper-large-v3"}, "llama-3.3-70b-versatile")
    assert model == "openai/gpt-oss-120b" and "no longer offered" in note
    model, _ = pick_model({"whisper-large-v3", "newco/mega-70b-chat", "tiny-1b"}, "gone")
    assert model == "newco/mega-70b-chat"
    assert pick_model({"whisper-large-v3"}, "gone")[0] is None


def test_retired_groq_model_is_replaced_automatically():
    fake = FakeOpenAI(['{"ok": 1}'])
    fake.models = _models("openai/gpt-oss-120b", "whisper-large-v3-turbo")
    client = LLMClient(Settings(groq_api_key="k"), client=fake)
    client.complete_json("p")
    assert fake.calls[0]["model"] == "openai/gpt-oss-120b"
    ok, msg = LLMClient(Settings(groq_api_key="k"), client=fake).check()
    assert ok and "using 'openai/gpt-oss-120b'" in msg


def test_model_without_json_mode_is_retried_without_it():
    err = _exc(openai.APIStatusError, status=400, message="response_format json_object is not supported by this model")
    fake = FakeOpenAI([err, '{"ok": 1}'])
    assert LLMClient(Settings(groq_api_key="k"), client=fake).complete_json("p") == '{"ok": 1}'
    assert "response_format" in fake.calls[0] and "response_format" not in fake.calls[1]


def test_gpt_oss_gets_low_reasoning_effort_and_room_to_answer():
    fake = FakeOpenAI(['{"ok": 1}'])
    fake.models = _models("openai/gpt-oss-120b")
    LLMClient(Settings(groq_api_key="k"), client=fake).complete_json("p")
    call = fake.calls[0]
    assert call["model"] == "openai/gpt-oss-120b"
    assert call["reasoning_effort"] == "low" and call["max_tokens"] >= 8192


def test_rejected_reasoning_effort_is_dropped_and_retried():
    err = _exc(openai.APIStatusError, status=400, message="unknown parameter: reasoning_effort")
    fake = FakeOpenAI([err, '{"ok": 1}'])
    fake.models = _models("openai/gpt-oss-120b")
    assert LLMClient(Settings(groq_api_key="k"), client=fake).complete_json("p") == '{"ok": 1}'
    assert "reasoning_effort" not in fake.calls[1]


# -- Ollama memory ---------------------------------------------------------------
def test_ollama_killed_by_oom_gives_a_memory_hint(monkeypatch):
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 1.9)
    err = _exc(openai.APIStatusError, status=500,
               message="llama-server process has terminated: signal: killed")
    fake = FakeOpenAI([err])
    with pytest.raises(LLMError, match=r"ran out of memory loading 'qwen3:8b' \(1.9 GB free\).*qwen3:4b"):
        LLMClient(Settings(provider="ollama"), client=fake).complete_json("p")
    assert len(fake.calls) == 1


def test_ollama_check_warns_when_the_model_will_not_fit(monkeypatch):
    import io
    tags = json.dumps({"models": [{"name": "qwen3:8b", "size": 5_200_000_000}]}).encode()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("ems_handoff.llm.urllib.request.urlopen", lambda url, timeout: Resp(tags))
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 2.0)
    ok, msg = LLMClient(Settings(provider="ollama"), client=FakeOpenAI([])).check()
    assert ok and "needs ~6.8 GB, 2.0 GB free" in msg and "killed for lack of memory" in msg
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 7.5)
    ok, msg = LLMClient(Settings(provider="ollama"), client=FakeOpenAI([])).check()
    assert ok and "killed" not in msg
