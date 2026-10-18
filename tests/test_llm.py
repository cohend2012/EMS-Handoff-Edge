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
