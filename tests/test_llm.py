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


def _tags(monkeypatch, **sizes_gb):
    import io
    tags = json.dumps({"models": [{"name": n, "size": int(s * 1e9)} for n, s in sizes_gb.items()]}).encode()

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("ems_handoff.llm.urllib.request.urlopen", lambda url, timeout: Resp(tags))


def test_pick_ollama_model_by_free_memory():
    from ems_handoff.llm import pick_ollama_model
    models = {"qwen3:8b": 5.2, "qwen3:4b": 2.5, "qwen3:1.7b": 1.4}
    assert pick_ollama_model(models, "qwen3:8b", 7.5) == ("qwen3:8b", "")
    assert pick_ollama_model(models, "qwen3:8b", None) == ("qwen3:8b", "")      # memory unknown: trust it
    model, note = pick_ollama_model(models, "qwen3:8b", 2.8)
    assert model == "qwen3:1.7b"
    assert note == "'qwen3:8b' needs ~6.8 GB but only 2.8 GB is free; using 'qwen3:1.7b'"
    assert pick_ollama_model(models, "qwen3:8b", 4.0)[0] == "qwen3:4b"
    model, note = pick_ollama_model({"qwen3:8b": 5.2}, "qwen3:8b", 2.8)
    assert model == "qwen3:8b" and "no smaller model is pulled" in note
    assert pick_ollama_model(models, "missing:1b", 7.5) == (None, "")


def test_ollama_report_uses_a_model_that_fits(monkeypatch):
    _tags(monkeypatch, **{"qwen3:8b": 5.2, "qwen3:4b": 2.5})
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 3.6)   # 4b needs ~3.3 GB
    fake = FakeOpenAI(['{"ok": 1}'])
    client = LLMClient(Settings(provider="ollama", ollama_model="qwen3:8b"), client=fake)
    client.complete_json("p")
    assert fake.calls[0]["model"] == "qwen3:4b"
    ok, msg = LLMClient(Settings(provider="ollama", ollama_model="qwen3:8b"), client=fake).check()
    assert ok and msg.startswith("ollama ready (qwen3:4b)") and "using 'qwen3:4b'" in msg


def test_ollama_check_warns_when_nothing_smaller_is_pulled(monkeypatch):
    _tags(monkeypatch, **{"qwen3:8b": 5.2})
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 2.0)
    ok, msg = LLMClient(Settings(provider="ollama"), client=FakeOpenAI([])).check()
    assert ok and "needs ~6.8 GB, 2.0 GB free" in msg and "ollama pull qwen3:4b" in msg
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 7.5)
    ok, msg = LLMClient(Settings(provider="ollama"), client=FakeOpenAI([])).check()
    assert ok and msg == "ollama ready (qwen3:8b); needs ~6.8 GB, 7.5 GB free"


def test_ollama_preload_loads_the_model_that_fits(monkeypatch):
    _tags(monkeypatch, **{"qwen3:8b": 5.2, "qwen3:4b": 2.5})
    monkeypatch.setattr("ems_handoff.llm.available_memory_gb", lambda: 3.6)
    posted = []
    real = __import__("ems_handoff.llm", fromlist=["urllib"]).urllib.request.urlopen

    def urlopen(req, timeout):
        if isinstance(req, str):
            return real(req, timeout=timeout)
        posted.append(json.loads(req.data))
        return real("tags", timeout=timeout)
    monkeypatch.setattr("ems_handoff.llm.urllib.request.urlopen", urlopen)
    note = LLMClient(Settings(provider="ollama", ollama_model="qwen3:8b"), client=FakeOpenAI([])).preload()
    assert posted == [{"model": "qwen3:4b", "keep_alive": "30m"}] and "using 'qwen3:4b'" in note
    assert LLMClient(Settings(groq_api_key="k"), client=FakeOpenAI([])).preload() == ""


def test_usage_is_logged_when_the_provider_reports_it(caplog):
    import logging
    reply = _reply('{"ok": 1}')
    reply.usage = SimpleNamespace(prompt_tokens=2100, completion_tokens=850)
    fake = FakeOpenAI([])
    fake.chat.completions.create = lambda **kw: reply     # a reply object with usage
    with caplog.at_level(logging.INFO, logger="ems_handoff.llm"):
        LLMClient(Settings(provider="ollama"), client=fake).complete_json("p")
    assert "2100 prompt + 850 output tokens" in caplog.text and "tok/s" in caplog.text
