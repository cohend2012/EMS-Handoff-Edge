"""
fakes.py — stand-ins for the mic, recognizers, transcriber and LLM.

They let every layer run in tests (and in CI) with no microphone, no models and
no network, while exercising the real threading, state machine and API code.
"""

from __future__ import annotations

import json
import threading
import time
from typing import List, Optional

import numpy as np

from ems_handoff.audio.capture import AudioCapture
from ems_handoff.audio.listeners import CaptionUpdate
from ems_handoff.transcribe import TranscriptionError

RATE = 16000


def tone(seconds: float, amp: float = 0.05, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def report_dict(**overrides) -> dict:
    data = {
        "alert_type": "STEMI", "alert_level": "RED", "eta_minutes": 8,
        "patient": {"age": 67, "age_unit": "years", "sex": "Male"},
        "chief_complaint": "Chest pain radiating to left arm",
        "mechanism_or_onset": "45 minutes ago at rest",
        "vital_signs": {"blood_pressure": "158/92", "heart_rate": "98", "respiratory_rate": "22",
                        "spo2": "98% on 4L NC", "temperature": None, "blood_glucose": None, "gcs": None},
        "assessment_findings": {"level_of_consciousness": "A&Ox4", "airway": "Patent",
                                "breathing": "Mild distress", "circulation": "Diaphoretic",
                                "neuro": "Intact", "other_findings": ["ST elevation V1-V4"]},
        "ems_interventions": [{"intervention": "Aspirin", "details": "324 mg PO", "response": ""}],
        "history": {"past_medical": ["HTN", "T2DM"], "medications": ["metformin", "lisinopril"],
                    "allergies": ["NKDA"], "code_status": "Full code"},
        "hospital_recommendations": {"activation_requested": "Cath lab", "suggested_destination": "Cath Lab",
                                     "resources_to_prepare": ["Cath team"],
                                     "time_critical_actions": ["Activate cath lab"]},
        "handoff_summary": {"one_liner": "67M anterior STEMI, ETA 8", "key_concerns": ["Anterior STEMI"],
                            "pending_needs": ["Cath lab"]},
        "data_quality": {"confidence": "HIGH", "missing_info": [], "assumptions": []},
    }
    data.update(overrides)
    return data


class FakeLLM:
    """Returns queued replies (str) or raises queued exceptions, in order."""

    def __init__(self, replies: Optional[list] = None, provider: str = "groq", model: str = "fake-model",
                 delay: float = 0.0):
        self.replies = list(replies) if replies is not None else [json.dumps(report_dict())]
        self.provider = provider
        self.model = model
        self.prompts: List[str] = []
        self.delay = delay
        self.on_wait = None            # the pipeline wires this to its stage callback, like LLMClient
        self.last_recovery: dict = {}
        self.restart_notice: Optional[str] = None   # set to report a fake "model server restarting" wait

    def complete_json(self, prompt: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        self.prompts.append(prompt)
        if self.delay:
            time.sleep(self.delay)
        if self.restart_notice:
            self.last_recovery = {"model_server": "restarted", "reason": "cannot reach Ollama", "waited_s": 5.0, "retries": 2}
            if self.on_wait:
                self.on_wait(self.restart_notice)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply

    def check(self):
        return True, f"{self.provider} ready ({self.model})"

    # what the Settings screen asks an Ollama client (sizes in GB on disk; free memory)
    ollama_models = {"qwen3:8b": 5.23, "qwen3:4b-instruct": 2.5, "llama3.2:3b": 2.02}
    free_gb = 6.0

    def ollama_inventory(self):
        from ems_handoff.llm import ollama_need_gb
        models = [{"name": n, "size_gb": s, "need_gb": ollama_need_gb(s), "fits": ollama_need_gb(s) <= self.free_gb}
                  for n, s in self.ollama_models.items()]
        return {"models": models, "free_gb": self.free_gb}

    def ollama_model_status(self, name):
        from ems_handoff.llm import LLMClient
        return LLMClient.ollama_model_status(self, name)

    def _connection_hint(self):
        return "is Ollama running?"


class FakeTranscriber:
    name = "groq"

    def __init__(self, text: str = "Medic 21 inbound with a 45 year old male found down, GCS 14.",
                 fail: bool = False, name: str = "groq"):
        self.text = text
        self.fail = fail
        self.name = name
        self.calls = 0
        self.last_audio_len = 0

    def describe(self) -> str:
        return f"fake {self.name}"

    def transcribe(self, audio: np.ndarray) -> str:
        self.calls += 1
        self.last_audio_len = len(audio)
        if self.fail:
            raise TranscriptionError(f"{self.name} transcriber is down")
        return self.text


class FakeStream:
    def __init__(self):
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        self.closed = True


def fake_capture(**kw) -> AudioCapture:
    return AudioCapture("", stream_factory=lambda dev, rate, block, cb: FakeStream(),
                        device_info=(7, "Fake USB mic", 16000.0), **kw)


class AutoFeedCapture(AudioCapture):
    """A fake mic that produces 100 ms blocks in real time until stopped."""

    def __init__(self, block_s: float = 0.1, amp: float = 0.05):
        super().__init__("", stream_factory=lambda dev, rate, block, cb: FakeStream(),
                         device_info=(7, "Fake USB mic", 16000.0))
        self._block_s = block_s
        self._amp = amp
        self._run_feed = threading.Event()

    def start(self):
        super().start()
        self._run_feed.set()
        threading.Thread(target=self._feeder, daemon=True, name="fake-mic").start()

    def _feeder(self):
        while self._run_feed.is_set():
            self.feed(tone(self._block_s, self._amp))
            time.sleep(self._block_s)

    def stop(self):
        self._run_feed.clear()
        return super().stop()


class ScriptedCaptioner:
    """Reveals one word of ``script`` every ``blocks_per_word`` blocks, as live captions do."""

    name = "scripted"

    def __init__(self, script: str, blocks_per_word: int = 2, phrase: str = ""):
        self.words = script.split()
        self.blocks_per_word = blocks_per_word
        self.phrase = phrase
        self.blocks = 0
        self._last_n = 0

    def feed(self, block) -> CaptionUpdate:
        self.blocks += 1
        n = min(len(self.words), self.blocks // self.blocks_per_word)
        final = " ".join(self.words[:max(0, n - 1)])
        partial = self.words[n - 1] if n else ""
        changed = n != self._last_n
        self._last_n = n
        heard = bool(self.phrase) and self.phrase in " ".join(self.words[:n]).lower()
        return CaptionUpdate(final, partial, changed, heard)


class TriggerSpotter:
    """Fires once after ``after_blocks`` blocks (or never if None)."""

    def __init__(self, after_blocks: Optional[int] = None):
        self.after = after_blocks
        self.blocks = 0

    def feed(self, block) -> bool:
        self.blocks += 1
        return self.after is not None and self.blocks == self.after
