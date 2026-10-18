"""Shared pytest setup: put the project on sys.path and keep tests out of the real output/ folder."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture
def settings(tmp_path):
    from ems_handoff.config import Settings
    return Settings(provider="groq", groq_api_key="test-key", output_dir=tmp_path / "output",
                    textbook_dir=tmp_path / "no-textbooks", max_recording_s=30)


@pytest.fixture(autouse=True)
def isolated_user_settings(tmp_path, monkeypatch):
    import ems_handoff.server as server
    monkeypatch.setattr(server, "USER_SETTINGS_FILE", tmp_path / "user_settings.json")


def wait_for(predicate, timeout=5.0, step=0.02):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(step)
    return False
