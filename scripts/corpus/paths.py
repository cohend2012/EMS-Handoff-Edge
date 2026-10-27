"""
paths.py — where the corpus scripts find their data and tools. One place, settings with defaults.

    HANDOFF_DATASETS   generated and derived data (default ~/datasets); see docs/DATA_LAYOUT.md
    HANDOFF_CALLSETS   the real call-sets (default ~/callsets); the same setting the app's replay mode uses
    HANDOFF_LABELS     labels, references and gold for the real corpus (default ~/labels)
    HANDOFF_PIPER      the Piper executable (default ~/piper-venv/bin/piper)
    HANDOFF_AIR_SINK   PipeWire sink the air test plays into (default: the JMTek USB dongle's node name)

Everything is read at call time so a script can be pointed elsewhere for one run.
"""

from __future__ import annotations

import os
from pathlib import Path


def _path(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or default).expanduser()


def datasets() -> Path:
    return _path("HANDOFF_DATASETS", "~/datasets")


def callsets() -> Path:
    return _path("HANDOFF_CALLSETS", "~/callsets")


def labels() -> Path:
    return _path("HANDOFF_LABELS", "~/labels")


def piper() -> Path:
    return _path("HANDOFF_PIPER", "~/piper-venv/bin/piper")


def air_sink() -> str:
    return os.environ.get("HANDOFF_AIR_SINK") or "alsa_output.usb-0c76_USB_PnP_Audio_Device-00.analog-stereo"


# the sub-folders of the datasets tree (docs/DATA_LAYOUT.md)
def voices_refs() -> Path: return datasets() / "voices" / "refs"
def piper_voice(name: str = "en_US-lessac-medium") -> Path: return datasets() / "voices" / "piper" / f"{name}.onnx"
def noise() -> Path: return datasets() / "noise"
def spectral() -> Path: return datasets() / "spectral"
def scripts_v0() -> Path: return datasets() / "scripts_v0"
def air_runs() -> Path: return datasets() / "air_runs"
