#!/usr/bin/env python3
"""
clone_voice.py — read the practice scripts in a paramedic's voice, cloned from a corpus clip (XTTS-v2).

    ~/xtts-venv/bin/python scripts/corpus/clone_voice.py $HANDOFF_DATASETS/voices/refs/medic_cs008_s025.wav --scripts stemi
    ~/xtts-venv/bin/python scripts/corpus/clone_voice.py $HANDOFF_DATASETS/voices/refs/medic_cs008_s025.wav --tag cs008

Inputs
    ref        a 6-15 s WAV of one speaker (cut from ~/callsets/<CS>/<segment>.wav; see $HANDOFF_DATASETS/voices/refs/)
    --scripts  stemi trauma sepsis (default all); the spoken form from scripts/air_check.py is used, so
               "IV" is read as "I V" and "BP 158/92" as "B P 158 over 92"
    --tag      suffix of the output, $HANDOFF_DATASETS/scripts_v0/<script>_xtts<tag>.wav (default _xtts)
    --device   cpu (default) or cuda

The first run downloads the XTTS-v2 model (about 1.9 GB) into ~/.local/share/tts. Runs in ~/xtts-venv
(coqui-tts), not the app's venv. CPU synthesis of a 75 s script takes minutes on the Jetson.
Output is 24 kHz mono; radioize.py can then put it through the radio channel.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

os.environ.setdefault("COQUI_TOS_AGREED", "1")


def load_air_check():
    spec = importlib.util.spec_from_file_location("air_check", ROOT / "scripts" / "air_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("ref", type=Path)
    ap.add_argument("--scripts", nargs="+", default=["stemi", "trauma", "sepsis"])
    ap.add_argument("--tag", default="_xtts")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=paths.scripts_v0())
    args = ap.parse_args()

    air = load_air_check()
    from TTS.api import TTS
    t0 = time.monotonic()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(args.device)
    print(f"[OK] XTTS-v2 loaded in {time.monotonic() - t0:.0f} s on {args.device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.scripts:
        text = air.spoken(air.script_text(name))
        dst = args.out / f"{name}{args.tag}.wav"
        t1 = time.monotonic()
        tts.tts_to_file(text=text, speaker_wav=str(args.ref), language="en", file_path=str(dst),
                        split_sentences=True)
        print(f"[OK] {dst} in {time.monotonic() - t1:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
