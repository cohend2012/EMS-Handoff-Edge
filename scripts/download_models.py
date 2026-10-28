#!/usr/bin/env python3
"""
download_models.py — fetch the speech models once, so the app can run offline.

    python scripts/download_models.py            Vosk caption model + Whisper models from .env
    python scripts/download_models.py --vosk     only the Vosk caption model
    python scripts/download_models.py --whisper  only the Whisper models

Downloads
    Vosk small English model (~40 MB) -> models/vosk-model-small-en-us-0.15/
        from https://alphacephei.com/vosk/models
    faster-whisper models -> the Hugging Face cache (~/.cache/huggingface/hub)
        LOCAL_WHISPER_MODEL (default medium, about 1.5 GB): offline final transcript
        CAPTION_WHISPER_MODEL (default tiny.en, about 75 MB): caption fallback

Safe to re-run: anything already present is skipped.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VOSK_URL = "https://alphacephei.com/vosk/models/{name}.zip"


def download_vosk(model_dir: Path) -> None:
    if (model_dir / "conf").exists() or (model_dir / "am").exists():
        print(f"[=] Vosk model already present: {model_dir}")
        return
    url = VOSK_URL.format(name=model_dir.name)
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading {url}")
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp, "model.zip")
        with urllib.request.urlopen(url, timeout=60) as r, open(zpath, "wb") as f:
            shutil.copyfileobj(r, f)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(model_dir.parent)
    print(f"[OK] Vosk model ready: {model_dir}")


def download_whisper(size: str) -> None:
    from faster_whisper import download_model
    print(f"[*] Fetching faster-whisper '{size}' (skipped if cached)")
    path = download_model(size)
    print(f"[OK] Whisper {size}: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--vosk", action="store_true")
    ap.add_argument("--whisper", action="store_true")
    args = ap.parse_args()
    both = not (args.vosk or args.whisper)

    from ems_handoff.config import load_env_files
    load_env_files()
    from ems_handoff.config import load_settings
    s = load_settings()

    failed = False
    if args.vosk or both:
        try:
            download_vosk(Path(s.vosk_model_dir))
        except Exception as e:  # noqa: BLE001
            failed = True
            print(f"[X] Vosk download failed: {e}")
    if args.whisper or both:
        for size in dict.fromkeys([s.local_whisper_model, s.caption_whisper_model]):
            try:
                download_whisper(size)
            except Exception as e:  # noqa: BLE001
                failed = True
                print(f"[X] Whisper {size} download failed: {e}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
