#!/usr/bin/env python3
"""Candidates for a reference transcript: every burst of a call-set through several transcribers.

    draft_reference.py CS008 CS014 CS001 [--out ~/labels/reference] [--no-local]

For each burst WAV the script saves <out>/<callset>/<segment>.candidates.json with one text per
transcriber:
    groq_turbo        Groq whisper-large-v3-turbo with the app's current vocabulary hint
    groq_turbo_radio  the same model with the radio hint below (unit names, hospital, priority
                      words, "how do you copy", the drugs heard on these calls)
    groq_large        Groq whisper-large-v3 with the radio hint
    local_medium      faster-whisper medium on the CPU with the radio hint (slow; --no-local skips)
    vosk_small        the caption model (no hint)
A person (or Claude, reading the candidates side by side) writes <segment>.ref.txt from them,
marking every correction in <callset>/corrections.md. Nothing here is ground truth by itself.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ems_handoff.audio.sources import load_wav  # noqa: E402
from ems_handoff.config import load_env_files, load_settings  # noqa: E402
from ems_handoff.lexicon_rag import whisper_hint  # noqa: E402
from ems_handoff.transcribe import to_wav_bytes  # noqa: E402

RADIO_HINT = ("Paramedic radio report to Shady Grove. Medic 729, Medic 731, Ambulance 721. "
              "Priority 1, priority 2, priority 3. How do you copy? Copy. ETA ten minutes. "
              "62-year-old female, three-month-old female. Chest pain, substernal, radiates to the back. "
              "Aspirin 324, nitroglycerin 400 micrograms sublingual, 12-lead, normal sinus with PVCs. "
              "Respiratory distress, drooling, croup, blow-by oxygen, dexamethasone 2 mg, racemic epinephrine 2.5 mg, "
              "responsive to pain. Stroke alert, LAMS zero, Cincinnati negative, last known well, blood glucose 94. "
              "BP 188 over 111, heart rate 58, SpO2 95 percent on room air.")


def groq_client(settings):
    from openai import OpenAI
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key, timeout=120, max_retries=2)


def groq_text(client, audio: np.ndarray, model: str, prompt: str) -> str:
    r = client.audio.transcriptions.create(file=("burst.wav", to_wav_bytes(audio)), model=model, prompt=prompt[-800:],
                                           language="en", temperature=0.0, response_format="json")
    return (r.text or "").strip()


def local_text(audio: np.ndarray, size: str, prompt: str) -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, language="en", beam_size=5, initial_prompt=prompt, vad_filter=True, temperature=0.0)
    return " ".join(s.text.strip() for s in segments).strip()


def vosk_text(audio: np.ndarray, settings) -> str:
    import vosk
    from ems_handoff.audio.listeners import load_vosk_model
    vosk.SetLogLevel(-1)
    rec = vosk.KaldiRecognizer(load_vosk_model(settings.vosk_model_dir), 16000)
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    out = []
    for i in range(0, len(pcm), 8000):
        if rec.AcceptWaveform(pcm[i:i + 8000]):
            out.append(json.loads(rec.Result()).get("text", ""))
    out.append(json.loads(rec.FinalResult()).get("text", ""))
    return " ".join(x for x in out if x).strip()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("callsets", nargs="+")
    ap.add_argument("--callsets-dir", type=Path, default=Path.home() / "callsets")
    ap.add_argument("--out", type=Path, default=Path.home() / "labels" / "reference")
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--local-size", default="medium")
    args = ap.parse_args(argv)
    load_env_files()
    settings = load_settings()
    client = groq_client(settings) if settings.groq_api_key else None
    hint = whisper_hint()
    for cid in args.callsets:
        info = json.loads((args.callsets_dir / cid / "callset_info.json").read_text())
        out_dir = args.out / cid
        out_dir.mkdir(parents=True, exist_ok=True)
        for b in info["bursts"]:
            seg = b["segment_id"]
            wav = args.callsets_dir / cid / f"{seg}.wav"
            audio = load_wav(wav)
            path = out_dir / f"{seg}.candidates.json"
            cands = json.loads(path.read_text()) if path.exists() else {}
            cands["_audio_s"] = round(len(audio) / 16000, 2)
            cands["_source"] = b.get("source_file")
            cands["_stage3"] = (args.callsets_dir / cid / f"{seg}.txt").read_text().strip() if (args.callsets_dir / cid / f"{seg}.txt").exists() else ""
            jobs = []
            if client:
                jobs += [("groq_turbo", lambda: groq_text(client, audio, "whisper-large-v3-turbo", hint)),
                         ("groq_turbo_radio", lambda: groq_text(client, audio, "whisper-large-v3-turbo", RADIO_HINT)),
                         ("groq_large", lambda: groq_text(client, audio, "whisper-large-v3", RADIO_HINT))]
            jobs.append(("vosk_small", lambda: vosk_text(audio, settings)))
            if not args.no_local:
                jobs.append((f"local_{args.local_size}", lambda: local_text(audio, args.local_size, RADIO_HINT)))
            for name, fn in jobs:
                if name in cands:
                    continue
                t0 = time.time()
                try:
                    cands[name] = fn()
                except Exception as e:  # noqa: BLE001
                    cands[name] = f"<error: {e}>"
                print(f"{cid} {seg} {name}: {time.time() - t0:.1f}s  {cands[name][:80]}", flush=True)
                path.write_text(json.dumps(cands, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
