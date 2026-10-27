#!/usr/bin/env python3
"""
clone_voice_cloud.py — read the practice scripts in a cloned voice on ElevenLabs, Cartesia or Fish Audio.

    python scripts/corpus/clone_voice_cloud.py elevenlabs --ref ~/voice_refs/me.wav --ref-text ~/voice_refs/me.txt
    python scripts/corpus/clone_voice_cloud.py cartesia   --ref ~/voice_refs/me.wav
    python scripts/corpus/clone_voice_cloud.py fish       --ref ~/voice_refs/me.wav --ref-text ~/voice_refs/me.txt
    python scripts/corpus/clone_voice_cloud.py elevenlabs --voice-id <id>        # a stock or designed voice
    python scripts/corpus/clone_voice_cloud.py all --ref ... --scripts stemi     # every service with a key

Keys (in ~/.config/handoff-ai/secrets.env or the environment): ELEVENLABS_API_KEY, CARTESIA_API_KEY,
FISH_AUDIO_API_KEY. A service without a key is skipped with a note. `--check` only says which keys exist.

The reference clip must be a speaker who agreed to be cloned: your own 10-30 s recording of one of the
scripts, read close to the mic, is the right one (Fish wants its exact transcript in --ref-text; ElevenLabs
and Cartesia do not). The paramedics in the radio corpus did not agree, so their voices are not cloned
here; the corpus gives the channel (scripts/corpus/radioize.py) and the noise (~/noise_bank), not a person.

Outputs ~/air_scripts/<script>_<service>.wav, 16 kHz mono, in the spoken form of the scripts
(scripts/air_check.py `spoken()`), ready for `air_check.py <script> --play --variant _<service>` or for
radioize.py first. Each service's clone is kept (voice id printed) so later runs can pass --voice-id.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import mimetypes
import os
import struct
import subprocess
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from ems_handoff.config import load_env_files  # noqa: E402

SERVICES = ("elevenlabs", "cartesia", "fish")
KEYS = {"elevenlabs": "ELEVENLABS_API_KEY", "cartesia": "CARTESIA_API_KEY", "fish": "FISH_AUDIO_API_KEY"}
OUT = Path("~/air_scripts").expanduser()


def load_air_check():
    spec = importlib.util.spec_from_file_location("air_check", ROOT / "scripts" / "air_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- small HTTP helpers (no extra packages) -------------------------------------------------------

def http(method: str, url: str, headers: Dict[str, str], body: bytes) -> bytes:
    req = urllib.request.Request(url, method=method, data=body, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {url}: HTTP {e.code}: {e.read().decode(errors='replace')[:400]}") from None


def multipart(fields: Dict[str, str], files: Dict[str, Path]) -> tuple:
    boundary = "----handoff" + uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode()
    for k, p in files.items():
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        out += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"; filename=\"{p.name}\"\r\n"
                f"Content-Type: {ctype}\r\n\r\n").encode() + p.read_bytes() + b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), f"multipart/form-data; boundary={boundary}"


def msgpack(obj: Any) -> bytes:
    """Enough of MessagePack for the Fish Audio request body: nil, bool, int, float, str, bytes, list, dict."""
    if obj is None:
        return b"\xc0"
    if obj is True:
        return b"\xc3"
    if obj is False:
        return b"\xc2"
    if isinstance(obj, int):
        if 0 <= obj < 128:
            return bytes([obj])
        return b"\xd3" + struct.pack(">q", obj)
    if isinstance(obj, float):
        return b"\xcb" + struct.pack(">d", obj)
    if isinstance(obj, str):
        b = obj.encode()
        if len(b) < 32:
            return bytes([0xA0 | len(b)]) + b
        return b"\xdb" + struct.pack(">I", len(b)) + b
    if isinstance(obj, (bytes, bytearray)):
        return b"\xc6" + struct.pack(">I", len(obj)) + bytes(obj)
    if isinstance(obj, (list, tuple)):
        head = bytes([0x90 | len(obj)]) if len(obj) < 16 else b"\xdd" + struct.pack(">I", len(obj))
        return head + b"".join(msgpack(x) for x in obj)
    if isinstance(obj, dict):
        head = bytes([0x80 | len(obj)]) if len(obj) < 16 else b"\xdf" + struct.pack(">I", len(obj))
        return head + b"".join(msgpack(k) + msgpack(v) for k, v in obj.items())
    raise TypeError(type(obj))


def to_wav16k(audio: bytes, dst: Path) -> None:
    """Whatever container came back (mp3, wav at any rate, raw pcm16 at 16 kHz), as 16 kHz mono WAV."""
    args = ["ffmpeg", "-v", "error", "-y"]
    if audio[:4] not in (b"RIFF", b"ID3\x03", b"ID3\x04", b"fLaC") and not audio[:2] == b"\xff\xfb":
        args += ["-f", "s16le", "-ar", "16000", "-ac", "1"]
    subprocess.run(args + ["-i", "pipe:0", "-ac", "1", "-ar", "16000", str(dst)], input=audio, check=True)


# ---- the services ----------------------------------------------------------------------------------

class ElevenLabs:
    base = "https://api.elevenlabs.io/v1"

    def __init__(self, key: str):
        self.h = {"xi-api-key": key}

    def clone(self, ref: Path, name: str) -> str:
        body, ctype = multipart({"name": name, "remove_background_noise": "false"}, {"files": ref})
        r = json.loads(http("POST", f"{self.base}/voices/add", {**self.h, "Content-Type": ctype}, body))
        return r["voice_id"]

    def speak(self, voice_id: str, text: str) -> bytes:
        body = json.dumps({"text": text, "model_id": "eleven_multilingual_v2",
                           "voice_settings": {"stability": 0.5, "similarity_boost": 0.8}}).encode()
        return http("POST", f"{self.base}/text-to-speech/{voice_id}?output_format=pcm_16000",
                    {**self.h, "Content-Type": "application/json"}, body)


class Cartesia:
    base = "https://api.cartesia.ai"

    def __init__(self, key: str):
        self.h = {"Authorization": f"Bearer {key}", "Cartesia-Version": "2026-08-14"}

    def clone(self, ref: Path, name: str) -> str:
        body, ctype = multipart({"name": name, "language": "en", "description": "HANDOFF.AI test voice"}, {"clip": ref})
        r = json.loads(http("POST", f"{self.base}/voices/clone", {**self.h, "Content-Type": ctype}, body))
        return r["id"]

    def speak(self, voice_id: str, text: str) -> bytes:
        body = json.dumps({"model_id": "sonic-3.6", "transcript": text, "voice": {"id": voice_id}, "language": "en",
                           "output_format": {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 16000}}).encode()
        return http("POST", f"{self.base}/tts/bytes", {**self.h, "Content-Type": "application/json"}, body)


class Fish:
    base = "https://api.fish.audio/v1"

    def __init__(self, key: str):
        self.h = {"Authorization": f"Bearer {key}"}
        self.ref: Optional[Dict[str, Any]] = None

    def clone(self, ref: Path, name: str, ref_text: str = "") -> str:
        # zero-shot: the clip travels with every request; nothing is stored on the service
        self.ref = {"audio": ref.read_bytes(), "text": ref_text}
        return "zero-shot"

    def speak(self, voice_id: str, text: str) -> bytes:
        body: Dict[str, Any] = {"text": text, "format": "wav", "sample_rate": 16000, "normalize": True}
        if voice_id != "zero-shot":
            body["reference_id"] = voice_id
        elif self.ref:
            body["references"] = [self.ref]
        return http("POST", f"{self.base}/tts", {**self.h, "Content-Type": "application/msgpack", "model": "s2.1-pro"},
                    msgpack(body))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("service", choices=SERVICES + ("all",))
    ap.add_argument("--ref", type=Path, help="10-30 s clip of a consenting speaker to clone")
    ap.add_argument("--ref-text", type=Path, help="exact transcript of the clip (Fish needs it)")
    ap.add_argument("--voice-id", help="use an existing voice on the service instead of cloning")
    ap.add_argument("--scripts", nargs="+", default=["stemi", "trauma", "sepsis"])
    ap.add_argument("--name", default="handoff-test-voice")
    ap.add_argument("--check", action="store_true", help="only report which keys are present")
    args = ap.parse_args()

    load_env_files()
    services = SERVICES if args.service == "all" else (args.service,)
    have = {s: bool(os.environ.get(KEYS[s])) for s in services}
    for s in services:
        print(f"{s:<11} key {KEYS[s]}: {'present' if have[s] else 'missing'}")
    if args.check:
        return 0
    if not args.voice_id and not args.ref:
        ap.error("give --ref <clip> to clone, or --voice-id")
    air = load_air_check()
    OUT.mkdir(parents=True, exist_ok=True)
    ref_text = args.ref_text.read_text().strip() if args.ref_text else ""
    failures = 0
    for s in services:
        if not have[s]:
            continue
        svc = {"elevenlabs": ElevenLabs, "cartesia": Cartesia, "fish": Fish}[s](os.environ[KEYS[s]])
        try:
            if args.voice_id:
                vid = args.voice_id
            elif s == "fish":
                vid = svc.clone(args.ref, args.name, ref_text)
            else:
                vid = svc.clone(args.ref, args.name)
            print(f"[{s}] voice {vid}")
            for name in args.scripts:
                audio = svc.speak(vid, air.spoken(air.script_text(name)))
                dst = OUT / f"{name}_{s}.wav"
                to_wav16k(audio, dst)
                print(f"[{s}] {dst} ({dst.stat().st_size // 32000} s)")
        except (RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as e:
            failures += 1
            print(f"[{s}] FAILED: {e}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
