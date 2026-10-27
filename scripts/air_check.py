#!/usr/bin/env python3
"""
air_check.py — put a practice script through the real speaker-and-mic path and score what came back.

    python scripts/air_check.py stemi --play                 # Piper reads the script on the USB speaker
    python scripts/air_check.py all --play                   # stemi, trauma, sepsis in turn
    python scripts/air_check.py trauma --read --label mic2   # you read it aloud; press Enter when done
    python scripts/air_check.py all --play --variant _radio --label usb-dongle-radioeq
                                                             # the radio-filtered WAVs (radioize.py)

What it does
    1. Starts an ordinary live session in the running app (POST /api/live/start), like pressing Start.
    2. Plays the script's WAV into the PipeWire sink (--play), or waits for you to read it (--read).
    3. Presses Done, waits for the report, and collects the live captions (SSE), the final transcript
       and the report.
    4. Scores the transcript and the captions against the script text: word error rate (same
       normalisation as scripts/corpus/score_callsets.py), the reference words that were dropped or
       replaced, and the alert type / level / ETA the report produced.
    5. Saves everything under $HANDOFF_DATASETS/air_runs/<script>_<label>_<stamp>/ and appends one line to
       $HANDOFF_DATASETS/air_runs/INDEX.tsv, so runs with different mics or speakers can be compared.

Inputs
    The script texts are ems_handoff/samples.py (SAMPLE_STEMI, SAMPLE_TRAUMA, SAMPLE_SEPSIS). The WAVs
    are made once with Piper (see --make-wavs) into $HANDOFF_DATASETS/scripts_v0/<name>.wav; --wav overrides.
    The app must be running (python -m ems_handoff) and idle; --server points at it.

Threads: one background thread reads the event stream for captions while the main thread drives the run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "corpus"))

import paths  # noqa: E402  (scripts/corpus: HANDOFF_DATASETS, HANDOFF_PIPER, HANDOFF_AIR_SINK)
from score_callsets import normalize, wer  # noqa: E402  (scripts/corpus)

SCRIPTS = ("stemi", "trauma", "sepsis")
WAV_DIR = paths.scripts_v0()
OUT_DIR = paths.air_runs()
PIPER = paths.piper()
PIPER_VOICE = paths.piper_voice()


def script_text(name: str) -> str:
    from ems_handoff import samples
    return getattr(samples, f"SAMPLE_{name.upper()}").strip()


# How a paramedic says the written abbreviations. Piper's phonemizer otherwise reads "IV" as the Roman
# numeral and "SpO2" as a word; the first air run heard "Roman 4" and "sbo2" for them.
SPOKEN = [("SpO2", "S P O 2"), ("IVs", "I V's"), ("IV", "I V"), ("BP", "B P"), ("ETA", "E T A"), ("GCS", "G C S"),
          ("MVC", "M V C"), ("AC", "A C"), ("DNR", "D N R"), ("DNI", "D N I"), ("POLST", "polst"), ("CHF", "C H F"),
          ("STEMI", "stemmy"), ("12-lead", "twelve lead"), ("c-spine", "C spine"), ("C-collar", "C collar"),
          ("911", "nine one one"), ("COPD", "C O P D"), ("CAD", "C A D"), ("CKD", "C K D"), ("ESRD", "E S R D"),
          ("DKA", "D K A"), ("CVA", "C V A"), ("TIA", "T I A"), ("LOC", "L O C"), ("TBI", "T B I"), ("GSW", "G S W"),
          ("UTI", "U T I"), ("NKDA", "N K D A"), ("PVCs", "P V C's"), ("EKG", "E K G"), ("AFib", "A fib"),
          ("RVR", "R V R"), ("ROSC", "rosk"), ("CPR", "C P R"), ("AED", "A E D"), ("BVM", "B V M"), ("IM", "I M"),
          ("IO", "I O"), ("O2", "O 2"), ("D10", "D 10"), ("D50", "D 50"), ("LR", "L R"), ("PM", "P M"), ("AM", "A M"),
          ("mcg", " micrograms"), ("mg", " milligrams"), ("mL", " milliliters"), ("ml", " milliliters"),
          ("%", " percent"), ("/", " over ")]


def spoken(text: str) -> str:
    """The script as it is read aloud: abbreviations spelled out, units and slashes said in words."""
    import re
    for a, b in SPOKEN:
        if a[0].isalnum() and a[-1].isalnum():
            text = re.sub(rf"(?<![A-Za-z]){re.escape(a)}(?![A-Za-z])", b, text)
        else:
            text = text.replace(a, b)
    return re.sub(r"[ ]{2,}", " ", text)


def make_wavs(names) -> None:
    """Piper reads each script at a slightly slower-than-default pace, a pause between sentences."""
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        text = script_text(name)
        (WAV_DIR / f"{name}.txt").write_text(text + "\n")
        (WAV_DIR / f"{name}.spoken.txt").write_text(spoken(text) + "\n")
        wav = WAV_DIR / f"{name}.wav"
        subprocess.run([str(PIPER), "-m", str(PIPER_VOICE), "--length-scale", "1.15",
                        "--sentence-silence", "0.4", "-f", str(wav)],
                       input=spoken(text).encode(), check=True, capture_output=True)
        print(f"[OK] {wav} ({wav_seconds(wav):.0f} s)")


def wav_seconds(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


# ---- the app -------------------------------------------------------------------------------------

class App:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.load(r)

    def post(self, path: str) -> dict:
        req = urllib.request.Request(self.base + path, method="POST", data=b"{}",
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{path}: {e.read().decode(errors='replace')[:200]}") from None

    def wait_state(self, wanted: Tuple[str, ...], timeout_s: float) -> str:
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout_s:
            st = self.get("/api/status")
            if st["state"] in wanted:
                return st["state"]
            time.sleep(1.0)
        raise TimeoutError(f"app did not reach {wanted} within {timeout_s:.0f} s")


class Captions(threading.Thread):
    """Reads /api/events and keeps the caption text (each final is the whole text so far)."""

    def __init__(self, base: str):
        super().__init__(name="captions", daemon=True)
        self.base = base.rstrip("/")
        self.finals: List[str] = []
        self.transcript_event: Optional[str] = None
        self.stop = threading.Event()

    def run(self) -> None:
        try:
            with urllib.request.urlopen(self.base + "/api/events", timeout=600) as r:
                for raw in r:
                    if self.stop.is_set():
                        return
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        ev = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "caption" and ev.get("final"):
                        if not self.finals or ev["final"] != self.finals[-1]:
                            self.finals.append(ev["final"])
                    elif ev.get("type") == "transcript":
                        self.transcript_event = ev.get("text")
        except Exception as e:  # noqa: BLE001 - the stream ending is not an error for the run
            if not self.stop.is_set():
                print(f"[!] event stream: {e}", file=sys.stderr)

    def text(self) -> str:
        """Every caption final is the whole text so far (listeners.py), so the last one is the captions."""
        return self.finals[-1] if self.finals else ""


# ---- scoring -------------------------------------------------------------------------------------

def word_diff(ref: str, hyp: str) -> Dict[str, List[str]]:
    """Reference words the hypothesis dropped, and the (ref -> heard) replacements, in order."""
    r, h = normalize(ref), normalize(hyp)
    dropped: List[str] = []
    replaced: List[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=r, b=h, autojunk=False).get_opcodes():
        if tag == "delete":
            dropped.append(" ".join(r[i1:i2]))
        elif tag == "replace":
            replaced.append(f"{' '.join(r[i1:i2])} -> {' '.join(h[j1:j2])}")
    return {"dropped": dropped, "replaced": replaced}


def stated(ref: str) -> Dict[str, Optional[str]]:
    """What the script says outright, to compare with the report: ETA minutes and activation words."""
    import re
    m = re.search(r"ETA[^0-9]{0,40}(\d+)\s*minutes", ref, flags=re.I)
    words = [w for w in ("STEMI", "trauma activation", "cath lab", "sepsis", "stroke") if w.lower() in ref.lower()]
    return {"eta_minutes": m.group(1) if m else None, "alert_words": ", ".join(words)}


# ---- one run -------------------------------------------------------------------------------------

def run_one(name: str, app: App, play: bool, wav: Optional[Path], sink: str, label: str,
            lead_s: float, tail_s: float, report_timeout_s: float) -> dict:
    ref = script_text(name)
    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    out = OUT_DIR / f"{name}_{label}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    st = app.get("/api/status")
    if st["state"] not in ("idle", "done", "error"):
        raise RuntimeError(f"app is busy ({st['state']}); finish or cancel that first")
    captions = Captions(app.base)
    captions.start()
    time.sleep(0.5)

    print(f"\n== {name} ({label}) -> {out}")
    app.post("/api/live/start")
    app.wait_state(("recording",), 15)
    t_start = time.monotonic()
    time.sleep(lead_s)
    if play:
        wav = wav or WAV_DIR / f"{name}.wav"
        print(f"   playing {wav.name} ({wav_seconds(wav):.0f} s) on {sink}")
        subprocess.run(["pw-play", "--target", sink, str(wav)], check=True)
    else:
        print("   READ THE SCRIPT NOW. Press Enter when you have finished.")
        input()
    time.sleep(tail_s)
    played_s = round(time.monotonic() - t_start, 1)
    app.post("/api/live/done")
    t_done = time.monotonic()
    state = app.wait_state(("done", "error", "idle"), report_timeout_s)
    report_wait_s = round(time.monotonic() - t_done, 1)
    captions.stop.set()

    status = app.get("/api/status")
    report = app.get("/api/result").get("report") if state == "done" else None
    transcript = (report or {}).get("transcript") or captions.transcript_event or ""
    cap_text = captions.text()

    w_t = wer(ref, transcript)
    w_c = wer(ref, cap_text)
    result = {
        "script": name, "label": label, "played_s": played_s, "report_wait_s": report_wait_s,
        "state": state, "error": status.get("error"),
        "words_ref": w_t[2], "words_heard": len(normalize(transcript)),
        "wer_transcript": w_t[0], "wer_captions": w_c[0],
        "transcript_diff": word_diff(ref, transcript),
        "report": {k: (report or {}).get(k) for k in ("alert_type", "alert_level", "eta_minutes")},
        "stated": stated(ref),
        "transcription_meta": (report or {}).get("_meta", {}).get("transcription"),
        "server": {"commit": status.get("commit"), "captions": status.get("captions"),
                   "transcriber": status.get("transcriber"),
                   "provider": status.get("settings", {}).get("provider"),
                   "model": status.get("settings", {}).get("model")},
    }
    (out / "reference.txt").write_text(ref + "\n")
    (out / "transcript.txt").write_text(transcript + "\n")
    (out / "captions.txt").write_text(cap_text + "\n")
    (out / "report.json").write_text(json.dumps(report, indent=1))
    (out / "result.json").write_text(json.dumps(result, indent=1))
    OUT_DIR.joinpath("INDEX.tsv").open("a").write(
        "\t".join(str(x) for x in (stamp, name, label, played_s, report_wait_s, w_t[0], w_c[0],
                                   result["report"]["alert_type"], result["report"]["alert_level"],
                                   result["report"]["eta_minutes"], result["stated"]["eta_minutes"],
                                   result["server"]["commit"], out)) + "\n")

    print(f"   transcript WER {w_t[0]:.0%} ({w_t[1]} edits / {w_t[2]} words), captions WER {w_c[0]:.0%}")
    print(f"   report: {result['report']['alert_type']} {result['report']['alert_level']} "
          f"ETA {result['report']['eta_minutes']} (script says ETA {result['stated']['eta_minutes']}, "
          f"{result['stated']['alert_words'] or 'no alert word'})")
    d = result["transcript_diff"]
    if d["dropped"]:
        print(f"   dropped: {' | '.join(d['dropped'])}")
    if d["replaced"]:
        print(f"   replaced: {' | '.join(d['replaced'][:12])}{' ...' if len(d['replaced']) > 12 else ''}")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("script", choices=SCRIPTS + ("all",))
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--play", action="store_true", help="play the Piper WAV on the speaker")
    mode.add_argument("--read", action="store_true", help="a person reads the script into the mic")
    mode.add_argument("--make-wavs", action="store_true", help="only (re)make the Piper WAVs")
    ap.add_argument("--server", default="http://127.0.0.1:5050")
    ap.add_argument("--wav", type=Path, help="play this WAV instead of $HANDOFF_DATASETS/scripts_v0/<script>.wav")
    ap.add_argument("--variant", default="", help="play $HANDOFF_DATASETS/scripts_v0/<script><variant>.wav, e.g. _radio "
                                                   "(made by scripts/corpus/radioize.py)")
    ap.add_argument("--sink", default=paths.air_sink(), help="PipeWire sink to play into (HANDOFF_AIR_SINK)")
    ap.add_argument("--label", default="usb-dongle", help="which mic / speaker this run used (for the index)")
    ap.add_argument("--lead", type=float, default=1.5, help="seconds of silence before the audio")
    ap.add_argument("--tail", type=float, default=2.0, help="seconds of silence after it, before Done")
    ap.add_argument("--report-timeout", type=float, default=400)
    args = ap.parse_args()

    names = SCRIPTS if args.script == "all" else (args.script,)
    if args.make_wavs:
        make_wavs(names)
        return 0
    if not (args.play or args.read):
        ap.error("choose --play, --read or --make-wavs")
    if args.play:
        missing = [n for n in names if not (args.wav or WAV_DIR / f"{n}{args.variant}.wav").exists()]
        if missing:
            if args.variant:
                ap.error(f"no WAV for {missing} with variant {args.variant!r}; run scripts/corpus/radioize.py first")
            make_wavs(missing)
    app = App(args.server)
    results = [run_one(n, app, args.play, args.wav or (WAV_DIR / f"{n}{args.variant}.wav" if args.variant else None),
                       args.sink, args.label, args.lead, args.tail, args.report_timeout) for n in names]
    print("\nscript   WER transcript  WER captions  report")
    for r in results:
        rp = r["report"]
        print(f"{r['script']:<8} {r['wer_transcript']:>13.0%} {r['wer_captions']:>13.0%}  "
              f"{rp['alert_type']} {rp['alert_level']} ETA {rp['eta_minutes']} (said {r['stated']['eta_minutes']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
