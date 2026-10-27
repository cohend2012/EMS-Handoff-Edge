"""
replay.py — play a recorded call-set through the live pipeline as a live call.

Purpose
    A call-set folder (~/callsets/CS014) holds one WAV per burst. Replay plays
    the bursts in epoch order with their real gaps, compressed by a speed
    factor (gaps only: a burst always plays at real time), through the same
    listeners and transcriber the mic uses, into an incident session.

The driver owns
    - burst start timing: the gap countdown, the start instant, the wall-clock
      log of every burst;
    - the audio played into the PipeWire null sink at the same instant the
      file source starts feeding, from the same samples, with a 50 ms click
      mixed over the first 50 ms so a recording can be checked for sync;
    - the end of the run: after the last burst it waits for the session to
      settle (no extraction running or due) and reports done.

Call-set timing
    callset_info.json carries a "bursts" list (feed, epoch, duration_sec,
    gap_from_previous_sec, source_file), written by
    scripts/corpus/enrich_callsets.py. Without it, feed and epoch are parsed
    from the source filename pattern mocomdps-<feed>-<epoch>, duration from
    the WAV, and the gap from epoch and duration. The corpus index is a third,
    optional source. A folder with none of these replays with no gaps.

Threads
    "replay-driver": the timing loop. "replay-transcribe" (in the server):
    transcribes each finished burst and adds it to the session. Neither
    touches the live mic path; nothing here runs unless replay was started.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .audio.capture import TARGET_RATE
from .audio.sources import load_wav

logger = logging.getLogger(__name__)

SOURCE_PATTERN = re.compile(r"mocomdps-(\d+)-(\d{9,11})")
CLICK_S = 0.05
CLICK_HZ = 1000.0
CLICK_AMP = 0.9


# ---------------------------------------------------------------------------
# Call-sets
# ---------------------------------------------------------------------------
@dataclass
class BurstClip:
    segment_id: str
    wav: Path
    feed: str = ""
    epoch: float = 0.0
    duration_s: float = 0.0
    gap_s: Optional[float] = None     # from the end of the previous burst; None for the first
    words: int = 0
    source_file: str = ""

    @property
    def overlap_s(self) -> float:
        """Seconds this burst started before the previous one ended (0 when it did not)."""
        return -self.gap_s if self.gap_s is not None and self.gap_s < 0 else 0.0


@dataclass
class CallSet:
    id: str
    folder: Path
    bursts: List[BurstClip]
    info: Dict[str, Any] = field(default_factory=dict)
    timing_source: str = "none"       # callset_info | filenames | index | none

    @property
    def total_audio_s(self) -> float:
        return sum(b.duration_s for b in self.bursts)

    @property
    def span_s(self) -> float:
        """First burst start to last burst end, real time."""
        if not self.bursts:
            return 0.0
        return max(b.epoch + b.duration_s for b in self.bursts) - self.bursts[0].epoch

    def summary(self) -> Dict[str, Any]:
        info = self.info
        patient = info.get("patient") or {}

        def joined(value) -> str:
            """callset_info fields are lists, or a plain string such as "not stated"."""
            if isinstance(value, str):
                return value
            return ", ".join(str(v) for v in (value or []))

        return {"id": self.id, "bursts": len(self.bursts), "words": sum(b.words for b in self.bursts),
                "audio_s": round(self.total_audio_s, 1), "span_s": round(self.span_s, 1),
                "hospital": joined(info.get("hospital")), "units": joined(info.get("units")),
                "ages": joined(patient.get("ages")), "sex": joined(patient.get("sex")),
                "keywords": joined(info.get("complaint_keywords")),
                "first_burst_utc": info.get("first_burst_utc", ""), "timing_source": self.timing_source}


def wav_duration_s(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate() or TARGET_RATE)
    except (OSError, wave.Error, EOFError):
        return 0.0


def parse_source_name(name: str) -> Optional[tuple]:
    """(feed, epoch) from a name like mocomdps-4390-1792827662.m4a, else None."""
    m = SOURCE_PATTERN.search(name or "")
    return (m.group(1), float(m.group(2))) if m else None


def read_index(index_csv: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """{segment_id: row} from the corpus index (~/labels/callsets.csv), or {} when absent."""
    if not index_csv:
        return {}
    try:
        with open(index_csv, newline="") as f:
            return {r["segment_id"]: r for r in csv.DictReader(f) if r.get("segment_id")}
    except (OSError, KeyError, csv.Error):
        return {}


def load_callset(folder: Path, index_csv: Optional[Path] = None) -> CallSet:
    folder = Path(folder)
    info_path = folder / "callset_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"{folder} has no callset_info.json")
    info = json.loads(info_path.read_text())
    cs_id = info.get("callset_id") or folder.name
    segments = info.get("segments") or []
    detail = {b.get("segment_id"): b for b in (info.get("bursts") or []) if isinstance(b, dict)}
    index = read_index(index_csv)
    bursts: List[BurstClip] = []
    sources = set()
    for sid in segments:
        wav = folder / f"{sid}.wav"
        if not wav.exists():
            raise FileNotFoundError(f"{cs_id}: {wav.name} is missing")
        d = detail.get(sid) or {}
        words = int(d.get("words") or 0)
        if not words and (folder / f"{sid}.txt").exists():
            words = len((folder / f"{sid}.txt").read_text().split())
        clip = BurstClip(segment_id=sid, wav=wav, words=words, source_file=str(d.get("source_file") or ""))
        if d.get("epoch") is not None:
            clip.feed, clip.epoch = str(d.get("feed") or info.get("feed") or ""), float(d["epoch"])
            clip.duration_s = float(d.get("duration_sec") or wav_duration_s(wav))
            sources.add("callset_info")
        else:
            parsed = parse_source_name(clip.source_file) or parse_source_name(wav.name)
            if parsed is None:
                row = index.get(sid) or {}
                parsed = parse_source_name(row.get("source_file", "")) or \
                    ((row.get("feed", ""), float(row["epoch"])) if row.get("epoch") else None)
                if parsed:
                    clip.source_file = row.get("source_file", "")
                    sources.add("index")
            else:
                sources.add("filenames")
            clip.duration_s = wav_duration_s(wav)
            if parsed:
                clip.feed, clip.epoch = parsed[0], float(parsed[1])
            else:
                clip.feed = str(info.get("feed") or "")
        bursts.append(clip)
    if any(b.epoch == 0.0 for b in bursts):
        # no timing for some burst: play back to back, in the listed order
        t = 0.0
        for b in bursts:
            if b.epoch == 0.0:
                b.epoch = t
            t = b.epoch + b.duration_s
        timing = "none"
    else:
        bursts.sort(key=lambda b: b.epoch)
        timing = "callset_info" if sources == {"callset_info"} else ("+".join(sorted(sources)) or "none")
    prev_end = None
    for b in bursts:
        if detail.get(b.segment_id, {}).get("gap_from_previous_sec") is not None and timing == "callset_info":
            b.gap_s = float(detail[b.segment_id]["gap_from_previous_sec"])
        else:
            b.gap_s = None if prev_end is None else round(b.epoch - prev_end, 1)
        prev_end = max(prev_end or 0.0, b.epoch + b.duration_s)
    if bursts:
        bursts[0].gap_s = None
    return CallSet(id=cs_id, folder=folder, bursts=bursts, info=info, timing_source=timing)


def list_callsets(root: Path, index_csv: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Summaries of every call-set folder under root, in id order; unreadable ones carry an error."""
    out = []
    root = Path(root).expanduser()
    if not root.is_dir():
        return out
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and (p / "callset_info.json").exists()):
        try:
            out.append(load_callset(folder, index_csv).summary())
        except Exception as e:  # noqa: BLE001 - one bad folder must not hide the rest
            out.append({"id": folder.name, "error": f"{type(e).__name__}: {e}"})
    return out


# ---------------------------------------------------------------------------
# Audio into the null sink
# ---------------------------------------------------------------------------
def click(seconds: float = CLICK_S, hz: float = CLICK_HZ, amp: float = CLICK_AMP) -> np.ndarray:
    t = np.arange(int(seconds * TARGET_RATE)) / TARGET_RATE
    env = np.minimum(1.0, np.minimum(t, seconds - t) / 0.005)        # 5 ms ramps, no spectral splash
    return (amp * np.sin(2 * np.pi * hz * t) * env).astype(np.float32)


def prepare_sink_wav(clip_wav: Path, out: Path, with_click: bool = True) -> Path:
    """The burst's samples with the sync click mixed over its first 50 ms, as 16 kHz s16 WAV."""
    audio = load_wav(Path(clip_wav)).copy()
    if with_click:
        c = click()
        n = min(len(c), len(audio))
        audio[:n] = np.clip(audio[:n] + c[:n], -1.0, 1.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TARGET_RATE)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return out


class SinkPlayer:
    """Plays a WAV into a PipeWire/PulseAudio sink by name (pw-play). Missing tools are logged once."""

    def __init__(self, sink: str = "handoff_replay", command: Optional[List[str]] = None):
        self.sink = sink
        self.command = command if command is not None else self._find()
        self.procs: List[subprocess.Popen] = []
        self.available = bool(self.command)
        self.last_started: Optional[float] = None

    @staticmethod
    def _find() -> Optional[List[str]]:
        if shutil.which("pw-play"):
            return ["pw-play", "--target", "{sink}", "{file}"]
        if shutil.which("paplay"):
            return ["paplay", "--device", "{sink}", "{file}"]
        logger.warning("Neither pw-play nor paplay found: replay audio will not reach the sink")
        return None

    def play(self, path: Path) -> Optional[subprocess.Popen]:
        if not self.command:
            return None
        cmd = [c.format(sink=self.sink, file=str(path)) for c in self.command]
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError as e:
            logger.warning("Sink playback failed (%s): %s", " ".join(cmd), e)
            self.available = False
            return None
        self.last_started = time.time()
        self.procs = [p for p in self.procs if p.poll() is None] + [proc]
        return proc

    def stop(self) -> None:
        for p in self.procs:
            if p.poll() is None:
                p.terminate()
        self.procs = []


def ensure_null_sink(name: str = "handoff_replay") -> bool:
    """Create the PipeWire null sink if it is not there. True when it exists afterwards."""
    if not shutil.which("pw-cli"):
        return False
    try:
        listed = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=5).stdout
        if f'node.name = "{name}"' in listed:
            return True
        props = (f'{{ factory.name=support.null-audio-sink node.name={name} node.description="HANDOFF replay sink" '
                 'media.class=Audio/Sink object.linger=true audio.position=[MONO] monitor.channel-volumes=true }')
        subprocess.run(["pw-cli", "create-node", "adapter", props], capture_output=True, text=True, timeout=5)
        listed = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=5).stdout
        return f'node.name = "{name}"' in listed
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Could not create null sink %s: %s", name, e)
        return False


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
class ReplayHost:
    """What the driver needs from its owner (the server's controller)."""

    def start_burst(self, clip: BurstClip, number: int) -> threading.Event:
        """Start feeding the clip into the pipeline now; the event is set when the clip has ended."""
        raise NotImplementedError

    def settled(self) -> bool:
        """True when no extraction is running, queued or due for the replay's session."""
        raise NotImplementedError


class ReplayDriver:
    def __init__(self, callset: CallSet, host: ReplayHost, sink: Optional[SinkPlayer] = None,
                 speed: float = 1.0, emit: Callable[[dict], None] = lambda e: None,
                 work_dir: Optional[Path] = None, settle_timeout_s: float = 600.0,
                 sink_lead_s: float = 0.0, before_first_burst: Optional[Callable[[], None]] = None):
        self.callset = callset
        self.host = host
        self.sink = sink
        self.speed = max(0.01, float(speed))
        # Positive: start the sink player this much before the file source (its startup). Negative:
        # after it. Measured on the Jetson from recordings: with +100 ms the click sat 256 ms ahead
        # of the flash in the file, so the calibrated value is about -155 ms (config default).
        self.sink_lead_s = float(sink_lead_s)
        self._before_first_burst = before_first_burst     # e.g. start the recorder and wait for it
        self._emit = emit
        self.work_dir = Path(work_dir) if work_dir else Path("/tmp") / "handoff_replay"
        self.settle_timeout_s = settle_timeout_s
        self.log: List[Dict[str, Any]] = []        # one entry per burst start, wall clock
        self.phase = "idle"
        self.burst_number = 0
        self.seconds_left = 0.0
        self.started_wall: Optional[float] = None
        self.finished_wall: Optional[float] = None
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -- control ---------------------------------------------------------------
    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="replay-driver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.sink:
            self.sink.stop()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    @property
    def elapsed_s(self) -> float:
        if self.started_wall is None:
            return 0.0
        return (self.finished_wall or time.time()) - self.started_wall

    def status(self) -> Dict[str, Any]:
        return {"callset": self.callset.id, "phase": self.phase, "burst": self.burst_number,
                "of": len(self.callset.bursts), "seconds_left": round(self.seconds_left, 1),
                "elapsed_s": round(self.elapsed_s, 1), "speed": self.speed, "error": self.error,
                "timing_source": self.callset.timing_source, "log": list(self.log)}

    # -- the loop ----------------------------------------------------------------
    def _publish(self, **extra: Any) -> None:
        self._emit({"type": "replay", **self.status(), **extra})

    def _run(self) -> None:
        self.started_wall = time.time()
        try:
            prepared = {b.segment_id: prepare_sink_wav(b.wav, self.work_dir / self.callset.id / f"{b.segment_id}.sink.wav")
                        for b in self.callset.bursts} if self.sink else {}
            prev_end_wall = time.monotonic()
            for i, clip in enumerate(self.callset.bursts, 1):
                gap = 0.0 if clip.gap_s is None else max(0.0, clip.gap_s) / self.speed
                note = (f"overlapped the previous burst by {clip.overlap_s:.0f} s on feed {clip.feed}, played after it"
                        if clip.overlap_s else "")
                deadline = prev_end_wall + gap
                self.phase, self.burst_number = "gap", i
                while True:
                    self.seconds_left = max(0.0, deadline - time.monotonic())
                    self._publish(next_feed=clip.feed, note=note)
                    if self.seconds_left <= 0 or self._stop.is_set():
                        break
                    time.sleep(min(0.2, self.seconds_left))
                if self._stop.is_set():
                    return
                if i == 1 and self._before_first_burst is not None:
                    self._before_first_burst()
                self.phase, self.seconds_left = "burst", 0.0
                # The sink player takes about 100 ms to open its stream: start it a fixed lead
                # earlier so the recording's audio lines up with the captions and the flash.
                sink_wall = None
                play_sink = self.sink is not None and clip.segment_id in prepared
                if play_sink and self.sink_lead_s >= 0:
                    self.sink.play(prepared[clip.segment_id])
                    sink_wall = self.sink.last_started
                    if self.sink_lead_s:
                        time.sleep(self.sink_lead_s)
                wall = time.time()
                ended = self.host.start_burst(clip, i)
                if play_sink and self.sink_lead_s < 0:
                    time.sleep(-self.sink_lead_s)
                    self.sink.play(prepared[clip.segment_id])
                    sink_wall = self.sink.last_started
                entry = {"burst": i, "segment_id": clip.segment_id, "feed": clip.feed, "epoch": clip.epoch,
                         "duration_s": round(clip.duration_s, 2), "gap_s": clip.gap_s, "played_gap_s": round(gap, 2),
                         "start_wall": wall, "start_iso": time.strftime("%H:%M:%S", time.localtime(wall)) + f".{int(wall % 1 * 1000):03d}",
                         "sink_lead_ms": round((wall - sink_wall) * 1000, 1) if sink_wall else None, "note": note}
                self.log.append(entry)
                logger.info("Replay %s: burst %d/%d %s started at %s (gap %.1f s%s)", self.callset.id, i,
                            len(self.callset.bursts), clip.segment_id, entry["start_iso"], gap, f"; {note}" if note else "")
                self._publish(flash=True, note=note, started=entry)
                if not ended.wait(timeout=clip.duration_s + 10):
                    logger.warning("Replay %s: burst %d did not signal its end in time", self.callset.id, i)
                prev_end_wall = time.monotonic()
                if self._stop.is_set():
                    return
            self.phase = "waiting"
            self._publish()
            end = time.monotonic() + self.settle_timeout_s
            while time.monotonic() < end and not self._stop.is_set():
                if self.host.settled():
                    break
                time.sleep(0.25)
            self.phase = "done"
        except Exception as e:  # noqa: BLE001 - the run fails loudly, never silently
            logger.exception("Replay %s failed", self.callset.id)
            self.error = f"{type(e).__name__}: {e}"
            self.phase = "error"
        finally:
            self.finished_wall = time.time()
            self._publish()
