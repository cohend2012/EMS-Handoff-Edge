"""
sync_check.py — are the audio and the video of a replay recording on one clock?

The replay driver plays a 50 ms 1 kHz click into the sink and flashes the
banner's sync marker (a 48 px square, black to white for two frames) at every
burst start. This finds the clicks in the audio track and the flashes in the
video track and reports the offset per burst. A recording passes when every
burst is within PASS_MS.

    python scripts/check_sync.py ~/recordings/<date>_<commit>/CS008.mp4
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PASS_MS = 150.0
CLICK_HZ = 1000.0
AUDIO_RATE = 16000
FRAME_W, FRAME_H = 320, 180          # analysis size: the 1280x720 frame at a quarter
MARKER_PX = 12                       # the 48 px marker at a quarter
MIN_SEPARATION_S = 0.5


def _run(cmd: List[str]) -> bytes:
    return subprocess.run(cmd, capture_output=True, timeout=600, check=True).stdout


def stream_starts(path: Path) -> Dict[str, float]:
    out = _run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time", "-of", "json", str(path)])
    starts = {}
    for st in json.loads(out)["streams"]:
        try:
            starts[st["codec_type"]] = float(st.get("start_time") or 0.0)
        except ValueError:
            starts[st["codec_type"]] = 0.0
    return starts


def audio_track(path: Path) -> np.ndarray:
    raw = _run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", "-"])
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def find_clicks(audio: np.ndarray, rate: int = AUDIO_RATE, hz: float = CLICK_HZ) -> List[float]:
    """Onsets (seconds) of windows dominated by the click tone, at least MIN_SEPARATION_S apart."""
    win, hop = int(0.010 * rate), int(0.005 * rate)
    if len(audio) < win:
        return []
    n = (len(audio) - win) // hop + 1
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    frames = audio[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    k = np.exp(-2j * np.pi * hz * np.arange(win) / rate)
    tone = np.abs(frames @ k) ** 2 / win
    total = (frames ** 2).sum(axis=1) + 1e-9
    ratio = tone / total
    energy = total / win
    floor = np.percentile(energy, 50) + 1e-9
    strong = (energy > max(20 * floor, 1e-4)) & (ratio > 0.45)
    times, last = [], -10.0
    for i in np.flatnonzero(strong):
        t = i * hop / rate
        if t - last >= MIN_SEPARATION_S:
            times.append(round(t, 4))
        last = t
    return times


def video_frames(path: Path) -> np.ndarray:
    raw = _run(["ffmpeg", "-v", "error", "-i", str(path), "-an", "-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "gray",
                "-s", f"{FRAME_W}x{FRAME_H}", "-"])
    n = len(raw) // (FRAME_W * FRAME_H)
    return np.frombuffer(raw[:n * FRAME_W * FRAME_H], dtype=np.uint8).reshape(n, FRAME_H, FRAME_W)


def frame_times(path: Path) -> List[float]:
    out = _run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "frame=pts_time,best_effort_timestamp_time",
                "-of", "json", str(path)])
    times: List[float] = []
    for f in json.loads(out).get("frames", []):
        t = f.get("best_effort_timestamp_time") or f.get("pts_time")
        times.append(float(t) if t not in (None, "N/A") else (times[-1] if times else 0.0))
    return times


def locate_marker(frames: np.ndarray, band: float = 0.35) -> Optional[tuple]:
    """(row, col) of the marker at analysis scale: the small square that flips dark -> bright."""
    if len(frames) < 2:
        return None
    top = int(frames.shape[1] * band)
    prev, cur = frames[:-1, :top, :].astype(np.int16), frames[1:, :top, :].astype(np.int16)
    flips = (prev < 70) & (cur > 190)
    counts = flips.reshape(len(flips), -1).sum(axis=1)
    lo, hi = (MARKER_PX * MARKER_PX) * 0.4, (MARKER_PX * MARKER_PX) * 3.0
    candidates = np.flatnonzero((counts >= lo) & (counts <= hi))
    if len(candidates) == 0:
        return None
    rows, cols = [], []
    for i in candidates:
        ys, xs = np.nonzero(flips[i])
        rows.append(np.median(ys))
        cols.append(np.median(xs))
    return int(np.median(rows)), int(np.median(cols))


def find_flashes(frames: np.ndarray, times: List[float], marker: Optional[tuple] = None) -> List[float]:
    marker = marker or locate_marker(frames)
    if marker is None:
        return []
    r, c = marker
    h = MARKER_PX // 2
    box = frames[:, max(0, r - h):r + h, max(0, c - h):c + h].reshape(len(frames), -1).mean(axis=1)
    bright = box > 128
    out, last = [], -10.0
    for i in range(len(bright)):
        if bright[i] and (i == 0 or not bright[i - 1]):
            t = times[i] if i < len(times) else times[-1]
            if t - last >= MIN_SEPARATION_S:
                out.append(round(t, 4))
            last = t
    return out


def check(path: Path, expected: Optional[int] = None, pass_ms: float = PASS_MS) -> Dict[str, Any]:
    path = Path(path)
    starts = stream_starts(path)
    audio = audio_track(path)
    clicks = [t + starts.get("audio", 0.0) for t in find_clicks(audio)]
    frames = video_frames(path)
    times = frame_times(path)
    if len(times) != len(frames):
        times = (times + [times[-1] if times else 0.0] * len(frames))[:len(frames)]
    marker = locate_marker(frames)
    flashes = [t + starts.get("video", 0.0) for t in find_flashes(frames, times, marker)]
    bursts = []
    for i, (f, c) in enumerate(zip(flashes, clicks), 1):
        bursts.append({"burst": i, "flash_s": round(f, 3), "click_s": round(c, 3), "offset_ms": round((c - f) * 1000, 1)})
    offsets = [abs(b["offset_ms"]) for b in bursts]
    n_expected = expected if expected is not None else max(len(flashes), len(clicks))
    ok = (len(flashes) == len(clicks) == n_expected and n_expected > 0 and all(o <= pass_ms for o in offsets))
    return {"file": str(path), "pass": ok, "pass_ms": pass_ms, "expected_bursts": n_expected,
            "flashes": len(flashes), "clicks": len(clicks), "marker_at": list(marker) if marker else None,
            "frames": int(len(frames)), "audio_s": round(len(audio) / AUDIO_RATE, 2),
            "max_abs_offset_ms": max(offsets) if offsets else None, "bursts": bursts,
            "flash_times": [round(x, 3) for x in flashes], "click_times": [round(x, 3) for x in clicks]}


def format_report(r: Dict[str, Any]) -> str:
    lines = [f"{r['file']}: {'PASS' if r['pass'] else 'FAIL'} (rule: every burst within {r['pass_ms']:.0f} ms)",
             f"  frames {r['frames']} · audio {r['audio_s']} s · flashes {r['flashes']} · clicks {r['clicks']} · expected {r['expected_bursts']}"
             f" · marker at {r['marker_at']}"]
    for b in r["bursts"]:
        over = "" if abs(b["offset_ms"]) <= r["pass_ms"] else "  <-- over"
        lines.append(f"  burst {b['burst']}: flash {b['flash_s']:.3f} s · click {b['click_s']:.3f} s · offset {b['offset_ms']:+.1f} ms{over}")
    if r["flashes"] != r["clicks"]:
        lines.append(f"  flash times {r['flash_times']}; click times {r['click_times']}")
    return "\n".join(lines)
