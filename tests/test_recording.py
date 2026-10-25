"""Recording a replay: the ffmpeg command, the window lookup, the index line, and the sync check."""

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from ems_handoff import recording, sync_check
from ems_handoff.recording import Geometry, append_index_line, ffmpeg_command, find_window, window_geometry
from ems_handoff.replay import click

TREE = """xwininfo: Window id: 0x5a3 (the root window) (has no name)

  Root window id: 0x5a3 (the root window) (has no name)
  Parent window id: 0x0 (none)
     7 children:
     0x800023 "HANDOFF.AI": ("mutter-x11-frames" "mutter-x11-frames")  1308x786+-14+20  +-14+20
        1 child:
        0x2600003 "HANDOFF.AI": ("127.0.0.1" "handoff-record")  1280x720+14+49  +0+69
     0x2e00001 "chrome": ("chrome" "handoff-record")  10x10+10+10  +10+10
     0x80001f "HANDOFF.AI — Mozilla Firefox": ("mutter-x11-frames" "mutter-x11-frames")  943x1077+-14+20  +-14+20
"""
INFO = """xwininfo: Window id: 0x2600003 "HANDOFF.AI"

  Absolute upper-left X:  0
  Absolute upper-left Y:  69
  Relative upper-left X:  14
  Relative upper-left Y:  49
  Width: 1280
  Height: 720
  Map State: IsViewable
"""


def test_ffmpeg_command_one_process_wallclock_passthrough_lightest():
    cmd = ffmpeg_command(Path("/tmp/x/CS008.mp4"), Geometry(0, 69, 1280, 720), ":1", "handoff_replay", 15)
    joined = " ".join(cmd)
    assert cmd[0] == "ffmpeg" and cmd[-1] == "/tmp/x/CS008.mp4"
    assert joined.count("-use_wallclock_as_timestamps 1") == 2, "both inputs on the wall clock"
    assert "-f x11grab -framerate 15 -video_size 1280x720 -draw_mouse 0 -i :1.0+0,69" in joined
    assert "-f pulse -i handoff_replay.monitor" in joined
    assert "-preset ultrafast" in joined and "-vsync passthrough" in joined
    assert "-map 0:v:0 -map 1:a:0" in joined and "-c:a aac" in joined
    assert cmd.index("-i") < cmd.index("-c:v"), "no post-mux: one encode of both inputs"


def test_find_window_skips_helper_windows_and_reads_client_geometry(monkeypatch):
    calls = []

    def fake_x(display, *cmd, timeout=5):
        calls.append(cmd)
        return TREE if cmd[1] == "-root" else INFO

    monkeypatch.setattr(recording, "_x", fake_x)
    assert find_window(":1", "handoff-record") == "0x2600003", "the 1280x720 page window, not the 10x10 helper"
    assert find_window(":1", "nothing") is None
    geo = window_geometry(":1", "0x2600003")
    assert (geo.x, geo.y, geo.width, geo.height) == (0, 69, 1280, 720)
    assert geo.x11grab_input == "+0,69" and geo.size == "1280x720"


def test_index_line_is_appended_with_a_pending_verdict(tmp_path):
    sync = {"pass": True, "flashes": 1, "expected_bursts": 1, "max_abs_offset_ms": 23.0}
    line = append_index_line(tmp_path, "CS008", "groq", "openai/gpt-oss-120b", "abc1234", 95.4, sync,
                             tmp_path / "2026-10-25_abc1234" / "CS008.mp4")
    text = (tmp_path / "INDEX.md").read_text()
    assert text.startswith("# Replay recordings") and "| call-set | provider |" in text
    assert line in text and "| 1:35 | pass (1/1 bursts, worst 23 ms) | pending | `2026-10-25_abc1234/CS008.mp4` |" in line
    append_index_line(tmp_path, "CS014", "groq", "m", "abc1234", None, {"pass": False, "flashes": 4, "expected_bursts": 5,
                                                                        "max_abs_offset_ms": 210.0}, tmp_path / "x.mp4")
    assert "FAIL (4/5 bursts, worst 210 ms)" in (tmp_path / "INDEX.md").read_text().splitlines()[-1]


# -- the sync check on a synthetic recording ------------------------------------------
FPS = 15
W, H = 1280, 720


def synthetic_recording(path: Path, flashes_s, clicks_s, seconds=6.0, marker=(1216, 80)):
    """A 1280x720 video with the marker square white for two frames at each flash time, and an
    audio track with a 50 ms 1 kHz click at each click time, muxed by ffmpeg like the real capture."""
    n = int(seconds * FPS)
    frames = np.full((n, H, W), 20, np.uint8)
    for t in flashes_s:
        i = int(round(t * FPS))
        for j in (i, i + 1):
            if j < n:
                frames[j, marker[1]:marker[1] + 48, marker[0]:marker[0] + 48] = 255
    # some page-like changes: text blocks appearing elsewhere must not look like the marker
    frames[int(2.2 * FPS):, 400:460, 100:700] = 200
    audio = np.zeros(int(seconds * 16000), np.float32)
    t = np.arange(len(audio)) / 16000
    audio += 0.05 * np.sin(2 * np.pi * 220 * t)
    c = click()
    for ct in clicks_s:
        i = int(ct * 16000)
        audio[i:i + len(c)] += c[:len(audio) - i]
    wav = path.with_suffix(".wav")
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(wav), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)]
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return path


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg not installed")
def test_sync_check_measures_the_click_to_flash_offset_per_burst(tmp_path):
    mp4 = synthetic_recording(tmp_path / "rec.mp4", flashes_s=[1.0, 3.0, 4.6], clicks_s=[1.04, 3.0, 4.72])
    r = sync_check.check(mp4, expected=3)
    assert r["flashes"] == 3 and r["clicks"] == 3
    assert [b["burst"] for b in r["bursts"]] == [1, 2, 3]
    offsets = [b["offset_ms"] for b in r["bursts"]]
    assert abs(offsets[0] - 40) <= 35 and abs(offsets[1]) <= 35 and abs(offsets[2] - 120) <= 35, offsets
    assert r["pass"] and r["max_abs_offset_ms"] <= 150
    assert r["marker_at"] is not None and abs(r["marker_at"][1] - (1216 + 24) / 4) < 6
    text = sync_check.format_report(r)
    assert "PASS" in text and "burst 3:" in text
    bad = sync_check.check(synthetic_recording(tmp_path / "bad.mp4", [1.0, 3.0], [1.3, 3.0]), expected=2)
    assert not bad["pass"] and bad["bursts"][0]["offset_ms"] > 150
    short = sync_check.check(synthetic_recording(tmp_path / "short.mp4", [1.0, 3.0], [1.0]), expected=2)
    assert not short["pass"] and short["clicks"] == 1 and "flash times" in sync_check.format_report(short)
