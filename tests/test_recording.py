"""Recording a replay: the ffmpeg command, the window lookup, the index line, and the sync check."""

import json
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from ems_handoff import recording, sync_check
from ems_handoff.recording import (Geometry, append_index_line, chromium_command, close_recording_browsers, ffmpeg_command,
                                   find_window, window_geometry)
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


def test_chromium_window_is_an_app_window_at_the_origin_with_the_page_scale(tmp_path):
    cmd = chromium_command("/opt/chrome", "http://127.0.0.1:5050/", 1920, 1080, tmp_path, scale=1.5)
    assert cmd[0] == "/opt/chrome" and "--app=http://127.0.0.1:5050/" in cmd
    assert "--window-size=1920,1080" in cmd and "--window-position=0,0" in cmd
    assert "--force-device-scale-factor=1.5" in cmd, "the 720p layout at 1080p"
    assert "--class=handoff-record" in cmd and f"--user-data-dir={tmp_path}" in cmd
    assert "--force-device-scale-factor=1" in chromium_command("c", "u", 1280, 720, tmp_path)
    assert "--kiosk" not in cmd and chromium_command("c", "u", 1920, 1080, tmp_path, kiosk=True)[-1] == "--kiosk"


def test_default_recording_is_1080p_with_the_720p_layout(settings):
    assert (settings.replay_record_width, settings.replay_record_height, settings.replay_record_scale) == (1920, 1080, 1.5)
    assert settings.replay_record_width / settings.replay_record_scale == 1280


def test_ffmpeg_capture_size_is_rounded_down_to_even():
    cmd = ffmpeg_command(Path("/tmp/x/a.mp4"), Geometry(66, 32, 1280, 757))
    assert "1280x756" in " ".join(cmd), "an odd height would stop the encoder from opening"
    assert "+66,32" in " ".join(cmd)


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


def _spawn(code, *argv):
    import sys
    return subprocess.Popen([sys.executable, "-c", code, *argv], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def test_close_recording_browsers_terminates_kills_survivors_and_helpers_and_matches_rewritten_argv(tmp_path):
    import os
    import uuid
    cls = f"handoff-test-{uuid.uuid4().hex[:8]}"
    polite = _spawn("import time; time.sleep(30)", f"--class={cls}")
    stubborn = _spawn("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)", f"--class={cls}")
    helper = _spawn("import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
                    f"--type=gpu-process", f"--class={cls}")
    other = _spawn("import time; time.sleep(30)", f"--class={cls}-not-this-one")
    rewritten = subprocess.Popen(["bash", "-c", f'exec -a "chrome --app=x --class={cls} --kiosk" sleep 30'],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import time
        time.sleep(0.5)
        assert sorted(recording._main_processes(cls)) == sorted([polite.pid, stubborn.pid, rewritten.pid]), "helpers are not main processes"
        assert helper.pid in recording._main_processes(cls, helpers=True)
        for name in ("SingletonLock", "SingletonSocket"):
            (tmp_path / name).write_text("x")
        n = close_recording_browsers(cls, wait_s=0.5, profile_dir=tmp_path)
        assert n == 3
        assert polite.wait(timeout=5) is not None and stubborn.wait(timeout=5) is not None
        assert helper.wait(timeout=5) is not None and rewritten.wait(timeout=5) is not None, "the orphaned helper is killed too"
        assert other.poll() is None, "another class is left alone"
        assert not (tmp_path / "SingletonLock").exists() and not (tmp_path / "SingletonSocket").exists()
        assert recording._main_processes(cls, helpers=True) == []
    finally:
        for pr in (polite, stubborn, helper, other, rewritten):
            if pr.poll() is None:
                pr.kill()
                pr.wait()


def test_wait_for_geometry_returns_when_the_window_reaches_the_size(monkeypatch):
    sizes = iter([(1854, 1011), (1854, 1011), (1920, 1080)])

    def fake_x(display, *cmd, timeout=5):
        w, h = next(sizes)
        return f"  Absolute upper-left X:  0\n  Absolute upper-left Y:  0\n  Width: {w}\n  Height: {h}\n"

    monkeypatch.setattr(recording, "_x", fake_x)
    geo = recording.wait_for_geometry(":1", "0x1", 1920, 1080, timeout=3.0)
    assert (geo.width, geo.height) == (1920, 1080)
    monkeypatch.setattr(recording, "_x", lambda display, *cmd, timeout=5: "  Absolute upper-left X:  66\n  Absolute upper-left Y:  69\n  Width: 1854\n  Height: 1011\n")
    geo = recording.wait_for_geometry(":1", "0x1", 1920, 1080, timeout=0.3)
    assert (geo.width, geo.height) == (1854, 1011), "after the timeout the last geometry seen is returned"


def test_window_pid_reads_net_wm_pid(monkeypatch):
    monkeypatch.setattr(recording, "_x", lambda display, *cmd, timeout=5: "_NET_WM_PID(CARDINAL) = 4242\n")
    assert recording.window_pid(":1", "0x1") == 4242
    monkeypatch.setattr(recording, "_x", lambda display, *cmd, timeout=5: "_NET_WM_PID:  not found.\n")
    assert recording.window_pid(":1", "0x1") is None


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
    assert short["bursts"][1]["click_s"] is None and "no click found" in sync_check.format_report(short)
    missed = sync_check.check(synthetic_recording(tmp_path / "missed.mp4", [3.0, 4.6], [1.04, 3.0, 4.6]), expected=3)
    assert not missed["pass"] and missed["bursts"][0]["flash_s"] is None, "the first flash is reported missing"
    assert [b["offset_ms"] for b in missed["bursts"][1:]] == [0.0, 0.0] or all(abs(b["offset_ms"]) <= 35 for b in missed["bursts"][1:]), \
        "the later bursts still pair with their own clicks"


def test_marker_is_found_at_the_1080p_size_too():
    """At 1080p with the page zoomed 1.5x the 48 px marker is 72 px: 18 px at analysis scale."""
    frames = np.full((30, sync_check.FRAME_H, sync_check.FRAME_W), 20, np.uint8)
    frames[10:12, 8:26, 290:308] = 255                      # 18x18 flash at frames 10-11
    frames[20:, 40:80, 40:200] = 220                          # a text block appearing is not the marker
    assert sync_check.locate_marker(frames) == (16, 298)
    times = [i / 15 for i in range(30)]
    assert sync_check.find_flashes(frames, times) == [round(10 / 15, 4)]


def test_marker_search_ignores_a_bright_flash_elsewhere_in_the_top_band():
    """CS003 on 2026-10-25: a card flashing bright at the left of the band pulled the median to (40, 238)
    and the three real flashes at (31, 305) were missed. The search is confined to the top-right."""
    frames = np.full((40, sync_check.FRAME_H, sync_check.FRAME_W), 20, np.uint8)
    for i in (5, 15, 25):
        frames[i:i + 2, 26:38, 299:311] = 255                 # the real marker, three flashes
    for i in (8, 9, 18, 19, 28, 29, 33, 34):
        frames[i:i + 2, 36:48, 230:246] = 255                 # a decoy of marker size, left of the marker
    r, c = sync_check.locate_marker(frames)
    assert abs(r - 32) <= 2 and abs(c - 305) <= 2, (r, c)
    times = [i / 15 for i in range(40)]
    assert len(sync_check.find_flashes(frames, times)) == 3


def test_pair_events_matches_nearest_and_keeps_strays():
    pairs = sync_check.pair_events([1.0, 3.0, 4.6], [1.04, 3.0, 4.72])
    assert [p["offset_ms"] for p in pairs] == [40.0, 0.0, 120.0]
    pairs = sync_check.pair_events([3.0, 4.6], [1.04, 3.0, 4.6])
    assert pairs[0]["flash_s"] is None and pairs[0]["click_s"] == 1.04 and pairs[1]["offset_ms"] == 0.0
    pairs = sync_check.pair_events([1.0, 9.0], [1.0])
    assert pairs[1]["click_s"] is None and pairs[1]["flash_s"] == 9.0
