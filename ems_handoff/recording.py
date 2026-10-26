"""
recording.py — record a replay on the Jetson: the app window and the null sink, one clock.

How it works (docs/design/incident-sessions.md, "Recording the acceptance run")
    A Chromium app window shows the page at exactly 1280x720. One ffmpeg process
    grabs that window region from the X display and the null sink's monitor,
    both with wall-clock timestamps, so the video and the audio share a clock
    and a dropped frame never shortens the video. Lightest preset, 15 fps. No
    post-mux. The replay driver plays a 50 ms click into the sink and flashes
    the banner marker at every burst start; scripts/check_sync.py measures the
    offset between the two in the finished file.

Outputs
    ~/recordings/<date>_<commit>/<callset>.mp4 with <callset>.summary.json (the
    replay status) and <callset>.sync.json (the sync report) beside it.

Nothing here runs unless a replay is started with record=true.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

CHROME_CANDIDATES = [
    str(Path.home() / "pw-browsers" / "chromium-*" / "chrome-linux*" / "chrome"),
    "/opt/pw-browsers/chromium-*/chrome-linux*/chrome",
]


def find_chromium(configured: str = "") -> Optional[str]:
    """A Chromium binary: the configured path, Playwright's build, or one on PATH."""
    if configured and os.path.exists(configured):
        return configured
    for pattern in CHROME_CANDIDATES:
        hits = sorted(glob.glob(pattern), reverse=True)
        if hits:
            return hits[0]
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            return path
    return None


@dataclass
class Geometry:
    x: int
    y: int
    width: int
    height: int

    @property
    def x11grab_input(self) -> str:
        return f"+{self.x},{self.y}"

    @property
    def size(self) -> str:
        return f"{self.width}x{self.height}"


def _x(display: str, *cmd: str, timeout: float = 5) -> str:
    env = dict(os.environ, DISPLAY=display)
    return subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, env=env).stdout


_TREE_LINE = re.compile(r'\s*(0x[0-9a-f]+)\s+"[^"]*":\s+\("[^"]*"\s+"([^"]*)"\)\s+(\d+)x(\d+)')


def find_window(display: str, wm_class: str) -> Optional[str]:
    """The X window id whose class is wm_class (set by --class on launch), skipping helper windows."""
    for line in _x(display, "xwininfo", "-root", "-tree").splitlines():
        m = _TREE_LINE.match(line)
        if m and m.group(2) == wm_class and int(m.group(3)) > 100:
            return m.group(1)
    return None


def window_pid(display: str, wid: str) -> Optional[int]:
    """The pid that owns a window (_NET_WM_PID), or None when the property is missing."""
    m = re.search(r"_NET_WM_PID\(CARDINAL\) = (\d+)", _x(display, "xprop", "-id", wid, "_NET_WM_PID") or "")
    return int(m.group(1)) if m else None


def window_geometry(display: str, wid: str) -> Optional[Geometry]:
    """Absolute client-area geometry of a window (the page, without any frame)."""
    out = _x(display, "xwininfo", "-id", wid)
    vals = {}
    for key, pat in (("x", r"Absolute upper-left X:\s+(-?\d+)"), ("y", r"Absolute upper-left Y:\s+(-?\d+)"),
                     ("w", r"Width:\s+(\d+)"), ("h", r"Height:\s+(\d+)")):
        m = re.search(pat, out)
        if not m:
            return None
        vals[key] = int(m.group(1))
    return Geometry(vals["x"], vals["y"], vals["w"], vals["h"])


def set_client_geometry(display: str, wid: str, x: int, y: int, width: int, height: int) -> bool:
    """Move and resize the window's client area with python-xlib (mutter re-sizes a window whose
    decorations were stripped after mapping: 1280x720 became 1280x757). False when xlib is missing."""
    try:
        from Xlib import display as xdisplay
    except ImportError:
        logger.warning("python-xlib not installed: cannot resize the recording window")
        return False
    try:
        d = xdisplay.Display(display)
        win = d.create_resource_object("window", int(wid, 16))
        win.configure(x=x, y=y, width=width, height=height)
        d.sync()
        d.close()
        return True
    except Exception as e:  # noqa: BLE001 - the window may have gone
        logger.warning("Could not set the recording window geometry: %s", e)
        return False


def request_fullscreen(display: str, wid: str) -> bool:
    """Ask the window manager (EWMH _NET_WM_STATE) to make the window fullscreen: the whole
    display, over the panels and dock. Chromium's --kiosk asks for the same, but the request is
    sometimes lost when a window is mapped right after another was killed; this repeats it."""
    try:
        from Xlib import X, display as xdisplay, protocol
    except ImportError:
        logger.warning("python-xlib not installed: cannot request fullscreen")
        return False
    try:
        d = xdisplay.Display(display)
        win = d.create_resource_object("window", int(wid, 16))
        state = d.intern_atom("_NET_WM_STATE")
        full = d.intern_atom("_NET_WM_STATE_FULLSCREEN")
        ev = protocol.event.ClientMessage(window=win, client_type=state, data=(32, [1, full, 0, 1, 0]))   # 1 = _NET_WM_STATE_ADD
        d.screen().root.send_event(ev, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
        d.sync()
        d.close()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not request fullscreen for the recording window: %s", e)
        return False


def wait_for_geometry(display: str, wid: str, width: int, height: int, timeout: float = 5.0) -> Optional[Geometry]:
    """The window's geometry once it is width x height, or the last seen geometry after timeout."""
    deadline = time.time() + timeout
    geo = None
    while time.time() < deadline:
        geo = window_geometry(display, wid)
        if geo and geo.width == width and geo.height == height:
            return geo
        time.sleep(0.25)
    return geo


def strip_decorations(display: str, wid: str) -> bool:
    """Ask the window manager to draw no title bar or border (Motif hints; mutter honours them)."""
    try:
        subprocess.run(["xprop", "-id", wid, "-f", "_MOTIF_WM_HINTS", "32c", "-set", "_MOTIF_WM_HINTS", "2, 0, 0, 0, 0"],
                       capture_output=True, text=True, timeout=5, env=dict(os.environ, DISPLAY=display))
        return True
    except (OSError, subprocess.SubprocessError) as e:
        logger.warning("Could not strip window decorations: %s", e)
        return False


@dataclass
class BrowserWindow:
    proc: Optional[subprocess.Popen]
    wid: Optional[str]
    geometry: Optional[Geometry]
    command: List[str] = field(default_factory=list)


def _main_processes(wm_class: str, helpers: bool = False) -> List[int]:
    """Pids of the Chromium browser processes started with --class=<wm_class>; with ``helpers`` also
    their renderer, GPU and zygote processes (they carry the parent's switches, and a GPU process
    left behind keeps the X windows open). Chromium rewrites its argv into one space-joined
    string, so the command line is matched as text, not as NUL-separated arguments."""
    flag = re.compile(rb"(?:^|\s)--class=" + re.escape(wm_class.encode()) + rb"(?:\s|$)")
    pids = []
    for pid_dir in Path("/proc").glob("[0-9]*"):
        try:
            text = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").strip()
        except OSError:
            continue
        if text and flag.search(text) and (helpers or not re.search(rb"\s--type=", text)):
            pids.append(int(pid_dir.name))
    return pids


def close_recording_browsers(wm_class: str = "handoff-record", wait_s: float = 5.0, profile_dir: Optional[Path] = None) -> int:
    """End every Chromium started with --class=<wm_class>, so the next recording gets a fresh page.
    (A page left open across a server restart keeps running the old script, and a new launch on
    the same profile would only hand its URL to the old instance.) SIGTERM first; whatever is
    still alive after wait_s gets SIGKILL. The profile's singleton lock is removed afterwards."""
    pids = _main_processes(wm_class)
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + wait_s
    while pids and time.time() < deadline and any(_alive(p) for p in pids):
        time.sleep(0.2)
    for pid in pids:
        if _alive(pid):
            logger.warning("Recording browser %d ignored SIGTERM for %.0f s; killing it and its helpers", pid, wait_s)
    leftovers = [p for p in _main_processes(wm_class, helpers=True) if _alive(p)]   # a helper alone keeps the windows
    for pid in leftovers:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    for pid in pids + leftovers:
        for _ in range(25):
            if not _alive(pid):
                break
            time.sleep(0.2)
    if profile_dir:
        for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            try:
                (profile_dir / name).unlink()
            except OSError:
                pass
    return len(pids)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    try:
        return "Z" not in (Path(f"/proc/{pid}/stat").read_text().split(")")[-1].split()[0])
    except OSError:
        return False


def chromium_command(exe: str, url: str, width: int, height: int, profile_dir: Path, wm_class: str = "handoff-record",
                     scale: float = 1.0, kiosk: bool = False) -> List[str]:
    """Kiosk: a fullscreen window over the panels and dock, the whole display (for a recording at
    the display's size, which the window manager would otherwise shrink to its work area)."""
    return [exe, "--no-sandbox", f"--app={url}", f"--window-size={width},{height}", "--window-position=0,0",
            "--no-first-run", "--no-default-browser-check", "--disable-infobars", "--disable-session-crashed-bubble",
            "--hide-scrollbars", f"--force-device-scale-factor={scale:g}", f"--user-data-dir={profile_dir}",
            f"--class={wm_class}", "--autoplay-policy=no-user-gesture-required"] + (["--kiosk"] if kiosk else [])


def launch_recording_browser(url: str, display: str = ":1", width: int = 1280, height: int = 720,
                             profile_dir: Optional[Path] = None, chromium: str = "",
                             wm_class: str = "handoff-record", wait_s: float = 8.0, fresh: bool = True,
                             scale: float = 1.0, kiosk: bool = False) -> BrowserWindow:
    """Open the page in a Chromium app window of exactly width x height at the origin, strip the
    window manager's decorations, and return the window's client geometry for the grab. ``scale``
    is the page zoom: the layout is laid out at width/scale x height/scale CSS pixels.

    With ``fresh`` (the default) any earlier window of the class is closed first, so the page and
    its script are the ones the running server serves; otherwise an open window is reused."""
    profile_dir = profile_dir or Path.home() / ".cache" / "handoff-ai" / "record-profile"
    wid = find_window(display, wm_class)
    if fresh:
        if close_recording_browsers(wm_class, profile_dir=profile_dir):
            logger.info("Closed the previous recording browser")
        wid = None
    if wid:
        geo = window_geometry(display, wid)
        logger.info("Recording window already open: %s at %s", wid, geo)
        return BrowserWindow(None, wid, geo)
    exe = find_chromium(chromium)
    if not exe:
        raise RuntimeError("No Chromium found for the recording window (install Playwright's chromium or set HANDOFF_CHROMIUM)")
    profile_dir.mkdir(parents=True, exist_ok=True)
    cmd = chromium_command(exe, url, width, height, profile_dir, wm_class, scale, kiosk)
    env = dict(os.environ, DISPLAY=display)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    deadline = time.time() + wait_s
    wid = None
    while time.time() < deadline and wid is None:
        time.sleep(0.5)
        wid = find_window(display, wm_class)
        if wid and (owner := window_pid(display, wid)) not in (None, proc.pid):
            logger.warning("Recording window %s belongs to pid %s, not the new browser %d", wid, owner, proc.pid)
    geo = None
    if wid and kiosk:
        geo = wait_for_geometry(display, wid, width, height, timeout=3.0)   # Chromium's own fullscreen request
        if geo and (geo.width != width or geo.height != height):
            logger.info("Recording window is %s after kiosk launch; asking the window manager for fullscreen", geo.size)
            request_fullscreen(display, wid)
            geo = wait_for_geometry(display, wid, width, height, timeout=5.0)
    elif wid:
        strip_decorations(display, wid)
        time.sleep(0.8)
        geo = window_geometry(display, wid)
        if geo and (geo.width != width or geo.height != height):
            set_client_geometry(display, wid, geo.x, geo.y, width, height)
            time.sleep(0.5)
            geo = window_geometry(display, wid)
        if geo and (geo.width != width or geo.height != height):
            logger.warning("Recording window is %s, wanted %dx%d", geo.size, width, height)
    logger.info("Recording window: %s at %s", wid, geo)
    return BrowserWindow(proc, wid, geo, cmd)


def ffmpeg_command(out: Path, geometry: Geometry, display: str = ":1", sink: str = "handoff_replay",
                   fps: int = 15, ffmpeg: str = "ffmpeg") -> List[str]:
    """One ffmpeg: the window region and the sink monitor, wall-clock timestamps on both inputs,
    frames passed through (a dropped frame leaves a gap, never a shorter video), lightest preset."""
    size = f"{geometry.width - geometry.width % 2}x{geometry.height - geometry.height % 2}"   # yuv420p needs even sides
    return [ffmpeg, "-y", "-hide_banner", "-loglevel", "warning",
            "-use_wallclock_as_timestamps", "1", "-f", "x11grab", "-framerate", str(fps),
            "-video_size", size, "-draw_mouse", "0", "-i", f"{display}.0{geometry.x11grab_input}",
            "-use_wallclock_as_timestamps", "1", "-f", "pulse", "-i", f"{sink}.monitor",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-crf", "23", "-pix_fmt", "yuv420p",
            "-g", str(fps * 2), "-vsync", "passthrough",
            "-c:a", "aac", "-b:a", "96k", "-ar", "48000", "-ac", "1",
            "-movflags", "+faststart", str(out)]


class Recorder:
    """Runs the ffmpeg capture for one replay. start() returns once ffmpeg has been up for warm_up_s."""

    def __init__(self, out: Path, geometry: Geometry, display: str = ":1", sink: str = "handoff_replay",
                 fps: int = 15, ffmpeg: str = "ffmpeg", runtime_dir: Optional[str] = None):
        self.out = Path(out)
        self.command = ffmpeg_command(self.out, geometry, display, sink, fps, ffmpeg)
        self.proc: Optional[subprocess.Popen] = None
        self.started_wall: Optional[float] = None
        self.stopped_wall: Optional[float] = None
        self.error: Optional[str] = None
        self._runtime_dir = runtime_dir
        self.log_path = self.out.with_suffix(".ffmpeg.log")

    def start(self, warm_up_s: float = 1.5) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        if self._runtime_dir:
            env["XDG_RUNTIME_DIR"] = self._runtime_dir
        env.setdefault("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        with open(self.log_path, "w") as log:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=log, env=env)
        self.started_wall = time.time()
        time.sleep(warm_up_s)                   # the first frames and the audio stream need a moment
        if self.proc.poll() is not None:
            self.error = f"ffmpeg exited with {self.proc.returncode}: {self.log_path.read_text()[-400:]}"
            raise RuntimeError(self.error)
        logger.info("Recording to %s", self.out)

    def stop(self, tail_s: float = 1.0) -> Dict[str, object]:
        if self.proc is None:
            return {}
        time.sleep(tail_s)
        try:
            self.proc.stdin.write(b"q")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.terminate()
            self.proc.wait(timeout=5)
        self.stopped_wall = time.time()
        info: Dict[str, object] = {"path": str(self.out), "bytes": self.out.stat().st_size if self.out.exists() else 0,
                                   "wall_s": round(self.stopped_wall - (self.started_wall or self.stopped_wall), 1),
                                   "ffmpeg_exit": self.proc.returncode, "duration_s": media_duration_s(self.out)}
        return info


def media_duration_s(path: Path) -> Optional[float]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return round(float(out), 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


INDEX_HEADER = "| call-set | provider | model | commit | date | duration | sync | verdict |\n|---|---|---|---|---|---|---|---|\n"


def append_index_line(record_dir: Path, callset: str, provider: str, model: str, commit: str,
                      duration_s: Optional[float], sync: Optional[Dict[str, object]], path: Path) -> str:
    """One line per recorded call-set in <record_dir>/INDEX.md; the verdict column is for a person."""
    index = Path(record_dir) / "INDEX.md"
    if not index.exists():
        index.write_text("# Replay recordings\n\n" + INDEX_HEADER)
    dur = f"{int(duration_s // 60)}:{int(duration_s % 60):02d}" if duration_s else "?"
    if sync is None:
        sync_text = "not checked"
    else:
        worst = sync.get("max_abs_offset_ms")
        sync_text = f"{'pass' if sync.get('pass') else 'FAIL'} ({sync.get('flashes')}/{sync.get('expected_bursts')} bursts" + \
            (f", worst {worst:.0f} ms" if worst is not None else "") + ")"
    rel = Path(path).relative_to(record_dir) if str(path).startswith(str(record_dir)) else path
    line = f"| {callset} | {provider} | {model} | {commit} | {time.strftime('%Y-%m-%d %H:%M')} | {dur} | {sync_text} | pending | `{rel}` |"
    with open(index, "a") as f:
        f.write(line + "\n")
    return line
