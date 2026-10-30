"""
server.py — the local web server behind the HANDOFF.AI app window.

Purpose
    Connect the screen (web/) to the engines. The browser never touches audio
    or AI directly. It calls this API and listens to one event stream.

HTTP API (all JSON; bound to 127.0.0.1 by default)
    GET  /                  the app (web/index.html)
    GET  /api/status        provider, model, engines, mic, current state
    GET  /api/mics          input devices on this machine
    GET  /api/samples       practice scripts and sample cases
    POST /api/settings      {"provider": "groq|ollama|openrouter", "mic_device": "..."}
    POST /api/check         is the selected AI provider reachable? -> {"ok", "message"}
    POST /api/live/start    open the mic and start listening
    POST /api/live/done     finish now and make the report (the DONE button)
    POST /api/live/cancel   discard the recording
    POST /api/report/text   {"text": "...", "source": "..."} -> make a report from text
    GET  /api/result        the last finished report (for a page reload)
    GET  /api/events        Server-Sent Events: state, caption, level, stage,
                            transcript, result, error
    POST /api/quit          stop the app (used by the Quit button)

States
    idle -> recording -> transcribing -> processing -> done
                                     \\-> error (from any step; the app returns to idle)

Threads
    Flask request threads (threaded server), the session's audio threads, and
    ONE "report-worker" thread that does transcription and the pipeline, so two
    reports can never run at once.

Testing
    ``create_app(settings, engines=Engines(...))`` lets tests swap in a fake
    mic, fake recognizers, a fake transcriber, and a fake LLM.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.io.wavfile as wavfile
from flask import Flask, Response, jsonify, request, send_from_directory

from . import __version__
from .audio.capture import AudioCapture, MicError, list_input_devices
from .audio.listeners import make_listeners
from .audio.session import LiveSession
from .audio.sources import FileAudioSource
from .incident import IncidentManager
from .recording import Geometry, Recorder, append_index_line, launch_recording_browser
from .replay import (BurstClip, CallSet, ReplayDriver, ReplayHost, SinkPlayer, ensure_null_sink, list_callsets,
                     load_callset)
from . import sync_check
from .config import PROJECT_ROOT, PROVIDERS, Settings
from .llm import LLMClient, LLMError
from .pipeline import PipelineError, ReportPipeline
from .samples import SAMPLE_SEPSIS, SAMPLE_STEMI, SAMPLE_TRAUMA
from .transcribe import LocalTranscriber, make_transcriber
from .transcript_checks import transcribe_checked

logger = logging.getLogger(__name__)

APP_ID = "handoff-app"   # /api/status says this, so the launcher can tell this app from the old one

WEB_DIR = Path(__file__).resolve().parent / "web"
USER_SETTINGS_FILE = PROJECT_ROOT / "user_settings.json"

QUICK_DEMO = """Medic 21 to General Hospital.
We are inbound with a 45-year-old male found down.
GCS is 14. History of seizures.
Vitals: BP 140 over 90, heart rate 110, SpO2 98 percent.
We have established an IV and given 5 milligrams of Versed.
ETA is 5 minutes."""

SAMPLES = [
    {"id": "quick", "title": "Quick demo (short)", "text": QUICK_DEMO},
    {"id": "stemi", "title": "STEMI: 67-year-old male, chest pain", "text": SAMPLE_STEMI.strip()},
    {"id": "trauma", "title": "Trauma: 34-year-old female, rollover MVC", "text": SAMPLE_TRAUMA.strip()},
    {"id": "sepsis", "title": "Sepsis: 82-year-old female, nursing home", "text": SAMPLE_SEPSIS.strip()},
]


# ---------------------------------------------------------------------------
# Engines (swappable for tests)
# ---------------------------------------------------------------------------
@dataclass
class Engines:
    make_llm: Callable[[Settings], Any] = LLMClient
    make_transcriber: Callable[[Settings], Any] = make_transcriber
    make_fallback_transcriber: Optional[Callable[[Settings], Any]] = LocalTranscriber
    make_capture: Callable[[Settings], AudioCapture] = lambda s: AudioCapture(s.mic_device, highpass_hz=s.mic_highpass_hz)
    make_listeners: Callable[[Settings], tuple] = make_listeners
    list_mics: Callable[[], List] = list_input_devices
    make_sink: Callable[[Settings], Any] = lambda s: SinkPlayer(s.replay_sink)   # replay audio out (null sink)
    # recording a replay: the browser window, the ffmpeg capture, the sync check (all swappable in tests)
    make_browser: Callable[[Settings, str], Any] = lambda s, url: launch_recording_browser(
        url, s.replay_display, s.replay_record_width, s.replay_record_height, chromium=s.replay_chromium,
        scale=s.replay_record_scale, kiosk=s.replay_record_kiosk)
    make_recorder: Callable[[Settings, Path, Any], Any] = lambda s, out, geo: Recorder(
        out, geo, s.replay_display, s.replay_sink, s.replay_record_fps)
    check_sync: Callable[[Path, int], dict] = lambda path, expected: sync_check.check(path, expected)
    lexicon_store: Any = None
    textbook_rag: Any = None


def default_engines(settings: Settings) -> Engines:
    lexicon = None
    textbook = None
    if settings.use_lexicon:
        from .lexicon_rag import LexiconStore
        lexicon = LexiconStore()
    if settings.use_textbook_rag:
        from .textbook_rag import TextbookRAG
        textbook = TextbookRAG(store_dir=str(settings.textbook_dir))
    return Engines(lexicon_store=lexicon, textbook_rag=textbook)


def app_commit() -> str:
    """Short git commit of this checkout, or "unknown" outside a repo."""
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


APP_COMMIT = app_commit()


# ---------------------------------------------------------------------------
# Event fan-out for /api/events
# ---------------------------------------------------------------------------
class EventBroker:
    def __init__(self):
        self._subs: List[queue.Queue] = []
        self._lock = threading.Lock()

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self, event: Dict[str, Any]) -> None:
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:  # a stalled browser tab must not block the app
                pass


# ---------------------------------------------------------------------------
# Controller: owns the state machine and the report worker
# ---------------------------------------------------------------------------
class Controller:
    def __init__(self, settings: Settings, engines: Engines, broker: EventBroker):
        self.settings = settings
        self.engines = engines
        self.broker = broker
        self.state = "idle"
        self.session: Optional[LiveSession] = None
        self.last_result: Optional[dict] = None
        self.last_error: Optional[str] = None
        self.captions_label = "not started"
        self._listeners_cache: Optional[tuple] = None
        self._transcribers: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._last_transcription: dict = {}     # transcript_checks log of the last live recording
        self._jobs: queue.Queue = queue.Queue()
        self._worker = threading.Thread(target=self._work, name="report-worker", daemon=True)
        self._worker.start()
        self._replay: Optional["ReplayRun"] = None   # only ever set when settings.replay_mode_enabled

    # -- state ----------------------------------------------------------------
    def _set_state(self, state: str, **extra) -> None:
        self.state = state
        self.broker.publish({"type": "state", "state": state, **extra})

    def _transcriber(self, role: str = "primary"):
        """Cached transcriber instances, so a local Whisper model is loaded only once."""
        key = f"{role}:{self.settings.effective_transcriber()}"
        with self._lock:
            if key not in self._transcribers:
                factory = (self.engines.make_transcriber if role == "primary"
                           else self.engines.make_fallback_transcriber)
                self._transcribers[key] = factory(self.settings) if factory else None
            return self._transcribers[key]

    def warm_up(self) -> None:
        """Load the caption engine (and local Whisper, if it is the primary) before the first recording."""
        try:
            with self._lock:
                if self._listeners_cache is None:
                    self._listeners_cache = self.engines.make_listeners(self.settings)
                self.captions_label = self._listeners_cache[2]
            primary = self._transcriber("primary")
            if getattr(primary, "name", "") == "local":
                primary.warm_up()
            logger.info("Engines ready: captions=%s, transcript=%s", self.captions_label,
                        primary.describe())
        except Exception as e:  # noqa: BLE001 - the app still works; engines load on first use
            logger.warning("Warm-up incomplete: %s", e)
        if self.settings.provider == "ollama":
            try:   # load the local model now, not during the first report
                self.engines.make_llm(self.settings).preload()
            except Exception as e:  # noqa: BLE001 - Ollama down or model missing: Check says why
                logger.warning("Ollama preload skipped: %s", e)
        self.broker.publish({"type": "status", "status": self.status()})

    def status(self) -> dict:
        s = self.settings
        try:
            transcriber = self._transcriber("primary").describe()
        except Exception as e:  # noqa: BLE001
            transcriber = f"unavailable: {e}"
        out = {"app": APP_ID, "version": __version__, "state": self.state, "settings": s.public_dict(),
               "captions": self.captions_label, "transcriber": transcriber,
               "has_result": self.last_result is not None, "error": self.last_error}
        if s.replay_mode_enabled:
            out["commit"] = APP_COMMIT
            out["replay"] = self._replay.status() if self._replay else None
        return out

    def ollama_models(self) -> dict:
        """Pulled Ollama models with memory needs, for the Settings screen."""
        llm = self.engines.make_llm(dataclasses.replace(self.settings, provider="ollama"))
        inv = llm.ollama_inventory()
        return {**inv, "selected": self.settings.ollama_model, "pinned": self.settings.ollama_model_pinned}

    def update_settings(self, provider: Optional[str] = None, mic_device: Optional[str] = None,
                        ollama_model: Optional[str] = None) -> None:
        changes = {}
        if provider is not None:
            if provider not in PROVIDERS:
                raise ValueError(f"Unknown provider '{provider}'")
            changes["provider"] = provider
        if mic_device is not None:
            changes["mic_device"] = str(mic_device).strip()
        if ollama_model is not None:
            name = str(ollama_model).strip()
            if name:
                # a chosen model is pinned: it must be pulled and fit, or the choice is refused
                llm = self.engines.make_llm(dataclasses.replace(self.settings, provider="ollama"))
                good, why = llm.ollama_model_status(name)
                if not good:
                    raise ValueError(f"Ollama model not saved: {why}")
                changes["ollama_model"] = name
                changes["ollama_model_pinned"] = True
        with self._lock:
            self.settings = dataclasses.replace(self.settings, **changes)
        save_user_settings(self.settings)
        self.broker.publish({"type": "status", "status": self.status()})

    # -- live recording -------------------------------------------------------
    def start_live(self) -> None:
        with self._lock:
            if self.state not in ("idle", "done", "error"):
                raise RuntimeError(f"Busy ({self.state}). Finish or cancel first.")
            if self._listeners_cache is None:
                self._listeners_cache = self.engines.make_listeners(self.settings)
            cap_f, spot_f, label = self._listeners_cache
            self.captions_label = label
            capture = self.engines.make_capture(self.settings)
            session = LiveSession(capture, cap_f, spot_f, emit=self.broker.publish,
                                  on_finish=self._on_session_finish,
                                  max_seconds=self.settings.max_recording_s, captions_label=label)
            self.session = session
            self.last_error = None
            try:
                session.start()
            except MicError as e:
                self.session = None
                self._fail(str(e))
                raise
            self._set_state("recording")

    def done(self) -> None:
        with self._lock:
            session = self.session
            if session is None or self.state != "recording":
                raise RuntimeError("Not recording.")
        session.finish("button")

    def cancel(self) -> None:
        with self._lock:
            session, self.session = self.session, None
        if session is not None:
            session.finish("cancel")
            session.stop()
        self._set_state("idle")

    def _on_session_finish(self, reason: str) -> None:
        # runs on an audio/listener thread: hand off and return immediately
        if reason == "cancel":
            return
        with self._lock:
            session = self.session
            self.session = None
        if session is not None:
            self.broker.publish({"type": "finished", "reason": reason})
            self._set_state("transcribing", reason=reason)
            self._jobs.put(("audio", session))

    # -- replay (recorded call-sets; reachable only when settings.replay_mode_enabled) ---------
    def replay_callsets(self) -> List[dict]:
        s = self.settings
        return list_callsets(s.replay_callsets_dir, Path(s.replay_index).expanduser() if s.replay_index else None)

    def start_replay(self, callset_id: str, provider: Optional[str] = None, speed: Optional[float] = None,
                     record: bool = False) -> dict:
        s = self.settings
        provider = (provider or s.replay_provider).strip().lower()
        if provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'")
        if not s.has_key(provider):
            raise ValueError(f"No API key set for {provider}.")
        folder = Path(s.replay_callsets_dir).expanduser() / str(callset_id)
        callset = load_callset(folder, Path(s.replay_index).expanduser() if s.replay_index else None)
        if not callset.bursts:
            raise ValueError(f"{callset.id} has no bursts")
        with self._lock:
            if self.state not in ("idle", "done", "error"):
                raise RuntimeError(f"Busy ({self.state}). Finish or cancel first.")
            if self._listeners_cache is None:
                self._listeners_cache = self.engines.make_listeners(self.settings)
            self.captions_label = self._listeners_cache[2]
            run = ReplayRun(self, callset, provider, float(speed) if speed else s.replay_speed, record=record)
            self._replay = run
            self.last_error = None
            self._set_state("replaying", callset=callset.id)
            try:
                run.start()
            except Exception as e:  # noqa: BLE001 - the window or the recorder did not come up: back to idle
                logger.exception("Replay could not start")
                run.stop()
                self.last_error = f"Replay could not start: {e}"
                self._set_state("idle")
                raise RuntimeError(self.last_error) from e
            return run.status()

    def stop_replay(self) -> None:
        with self._lock:
            run = self._replay
        if run is not None:
            run.stop()
        if self.state == "replaying":
            self._set_state("idle")

    def replay_run(self) -> "ReplayRun":
        if self._replay is None:
            raise RuntimeError("No replay has been started.")
        return self._replay

    def _replay_ended(self, run: "ReplayRun") -> None:
        """The driver reached done or error: the final report stays as the last result."""
        session = run.manager.sessions[-1] if run.manager.sessions else None
        if session is not None and session.current is not None:
            self.last_result = session.current
        if run.driver.error:
            self.last_error = run.driver.error
        if self.state == "replaying":
            self._set_state("idle", replay=run.driver.phase)

    # -- text reports -----------------------------------------------------------
    def submit_text(self, text: str, source: str = "text") -> None:
        with self._lock:
            if self.state in ("recording", "transcribing", "processing"):
                raise RuntimeError(f"Busy ({self.state}). Wait for the current report.")
            self.last_error = None
            self._set_state("processing", source=source)
        self._jobs.put(("text", text, source))

    # -- worker ---------------------------------------------------------------
    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job[0] == "audio":
                    self._process_audio(job[1])
                else:
                    self._make_report(job[1], source=job[2], timings={})
            except Exception as e:  # noqa: BLE001 - one bad job must not kill the worker
                logger.exception("Report job failed")
                self._fail(f"Unexpected error: {type(e).__name__}: {e}")

    def _process_audio(self, session: LiveSession) -> None:
        audio = session.stop()
        seconds = len(audio) / 16000
        save_wav(self.settings.output_dir / "recording.wav", audio)
        if seconds < 0.5:
            return self._fail("Nothing was recorded. Check the microphone and try again.")
        self._stage(f"Transcribing {seconds:.0f} s of audio")
        timings = {"audio_s": round(seconds, 1)}
        t0 = time.perf_counter()
        transcript = self._transcribe(audio, session.final_text)
        timings["transcribe_s"] = round(time.perf_counter() - t0, 2)
        if transcript is None:
            return
        self.broker.publish({"type": "transcript", "text": transcript})
        self._make_report(transcript, source="live", timings=timings,
                          meta_extra={"transcription": self._last_transcription})

    def _transcriber_pair(self):
        """(primary, fallback) for transcribe_checked; either may be None."""
        primary = None
        try:
            primary = self._transcriber("primary")
        except Exception as e:  # noqa: BLE001
            logger.warning("Primary transcriber unavailable: %s", e)
        fallback = None
        if getattr(primary, "name", "") != "local":
            fallback = self._transcriber("fallback")
        return primary, fallback

    def _transcribe(self, audio: np.ndarray, caption_text: str) -> Optional[str]:
        """Primary transcriber, then the local one, then the live captions as a last resort,
        with the silence gate and the words-per-second check (transcript_checks.py)."""
        primary, fallback = self._transcriber_pair()
        result = transcribe_checked(primary, fallback, audio, caption_text, on_stage=self._stage, caption_min_words=5)
        self._last_transcription = result.meta()
        if result.no_speech:
            self._fail("No speech detected: the microphone picked up nothing above the noise floor for the "
                       "whole recording. Check that the right mic is selected and try again.")
            return None
        if not result.ok:
            self._fail("Could not transcribe the recording. " + " ".join(result.notes))
            return None
        return result.text

    def _make_report(self, text: str, source: str, timings: Dict[str, float],
                     meta_extra: Optional[dict] = None) -> None:
        self._set_state("processing", source=source)
        try:
            llm = self.engines.make_llm(self.settings)
        except LLMError as e:
            return self._fail(str(e))
        pipeline = ReportPipeline(llm, stop_phrase=self.settings.stop_phrase,
                                  lexicon_store=self.engines.lexicon_store,
                                  textbook_rag=self.engines.textbook_rag, on_stage=self._stage)
        try:
            report = pipeline.run(text, source=source, timings=timings)
        except PipelineError as e:
            return self._fail(str(e))
        if meta_extra:
            report["_meta"].update(meta_extra)
        save_report(self.settings.output_dir, report)
        self.last_result = report
        self.broker.publish({"type": "result", "report": report})
        self._set_state("done")

    def _stage(self, message: str) -> None:
        logger.info("Stage: %s", message)
        self.broker.publish({"type": "stage", "text": message})
        if message.startswith("Model server restarting"):
            self._set_state("processing", detail="model server restarting")

    def _fail(self, message: str) -> None:
        logger.error("Report failed: %s", message)
        self.last_error = message
        self.broker.publish({"type": "error", "message": message})
        self._set_state("error")


# ---------------------------------------------------------------------------
# One replay: the driver, its session, and the transcription of each burst
# ---------------------------------------------------------------------------
class ReplayRun(ReplayHost):
    """Plays one call-set. Owns the incident session, a transcription thread and a poll thread.

    Nothing here runs unless start_replay was called, which only the replay endpoints do,
    and those are registered only when settings.replay_mode_enabled is on."""

    def __init__(self, ctl: Controller, callset: CallSet, provider: str, speed: float, record: bool = False):
        self.ctl = ctl
        self.callset = callset
        self.provider = provider
        s = ctl.settings
        self.record = record
        self.recording: Optional[dict] = None          # path, bytes, duration, sync (when record=True)
        self._recorder = None
        self._window = None
        extra = {"groq_llm_model": s.replay_groq_model} if provider == "groq" else {}
        self.settings = dataclasses.replace(s, provider=provider, **extra)
        self.llm = ctl.engines.make_llm(self.settings)        # raises LLMError when the key is missing
        self.model = self.llm.model
        self.fallback: Optional[dict] = None
        self.started = time.time()
        self.transcriptions: List[dict] = []                 # one per burst: who transcribed it, how long
        quiet = s.replay_quiet_offline_s if provider == "ollama" else s.replay_quiet_online_s
        close_quiet = getattr(s, "close_quiet_s", None) if getattr(s, "writer_mode", "full") != "full" else None
        self.manager = IncidentManager(self._extract, quiet_s=quiet, clock=time.time, emit=ctl.broker.publish, close_quiet_s=close_quiet)
        self.sink = ctl.engines.make_sink(s)
        self.driver = ReplayDriver(callset, host=self, sink=self.sink, speed=speed, emit=self._emit,
                                   work_dir=s.output_dir / "replay", sink_lead_s=s.replay_sink_lead_ms / 1000.0,
                                   before_first_burst=self._start_recording if record else None)
        self.session: Optional[LiveSession] = None
        self._jobs: queue.Queue = queue.Queue()
        self._busy = threading.Event()
        self._stopping = threading.Event()
        self._threads: List[threading.Thread] = []

    # -- lifecycle ------------------------------------------------------------
    def start(self) -> None:
        if self.sink is not None and getattr(self.sink, "available", False):
            if not ensure_null_sink(self.sink.sink):
                logger.warning("Replay sink %s is not available; audio will not reach a recording", self.sink.sink)
        if self.record:
            self._prepare_recording()
        for name, target in (("replay-transcribe", self._transcribe_loop), ("replay-poll", self._poll_loop)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)
        self.driver.start()

    def stop(self) -> None:
        self._stopping.set()
        self.driver.stop()
        session = self.session
        if session is not None and not session.is_finished:
            session.finish("cancel")
            session.stop()
        self._jobs.put(None)
        self.manager.stop()
        if self.record:
            self._finish_recording()

    def status(self) -> dict:
        sessions = self.manager.sessions
        current = sessions[-1] if sessions else None
        return {"callset": self.callset.id, "callset_info": self.callset.summary(), "provider": self.provider,
                "model": self.model, "commit": APP_COMMIT, "started": self.started,
                "fallback": self.fallback, "driver": self.driver.status(),
                "session": current.snapshot() if current else None, "sessions": [x.id for x in sessions],
                "transcriptions": list(self.transcriptions), "recording": self.recording}

    # -- recording (record=True): the window, ffmpeg around the bursts, then the sync check ----
    def _prepare_recording(self) -> None:
        s = self.ctl.settings
        url = f"http://127.0.0.1:{s.port}/"
        self._window = self.ctl.engines.make_browser(s, url)
        geo = getattr(self._window, "geometry", None)
        if geo is None:
            raise RuntimeError("The recording window did not open; nothing to capture")
        folder = Path(s.replay_record_dir) / f"{time.strftime('%Y-%m-%d')}_{APP_COMMIT}"
        folder.mkdir(parents=True, exist_ok=True)
        # the provider is part of the name: a call-set recorded on both lands twice in the same folder
        out = folder / f"{self.callset.id}.{self.provider}.mp4"
        self._recorder = self.ctl.engines.make_recorder(s, out, geo)
        self.recording = {"path": str(out), "folder": str(folder), "geometry": [geo.x, geo.y, geo.width, geo.height],
                          "display": s.replay_display, "fps": s.replay_record_fps, "scale": s.replay_record_scale,
                          "state": "prepared", "sink_lead_ms": s.replay_sink_lead_ms, "tail_s": s.replay_record_tail_s}

    def _start_recording(self) -> None:
        """Runs on the driver thread just before burst 1."""
        if self._recorder is None:
            return
        self._stage("Recording started")
        self._recorder.start()
        self.recording["state"] = "recording"
        self.recording["started_wall"] = self._recorder.started_wall
        self.ctl.broker.publish({"type": "replay", "event": "recording", **self.driver.status(), "provider": self.provider,
                                 "model": self.model, "commit": APP_COMMIT, "fallback": self.fallback, "recording": self.recording})

    def _finish_recording(self) -> None:
        """After done or error: stop ffmpeg, write the summary, run the sync check, add the index line."""
        if self._recorder is None or self.recording is None or self.recording.get("state") in ("done", "failed"):
            return
        try:
            info = self._recorder.stop(tail_s=self.ctl.settings.replay_record_tail_s)
            self.recording.update(info, state="done")
            folder = Path(self.recording["folder"])
            path = Path(self.recording["path"])
            stem = path.name[:-len(".mp4")]
            summary = folder / f"{stem}.summary.json"
            summary.write_text(json.dumps(self.status(), indent=1))
            self._stage("Checking audio/video sync")
            sync = self.ctl.engines.check_sync(path, len(self.callset.bursts))
            (folder / f"{stem}.sync.json").write_text(json.dumps(sync, indent=1))
            self.recording["sync"] = sync
            self.recording["index_line"] = append_index_line(Path(self.ctl.settings.replay_record_dir), self.callset.id,
                                                             self.provider, self.model, APP_COMMIT,
                                                             info.get("duration_s"), sync, path)
            summary.write_text(json.dumps(self.status(), indent=1))
            logger.info("Recording finished: %s (%s bytes, %s s), sync %s", path, info.get("bytes"), info.get("duration_s"),
                        "pass" if sync.get("pass") else "FAIL")
        except Exception as e:  # noqa: BLE001 - the replay itself is done; say what went wrong with the file
            logger.exception("Recording failed")
            self.recording.update(state="failed", error=f"{type(e).__name__}: {e}")
        self.ctl.broker.publish({"type": "replay", "event": "recorded", **self.driver.status(), "provider": self.provider,
                                 "model": self.model, "commit": APP_COMMIT, "fallback": self.fallback, "recording": self.recording})

    # -- events --------------------------------------------------------------
    def _emit(self, event: dict) -> None:
        event.update(provider=self.provider, model=self.model, commit=APP_COMMIT, fallback=self.fallback)
        if self.recording is not None:
            event["recording"] = self.recording
        self.ctl.broker.publish(event)
        if event.get("phase") in ("done", "error"):
            self.ctl._replay_ended(self)
            if self.record:
                threading.Thread(target=self._finish_recording, name="replay-record-finish", daemon=True).start()

    def _stage(self, message: str) -> None:
        logger.info("Replay stage: %s", message)
        self.ctl.broker.publish({"type": "stage", "text": message, "replay": True})

    # -- ReplayHost ------------------------------------------------------------
    def start_burst(self, clip: BurstClip, number: int) -> threading.Event:
        cap_f, _spot_f, label = self.ctl._listeners_cache
        ended = threading.Event()
        emit = lambda e: self.ctl.broker.publish({**e, "replay_burst": number})   # noqa: E731
        session = LiveSession(FileAudioSource(clip.wav), cap_f, None, emit=emit,
                              on_finish=lambda reason: self._burst_finished(session, clip, number, ended, reason),
                              max_seconds=clip.duration_s + 30, captions_label=label)
        self.session = session
        session.start()
        return ended

    def settled(self) -> bool:
        if self._busy.is_set() or not self._jobs.empty():
            return False
        s = self.manager.sessions[-1] if self.manager.sessions else None
        return s is None or (not s.updating and not s.pending and s.due_at is None and s.covered == len(s.bursts))

    def _burst_finished(self, session: LiveSession, clip: BurstClip, number: int,
                        ended: threading.Event, reason: str) -> None:
        ended.set()
        if reason != "cancel":
            self._busy.set()
            self._jobs.put((session, clip, number))

    # -- transcription, one burst at a time -----------------------------------------
    def _transcribe_loop(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            session, clip, number = job
            try:
                self._transcribe_burst(session, clip, number)
            except Exception as e:  # noqa: BLE001 - one bad burst must not end the replay
                logger.exception("Replay: burst %d failed", number)
                self.ctl.broker.publish({"type": "error", "message": f"Burst {number}: {e}", "replay": True})
            finally:
                if self._jobs.empty():
                    self._busy.clear()

    def _transcribe_burst(self, session: LiveSession, clip: BurstClip, number: int) -> None:
        audio = session.stop()
        entry = self.driver.log[number - 1] if len(self.driver.log) >= number else {}
        self._stage(f"Transcribing burst {number} ({len(audio) / 16000:.0f} s)")
        t0 = time.perf_counter()
        primary, fallback = self.ctl._transcriber_pair()
        result = transcribe_checked(primary, fallback, audio, session.final_text, on_stage=self._stage,
                                    caption_min_words=1)
        took = round(time.perf_counter() - t0, 2)
        text = result.text
        record = {"burst": number, "segment_id": clip.segment_id, "transcriber": result.who, "transcribe_s": took,
                  "words": len(text.split()), "audio_s": round(len(audio) / 16000, 1), "notes": list(result.notes),
                  "no_speech": result.no_speech, "attempts": list(result.attempts), "level": dict(result.stats)}
        self.transcriptions.append(record)
        self.ctl.broker.publish({"type": "transcript", "text": text, "replay_burst": number,
                                 "segment_id": clip.segment_id, **record})
        words = list(getattr(primary, "last_words", []) or []) if result.who == getattr(primary, "name", "primary") else []
        self.manager.add_burst(text, at=entry.get("start_wall", time.time()), duration_s=clip.duration_s,
                               feed=clip.feed, source_id=clip.segment_id, words=words)

    def _poll_loop(self) -> None:
        while not self._stopping.is_set():
            self.manager.poll()
            time.sleep(0.25)

    # -- extraction: the replay's provider, then local Ollama if it fails ------------------
    def _pipeline(self, llm) -> ReportPipeline:
        return ReportPipeline(llm, stop_phrase="", lexicon_store=self.ctl.engines.lexicon_store,
                              textbook_rag=self.ctl.engines.textbook_rag, on_stage=self._stage, attribution=True)

    def _extract(self, text: str, meta: dict) -> dict:
        source = f"replay:{self.callset.id}"
        try:
            from .pipeline import writer_mode_for
            report = self._pipeline(self.llm).run(text, source=source, timings={},
                                                  facts_only=writer_mode_for(getattr(self.ctl.settings, "writer_mode", "full"), bool(meta.get("closing"))))
            report["_meta"]["transcription"] = list(self.transcriptions)
            return report
        except PipelineError as e:
            if self.provider == "ollama":
                raise
            self.fallback = {"from": self.provider, "to": "ollama", "reason": str(e),
                             "at": time.strftime("%H:%M:%S"), "bursts": meta.get("bursts")}
            logger.warning("Replay: %s failed (%s); extracting with local Ollama", self.provider, e)
            self._stage(f"{self.provider} failed; writing the report with local Ollama")
            self.ctl.broker.publish({"type": "replay", "event": "fallback", **self.driver.status(),
                                     "provider": self.provider, "model": self.model, "commit": APP_COMMIT,
                                     "fallback": self.fallback})
            local = self.ctl.engines.make_llm(dataclasses.replace(self.settings, provider="ollama"))
            report = self._pipeline(local).run(text, source=source, timings={})
            report["_meta"]["fallback"] = dict(self.fallback)
            report["_meta"]["transcription"] = list(self.transcriptions)
            return report


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------
def save_wav(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, 16000, (np.clip(audio, -1, 1) * 32767).astype(np.int16))


def save_report(output_dir: Path, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reports").mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    body = json.dumps(report, indent=2)
    (output_dir / "handoff_report.json").write_text(body)
    path = output_dir / "reports" / f"handoff-{stamp}.json"
    path.write_text(body)
    return path


def load_user_settings(settings: Settings, path: Optional[Path] = None) -> Settings:
    """Apply choices saved from the app (provider, mic) on top of .env settings."""
    path = path or USER_SETTINGS_FILE
    try:
        saved = json.loads(path.read_text())
    except (OSError, ValueError):
        return settings
    changes = {}
    if saved.get("provider") in PROVIDERS:
        changes["provider"] = saved["provider"]
    if isinstance(saved.get("mic_device"), str):
        changes["mic_device"] = saved["mic_device"]
    if isinstance(saved.get("ollama_model"), str) and saved["ollama_model"].strip() and not settings.ollama_model_pinned:
        # a model chosen in the app is pinned, like OLLAMA_MODEL; OLLAMA_MODEL itself still wins
        changes["ollama_model"] = saved["ollama_model"].strip()
        changes["ollama_model_pinned"] = True
    return dataclasses.replace(settings, **changes)


def save_user_settings(settings: Settings, path: Optional[Path] = None) -> None:
    path = path or USER_SETTINGS_FILE
    try:
        saved = {"provider": settings.provider, "mic_device": settings.mic_device}
        if settings.ollama_model_pinned:
            saved["ollama_model"] = settings.ollama_model
        path.write_text(json.dumps(saved, indent=2))
    except OSError as e:
        logger.warning("Could not save user settings: %s", e)


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
def create_app(settings: Settings, engines: Optional[Engines] = None,
               on_quit: Optional[Callable[[], None]] = None) -> Flask:
    engines = engines or default_engines(settings)
    broker = EventBroker()
    ctl = Controller(settings, engines, broker)
    app = Flask(__name__, static_folder=None)
    app.config["controller"] = ctl
    app.config["broker"] = broker

    def ok(**body):
        return jsonify({"ok": True, **body})

    @app.errorhandler(Exception)
    def unexpected(e):  # every failure reaches the screen as words, never an HTML error page
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return jsonify({"ok": False, "message": e.description}), e.code
        logger.exception("Unhandled error")
        return jsonify({"ok": False, "message": f"Unexpected error: {type(e).__name__}: {e}"}), 500

    def fail(message: str, code: int = 400):
        return jsonify({"ok": False, "message": message}), code

    @app.get("/")
    def index():
        return send_from_directory(WEB_DIR, "index.html")

    @app.get("/web/<path:name>")
    def web_file(name: str):
        resp = send_from_directory(WEB_DIR, name)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.get("/api/status")
    def status():
        return jsonify(ctl.status())

    @app.get("/api/mics")
    def mics():
        try:
            devices = [{"index": i, "name": n} for i, n, _ in engines.list_mics()]
        except Exception as e:  # noqa: BLE001
            return fail(f"Cannot list microphones: {e}", 500)
        return ok(devices=devices, selected=ctl.settings.mic_device or "auto")

    @app.get("/api/samples")
    def samples():
        return jsonify(SAMPLES)

    @app.post("/api/settings")
    def settings_route():
        body = request.get_json(silent=True) or {}
        try:
            ctl.update_settings(provider=body.get("provider"), mic_device=body.get("mic_device"),
                                ollama_model=body.get("ollama_model"))
        except ValueError as e:
            return fail(str(e))
        return ok(status=ctl.status())

    @app.get("/api/ollama/models")
    def ollama_models_route():
        try:
            return ok(**ctl.ollama_models())
        except Exception as e:  # noqa: BLE001 - Ollama down: the screen says so
            return fail(f"Cannot list Ollama models: {e}", 503)

    @app.post("/api/check")
    def check():
        try:
            good, message = engines.make_llm(ctl.settings).check()
        except LLMError as e:
            good, message = False, str(e)
        return jsonify({"ok": good, "message": message})

    @app.post("/api/live/start")
    def live_start():
        try:
            ctl.start_live()
        except (RuntimeError, MicError) as e:
            return fail(str(e), 409)
        return ok()

    @app.post("/api/live/done")
    def live_done():
        try:
            ctl.done()
        except RuntimeError as e:
            return fail(str(e), 409)
        return ok()

    @app.post("/api/live/cancel")
    def live_cancel():
        ctl.cancel()
        return ok()

    @app.post("/api/report/text")
    def report_text():
        body = request.get_json(silent=True) or {}
        text = (body.get("text") or "").strip()
        if not text:
            return fail("Paste or choose a report first.")
        try:
            ctl.submit_text(text, source=body.get("source") or "text")
        except RuntimeError as e:
            return fail(str(e), 409)
        return ok()

    @app.get("/api/result")
    def result():
        return jsonify({"report": ctl.last_result})

    @app.get("/api/events")
    def events():
        q = broker.subscribe()

        def stream():
            try:
                yield f"data: {json.dumps({'type': 'status', 'status': ctl.status()})}\n\n"
                while True:
                    try:
                        event = q.get(timeout=15)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        continue
                    yield f"data: {json.dumps(event, default=str)}\n\n"
            finally:
                broker.unsubscribe(q)

        return Response(stream(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if settings.replay_mode_enabled:
        @app.get("/api/replay/callsets")
        def replay_callsets_route():
            return ok(callsets=ctl.replay_callsets(), folder=str(settings.replay_callsets_dir))

        @app.post("/api/replay/start")
        def replay_start_route():
            body = request.get_json(silent=True) or {}
            if not body.get("callset"):
                return fail("Choose a call-set.")
            try:
                status = ctl.start_replay(str(body["callset"]), body.get("provider"), body.get("speed"),
                                          record=bool(body.get("record")))
            except (ValueError, FileNotFoundError, LLMError) as e:
                return fail(str(e))
            except RuntimeError as e:
                return fail(str(e), 409)
            return ok(replay=status)

        @app.post("/api/replay/stop")
        def replay_stop_route():
            ctl.stop_replay()
            return ok(status=ctl.status())

        @app.get("/api/replay/status")
        def replay_status_route():
            return ok(replay=ctl._replay.status() if ctl._replay else None, state=ctl.state)

        @app.post("/api/replay/new_patient")
        def replay_new_patient_route():
            try:
                session = ctl.replay_run().manager.new_patient()
            except RuntimeError as e:
                return fail(str(e), 409)
            return ok(session=session.snapshot() if session else None)

        @app.post("/api/replay/merge")
        def replay_merge_route():
            try:
                session = ctl.replay_run().manager.merge_with_previous()
            except RuntimeError as e:
                return fail(str(e), 409)
            return ok(session=session.snapshot() if session else None)

    @app.post("/api/quit")
    def quit_app():
        if on_quit is None:
            return fail("Quit is only available when started from the app launcher.", 409)
        threading.Timer(0.3, on_quit).start()
        return ok()

    return app
