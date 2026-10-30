"""
incident.py — one session per patient, built burst by burst (docs/design/incident-sessions.md).

Purpose
    A radio call is a run of short bursts. Each finished burst's transcript is
    appended to the session transcript under a header (time, feed, burst
    number). The report is always re-extracted from the whole session
    transcript: never from a burst alone, never by patching the previous JSON.

Trigger rules
    Extraction runs after a quiet period on the channel (5 s online, 25 s
    offline, set by the owner) or at once on a closing cue ("see you on
    arrival", "... clear"). One extraction is in flight per session; bursts
    that arrive during a run coalesce into one queued run that covers
    everything received. Across sessions one worker runs jobs in turn, since
    the local model is shared.

Updating, not blank
    The previous report stays current with ``updating`` set and the number of
    bursts it covers until the new one replaces it. Every report is kept in the
    session's history, with a one-line "what this burst changed" summary.

Boundary rule
    A new session starts when the gap since the last burst is over
    ``new_session_gap_s`` (5 minutes) or the operator pressed New patient.
    Merge with previous puts the open session's bursts back into the previous
    one, re-sorted by time, and re-extracts once.

Burst attribution
    The prompt asks the model for the burst number behind each filled field
    ("sources"). After extraction, every clinical field with a source is checked
    mechanically against the cited burst's words; a field with no source, or
    whose words are not in the cited burst, is marked suspect.

Clock
    Burst times and ``clock()`` share one timebase, chosen by the owner: wall
    time for the live mic, the replay clock for a replayed call-set. Tests pass
    a fake clock and call ``poll`` themselves.

Threads
    "incident-worker": runs extractions one at a time. ``add_burst``, ``poll``,
    ``new_patient`` and ``merge_with_previous`` may be called from any thread.
    ``emit`` is called from the calling thread or the worker.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

QUIET_ONLINE_S = 5.0
QUIET_OFFLINE_S = 25.0
NEW_SESSION_GAP_S = 300.0

# Closing cues: the call is over, extract now rather than after the quiet period.
_CLOSING_CUES = re.compile(
    r"see you (?:on|at|upon) arrival|see you when (?:you|we) get (?:here|there)|see you (?:soon|shortly|there|in a (?:few|couple))"
    r"|\bend of report\b|\bwe'?ll see you\b|\bcopy,? see you\b", re.I)
_CLEAR_AT_END = re.compile(r"\bclear\b\W*$", re.I)


def is_closing_cue(text: str) -> bool:
    """True when the burst ends the call ("see you on arrival", "Medic 14 clear")."""
    t = (text or "").strip()
    if not t:
        return False
    if _CLOSING_CUES.search(t):
        return True
    tail = " ".join(t.split()[-6:])
    return bool(_CLEAR_AT_END.search(tail))


# ---------------------------------------------------------------------------
# Bursts and the session transcript
# ---------------------------------------------------------------------------
@dataclass
class Burst:
    number: int                 # 1-based within its session
    text: str
    time: float                 # start, in the manager's timebase (epoch seconds)
    duration_s: float = 0.0
    feed: str = ""
    source_id: str = ""         # e.g. the corpus segment id, or "" for the live mic

    @property
    def end(self) -> float:
        return self.time + self.duration_s

    @property
    def words(self) -> int:
        return len(self.text.split())


def burst_header(b: Burst) -> str:
    stamp = time.strftime("%H:%M:%S", time.gmtime(b.time))
    feed = f" · feed {b.feed}" if b.feed else ""
    return f"[Burst {b.number} · {stamp} UTC{feed}]"


def session_transcript(bursts: List[Burst]) -> str:
    """The text the model reads: every burst under its header, in order."""
    return "\n\n".join(f"{burst_header(b)}\n{b.text.strip()}" for b in bursts)


# ---------------------------------------------------------------------------
# What this burst changed
# ---------------------------------------------------------------------------
# Clinical fields a burst can state. Judgement fields (alert, summary, recommendations,
# data_quality) are derived by the model and are not attributed or diffed field by field.
_CLINICAL_PREFIXES = ("patient.", "chief_complaint", "mechanism_or_onset", "eta_minutes",
                      "vital_signs.", "assessment_findings.", "ems_interventions", "history.")
_SECTION_NAMES = {"patient": "patient", "chief_complaint": "complaint", "mechanism_or_onset": "onset",
                  "eta_minutes": "ETA", "vital_signs": "vitals", "assessment_findings": "assessment",
                  "ems_interventions": "interventions", "history": "history"}


def _is_empty(v: Any) -> bool:
    if isinstance(v, dict):
        return all(_is_empty(x) for x in v.values())
    if isinstance(v, list):
        return all(_is_empty(x) for x in v)
    return v is None or v == ""


def leaf_fields(report: Dict[str, Any]) -> Dict[str, Any]:
    """{dotted path: value} for every filled clinical field. Lists are one field each."""
    out: Dict[str, Any] = {}

    def walk(prefix: str, value: Any):
        if isinstance(value, dict):
            for k, v in value.items():
                walk(f"{prefix}.{k}" if prefix else k, v)
        elif not _is_empty(value):
            out[prefix] = value

    for key in ("patient", "chief_complaint", "mechanism_or_onset", "eta_minutes",
                "vital_signs", "assessment_findings", "ems_interventions", "history"):
        if key in report:
            walk(key, report[key])
    return out


def _section(path: str) -> str:
    return _SECTION_NAMES.get(path.split(".")[0], path.split(".")[0])


def _short(v: Any) -> str:
    if isinstance(v, list):
        return f"{len(v)} items"
    s = str(v)
    return s if len(s) <= 24 else s[:21] + "..."


def describe_change(old: Optional[Dict[str, Any]], new: Dict[str, Any], burst_number: int) -> str:
    """One line: what the report gained, changed or lost when burst N was added."""
    before = leaf_fields(old or {})
    after = leaf_fields(new)
    added = [p for p in after if p not in before]
    changed = [p for p in after if p in before and after[p] != before[p]]
    removed = [p for p in before if p not in after]
    parts = []
    if added:
        by_section: Dict[str, List[str]] = {}
        for p in added:
            by_section.setdefault(_section(p), []).append(p)
        bits = []
        for sec, paths in by_section.items():
            if sec == "ETA":
                bits.append(f"ETA {after['eta_minutes']} min")
            elif sec == "interventions":
                bits.append(f"interventions ({len(after['ems_interventions']) if isinstance(after['ems_interventions'], list) else 1})")
            elif len(paths) == 1 and "." not in paths[0]:
                bits.append(sec)
            else:
                bits.append(f"{sec} ({', '.join(p.split('.')[-1] for p in paths)})")
        parts.append("added: " + ", ".join(bits))
    if changed:
        parts.append("changed: " + ", ".join(f"{p} {_short(before[p])} -> {_short(after[p])}" for p in changed))
    if removed:
        parts.append("removed: " + ", ".join(removed))
    old_alert = (old or {}).get("alert_type"), (old or {}).get("alert_level")
    new_alert = new.get("alert_type"), new.get("alert_level")
    if new_alert != old_alert and new_alert[0]:
        parts.append(f"alert {new_alert[0]}{' ' + new_alert[1] if new_alert[1] else ''}")
    if not parts:
        return f"burst {burst_number}: no new facts"
    return f"burst {burst_number} " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Burst attribution, checked against the words of the cited burst
# ---------------------------------------------------------------------------
_NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                 "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12",
                 "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
                 "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30", "forty": "40",
                 "fifty": "50", "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100"}
_STOP = {"the", "and", "with", "for", "from", "that", "this", "was", "were", "has", "had", "not", "but",
         "per", "via", "patient", "year", "years", "old", "on", "of", "to", "in", "at",
         "unspecified", "unknown", "none", "stated", "mentioned", "reported", "given", "administered"}


def _words(text: str) -> List[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+(?:[/.-][a-z0-9]+)*", (text or "").lower()):
        out.append(w)
        out.extend(p for p in re.split(r"[/.-]", w) if p and p != w)
    return [_NUMBER_WORDS.get(w, w) for w in out]


def _value_words(value: Any) -> List[str]:
    if isinstance(value, dict):
        return [w for v in value.values() for w in _value_words(v)]
    if isinstance(value, list):
        return [w for v in value for w in _value_words(v)]
    if isinstance(value, bool) or value is None:
        return []
    return _words(str(value))


def value_in_text(value: Any, text: str) -> bool:
    """Does the value's content appear in the text? A number by its digits (number words
    count), a phrase by any content word (4-letter prefix match, so 'drooling' finds 'drool')."""
    text_words = _words(text)
    text_set = set(text_words)
    prefixes = {w[:4] for w in text_words if len(w) >= 4}
    candidates = [w for w in _value_words(value) if w not in _STOP]
    if not candidates:
        return False
    for w in candidates:
        if w in text_set:
            return True
        if w.isdigit():
            continue
        if len(w) >= 4 and w[:4] in prefixes:
            return True
    return False


def verify_attribution(report: Dict[str, Any], bursts: List[Burst]) -> Dict[str, Any]:
    """Check the model's "sources" against the bursts and mark the rest suspect.

    Returns {"verified": {path: burst}, "suspect": [paths], "unattributed": [paths]} and
    stores it under report["attribution"]. Suspect = cited burst does not contain the
    value's words, or the burst number does not exist. Unattributed = filled with no source.
    Both are shown as suspect on screen; the split says why."""
    sources = report.get("sources") if isinstance(report.get("sources"), dict) else {}
    by_number = {b.number: b for b in bursts}
    verified: Dict[str, int] = {}
    suspect: List[str] = []
    unattributed: List[str] = []
    fields = leaf_fields(report)
    for path, value in fields.items():
        if path == "patient.age_unit":
            continue            # the unit goes with the age: checked below, never on its own
        n = source_for(sources, path)
        if n is None:
            unattributed.append(path)
            continue
        burst = by_number.get(n)
        if burst is not None and value_in_text(value, burst.text):
            verified[path] = n
        else:
            # the model named the wrong burst: say so, but find the burst that does state it
            stated_in = next((b.number for b in bursts if value_in_text(value, b.text)), None)
            if stated_in is not None:
                verified[path] = stated_in
            else:
                suspect.append(path)
    if "patient.age_unit" in fields:
        if "patient.age" in verified:
            verified["patient.age_unit"] = verified["patient.age"]
        elif "patient.age" in suspect:
            suspect.append("patient.age_unit")
        elif "patient.age" in unattributed:
            unattributed.append("patient.age_unit")
    result = {"verified": verified, "suspect": suspect, "unattributed": unattributed}
    report["attribution"] = result
    return result


def source_for(sources: Dict[str, Any], path: str) -> Optional[int]:
    """The burst number the model gave for a field: the exact path, else its parent section,
    else any key under it (the model writes "ems_interventions[0].intervention" for a list)."""
    candidates = [sources.get(path), sources.get(path.split(".")[0])]
    candidates += [v for k, v in sources.items() if k.startswith(path + "[") or k.startswith(path + ".")]
    for n in candidates:
        if n is None or isinstance(n, bool):
            continue
        try:
            return int(n)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
@dataclass
class IncidentSession:
    id: str
    bursts: List[Burst] = field(default_factory=list)
    reports: List[Dict[str, Any]] = field(default_factory=list)   # history, oldest first
    current: Optional[Dict[str, Any]] = None                       # the report on screen
    covered: int = 0                # bursts the current report covers
    updating: bool = False          # an extraction is running for this session
    pending: bool = False           # bursts arrived during the run: one more run is queued
    queued_through: int = 0         # bursts the running/queued job will cover
    change_lines: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    closed: bool = False            # New patient was pressed, or a later session started
    due_at: Optional[float] = None  # quiet-period deadline, None when nothing is waiting
    events: List[str] = field(default_factory=list)   # operator actions, logged with the burst they applied to
    closing: Optional[str] = None   # why the call is closing ("closing cue in burst 3", "New patient pressed"), else None
    ask: Optional[str] = None       # the one line the receiver may act on when the call closes with a required slot empty
    asked: bool = False             # the ask rule has run at least once for this closing

    @property
    def last_burst(self) -> Optional[Burst]:
        return self.bursts[-1] if self.bursts else None

    @property
    def transcript(self) -> str:
        return session_transcript(self.bursts)

    def snapshot(self) -> Dict[str, Any]:
        """What the screen shows for this session."""
        return {
            "id": self.id,
            "bursts": [{"number": b.number, "time": b.time, "duration_s": b.duration_s, "feed": b.feed,
                        "source_id": b.source_id, "words": b.words, "text": b.text} for b in self.bursts],
            "transcript": self.transcript,
            "report": self.current,
            "covered": self.covered,
            "of": len(self.bursts),
            "updating": self.updating,
            "pending": self.pending,
            "change_lines": list(self.change_lines),
            "reports": len(self.reports),
            "error": self.last_error,
            "closed": self.closed,
            "state": self.state(),
            "closing": self.closing,
            "ask": self.ask,
        }

    def state(self) -> Dict[str, Any]:
        """The patient state (ems_handoff.state) built from this session's report history."""
        from .state import build_state
        history = []
        for report in self.reports:
            covers = int(((report.get("_meta") or {}).get("session") or {}).get("covers") or 0)
            at = self.bursts[covers - 1].end if 0 < covers <= len(self.bursts) else None
            history.append((f"burst {covers}" if covers else f"report {len(history) + 1}", at, report))
        try:
            return build_state(history).to_dict()
        except Exception as e:  # noqa: BLE001 - the state is a view; a bad report must not take the screen down
            logger.exception("Session %s: state failed", self.id)
            return {"error": f"{type(e).__name__}: {e}"}


class IncidentManager:
    """Owns the sessions, the trigger rules, and the one extraction worker."""

    def __init__(self, extract: Callable[[str, Dict[str, Any]], Dict[str, Any]],
                 quiet_s: float = QUIET_OFFLINE_S, new_session_gap_s: float = NEW_SESSION_GAP_S,
                 clock: Callable[[], float] = time.time,
                 emit: Callable[[dict], None] = lambda e: None, start_worker: bool = True):
        self._extract = extract
        self.quiet_s = quiet_s
        self.new_session_gap_s = new_session_gap_s
        self.clock = clock
        self._emit = emit
        self.sessions: List[IncidentSession] = []
        self._lock = threading.RLock()
        self._jobs: "queue.Queue[Optional[IncidentSession]]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        if start_worker:
            self._worker = threading.Thread(target=self._work, name="incident-worker", daemon=True)
            self._worker.start()

    # -- bursts ---------------------------------------------------------------
    @property
    def open_session(self) -> Optional[IncidentSession]:
        return self.sessions[-1] if self.sessions and not self.sessions[-1].closed else None

    def add_burst(self, text: str, at: Optional[float] = None, duration_s: float = 0.0,
                  feed: str = "", source_id: str = "") -> Tuple[IncidentSession, Burst]:
        """Append a finished burst to the open session, or start a new one by the boundary rule."""
        at = self.clock() if at is None else at
        with self._lock:
            session = self.open_session
            why = ""
            if session is None:
                why = "first burst" if not self.sessions else "after New patient"
            elif session.last_burst is not None and at - session.last_burst.end > self.new_session_gap_s:
                why = f"gap of {at - session.last_burst.end:.0f} s since the last burst"
            if why:
                session = self._new_session(why)
            burst = Burst(number=len(session.bursts) + 1, text=(text or "").strip(), time=at,
                          duration_s=duration_s, feed=feed, source_id=source_id)
            session.bursts.append(burst)
            if is_closing_cue(burst.text):
                session.due_at = None
                session.closing = f"closing cue in burst {burst.number}"
                self._trigger(session, session.closing)
            else:   # quiet starts when the channel went quiet: the latest end, not this burst's
                session.due_at = max(b.end for b in session.bursts) + self.quiet_s
            logger.info("Session %s: burst %d (%d words) at %.1f", session.id, burst.number, burst.words, at)
            self._publish(session, "burst", burst=burst.number)
            return session, burst

    def poll(self, now: Optional[float] = None) -> None:
        """Fire the quiet-period trigger for any session whose deadline has passed."""
        now = self.clock() if now is None else now
        with self._lock:
            for s in self.sessions:
                if s.due_at is not None and now >= s.due_at:
                    s.due_at = None
                    self._trigger(s, f"quiet for {self.quiet_s:.0f} s after burst {len(s.bursts)}")

    # -- operator buttons -------------------------------------------------------
    def new_patient(self) -> Optional[IncidentSession]:
        """Close the open session; the next burst starts a new one. Pending text is extracted."""
        with self._lock:
            session = self.open_session
            if session is None:
                return None
            session.closed = True
            session.closing = "New patient pressed"
            session.events.append(f"New patient pressed after burst {len(session.bursts)}")
            if session.due_at is not None:
                session.due_at = None
                self._trigger(session, "New patient pressed")
            self._publish(session, "closed")
            return session

    def merge_with_previous(self) -> Optional[IncidentSession]:
        """Undo a split: the newest session's bursts join the one before it, in time order,
        and one re-extraction runs on the merged transcript."""
        with self._lock:
            if len(self.sessions) < 2:
                return None
            newest = self.sessions.pop()
            target = self.sessions[-1]
            target.closed = False
            moved = len(newest.bursts)
            target.bursts = sorted(target.bursts + newest.bursts, key=lambda b: b.time)
            for i, b in enumerate(target.bursts, 1):
                b.number = i
            target.events.append(f"Merge with previous: {moved} burst(s) of {newest.id} joined after burst "
                                 f"{len(target.bursts) - moved}; renumbered")
            target.due_at = None
            self._trigger(target, "Merge with previous")
            self._publish(target, "merged", merged=newest.id)
            return target

    # -- extraction ------------------------------------------------------------
    def _new_session(self, why: str) -> IncidentSession:
        if self.sessions:
            self.sessions[-1].closed = True
        session = IncidentSession(id=f"P{len(self.sessions) + 1}")
        session.events.append(f"session started: {why}")
        self.sessions.append(session)
        logger.info("Session %s started (%s)", session.id, why)
        return session

    def _trigger(self, session: IncidentSession, why: str) -> None:
        """Queue one extraction covering everything received; coalesce while one runs."""
        if len(session.bursts) <= session.queued_through:
            return                      # nothing new since the job already queued/running
        if session.updating:
            session.pending = True      # the running job finishes first; one more run follows
            logger.info("Session %s: %s, run queued behind the current one", session.id, why)
            return
        session.updating = True
        session.pending = False
        session.queued_through = len(session.bursts)
        logger.info("Session %s: extraction (%s), covering %d bursts", session.id, why, session.queued_through)
        self._publish(session, "extracting", why=why)
        self._jobs.put(session)

    def run_pending(self, timeout: float = 60.0) -> None:
        """Run queued jobs on this thread (tests, and a worker-less owner)."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                job = self._jobs.get(timeout=0.05)
            except queue.Empty:
                return
            self._run_job(job)

    def _work(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._run_job(job)

    def _run_job(self, session: IncidentSession) -> None:
        with self._lock:
            bursts = list(session.bursts[:session.queued_through])
            covers = len(bursts)
            transcript = session_transcript(bursts)
            meta = {"session": session.id, "bursts": covers, "of": len(session.bursts), "closing": bool(session.closing)}
        try:
            report = self._extract(transcript, meta)
            verify_attribution(report, bursts)
            report.setdefault("_meta", {})
            report["_meta"]["session"] = {"id": session.id, "covers": covers, "burst_ids": [b.source_id for b in bursts]}
        except Exception as e:  # noqa: BLE001 - the session survives a failed run
            logger.exception("Session %s: extraction failed", session.id)
            with self._lock:
                session.last_error = f"{type(e).__name__}: {e}"
                session.updating = False
                self._publish(session, "error", message=session.last_error)
                self._after_job(session)
            return
        with self._lock:
            line = describe_change(session.current, report, covers)
            session.change_lines.append(line)
            report["_meta"]["change_line"] = line
            session.reports.append(report)
            session.current = report
            session.covered = covers
            session.last_error = None
            session.updating = False
            logger.info("Session %s: report covers %d of %d bursts; %s", session.id, covers, len(session.bursts), line)
            self._publish(session, "report", change_line=line)
            self._ask(session)
            self._after_job(session)

    def _ask(self, session: IncidentSession) -> None:
        """The ask rule: when the call is closing and a slot the alert type requires is still empty, one line.

        The system never transmits; the line is for the receiver, who may ask on the radio or let it go. No line
        while the call is open (the crew may still say it), none for optional slots, one line at a time.
        """
        if not session.closing:
            return
        st = session.state()
        missing = st.get("missing") or []
        alert = st.get("alert_type") or ""
        ask = f"Not stated for {alert or 'this call'}: {', '.join(missing)}" if missing else None
        if ask != session.ask or not session.asked:
            session.ask = ask
            session.asked = True
            session.events.append(f"ask after burst {len(session.bursts)} ({session.closing}): {ask or 'nothing missing'}")
            logger.info("Session %s: %s", session.id, ask or "closing with nothing missing")
            self._publish(session, "ask", ask=ask)

    def _after_job(self, session: IncidentSession) -> None:
        if session.pending or len(session.bursts) > session.queued_through:
            session.pending = False
            self._trigger(session, "bursts received during the run")

    def _publish(self, session: IncidentSession, event: str, **extra: Any) -> None:
        try:
            self._emit({"type": "incident", "event": event, "session": session.snapshot(), **extra})
        except Exception:  # noqa: BLE001 - a listener's failure is not the session's
            logger.exception("incident emit failed")

    def stop(self) -> None:
        if self._worker is not None:
            self._jobs.put(None)
            self._worker.join(timeout=5)
            self._worker = None
