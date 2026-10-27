"""
transcript_checks.py — two sanity checks around the final transcript.

Silence
    A recording whose level never rose above the noise floor has no speech.
    Whisper is not called: given silence it invents "Thank you." (a silent
    USB input on 2026-10-25 did exactly that, and the report came out
    UNCLEAR). The caller shows "no speech detected" instead.

Words per second
    Radio speech runs about 1.5 to 4 words per second. Fewer than one word per
    two seconds on a burst over ten seconds means the transcriber most likely
    dropped the burst (the post-fix S038_01 run: 15.2 s, 44 words on disk,
    transcribed as "... ..."). The transcriber is retried once, then the local
    one is tried; the most plausible result wins and every attempt is recorded
    so the report's _meta says what happened.

Both the live path (server.Controller._transcribe) and replay (server.ReplayRun)
use transcribe_checked; nothing here depends on either.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Tuple, Any, Callable, Dict, List, Optional

import numpy as np

from .transcribe import TranscriptionError

logger = logging.getLogger(__name__)

RATE = 16000
NOISE_FLOOR_RMS = 0.002          # -54 dBFS: a dead input reads 0.0; quiet speech peaks far above this
MIN_WORDS_PER_SECOND = 0.5       # one word per two seconds
MIN_SECONDS_FOR_RATE_CHECK = 10.0


def speech_stats(audio: np.ndarray, block_s: float = 0.1) -> Dict[str, float]:
    """Level statistics per 100 ms block: peak and mean RMS, and the share of blocks above the floor."""
    audio = np.asarray(audio, dtype=np.float32)
    n = int(block_s * RATE)
    if audio.size == 0 or n <= 0:
        return {"seconds": 0.0, "peak_rms": 0.0, "mean_rms": 0.0, "active_fraction": 0.0}
    blocks = [audio[i:i + n] for i in range(0, len(audio), n)]
    rms = np.array([float(np.sqrt(np.mean(b ** 2))) for b in blocks if b.size])
    return {"seconds": round(len(audio) / RATE, 2), "peak_rms": round(float(rms.max()), 5),
            "mean_rms": round(float(rms.mean()), 5),
            "active_fraction": round(float((rms > NOISE_FLOOR_RMS).mean()), 3)}


def has_speech(audio: np.ndarray) -> bool:
    return speech_stats(audio)["peak_rms"] > NOISE_FLOOR_RMS


def plausible_rate(words: int, seconds: float) -> bool:
    """False only for a long enough recording with too few words to be real speech."""
    if seconds <= MIN_SECONDS_FOR_RATE_CHECK:
        return True
    return words / seconds >= MIN_WORDS_PER_SECOND


@dataclass
class TranscriptionResult:
    text: str                      # "" when nothing usable came back
    who: str                       # which transcriber (or "live captions", "none")
    no_speech: bool = False
    notes: List[str] = field(default_factory=list)
    attempts: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.text.strip())

    def meta(self) -> Dict[str, Any]:
        """What goes into the report's _meta.transcription."""
        return {"transcriber": self.who, "no_speech": self.no_speech, "attempts": list(self.attempts),
                "notes": list(self.notes), "level": dict(self.stats)}


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def collapse_repeats(text: str, min_run: int = 3) -> Tuple[str, Optional[str]]:
    """Whisper loops on garbled radio audio: the same sentence three, four, five times in a row.
    A run of min_run or more identical sentences (ignoring case and punctuation) is cut to one.
    Returns (text, note or None). Two in a row stay: "Shady Grove online. Shady Grove online."
    is how a crew really calls."""
    parts = _SENTENCE.split(text.strip()) if text and text.strip() else []
    if len(parts) < min_run:
        return text, None
    key = [re.sub(r"[^\w\s]", "", s).lower().split() for s in parts]
    out, i, cut = [], 0, 0
    while i < len(parts):
        j = i
        while j + 1 < len(parts) and key[j + 1] == key[i] and key[i]:
            j += 1
        run = j - i + 1
        if run >= min_run:
            out.append(parts[i])
            cut += run - 1
        else:
            out.extend(parts[i:j + 1])
        i = j + 1
    notes = []
    if cut:
        notes.append(f"collapsed {cut} repeated sentence(s) (a transcriber loop)")
    out, counted = _cut_number_run(out)
    if counted:
        notes.append(f"cut {counted} counted-off numbers (a transcriber loop)")
    if not notes:
        return text, None
    return " ".join(out), "; ".join(notes)


_NUMBER_SENTENCE = re.compile(r"^\d{1,3}[.,!?]?$")


def _cut_number_run(parts: List[str], min_run: int = 10, keep: int = 3) -> Tuple[List[str], int]:
    """The other loop shape: after "GCS 14" the decoder counts, "4. 4. 6. 7. 7. 8. 8. 8. 9. 9. ... 12." for sixty
    tokens (large-v3-turbo on SY188 of the generated set, three SNRs). A run of min_run or more sentences that are
    each one bare number keeps its first `keep` (a GCS breakdown is three numbers) and drops the rest."""
    out: List[str] = []
    i = cut = 0
    while i < len(parts):
        j = i
        while j < len(parts) and _NUMBER_SENTENCE.match(parts[j].strip()):
            j += 1
        run = j - i
        if run >= min_run:
            out.extend(parts[i:i + keep])
            cut += run - keep
            i = j
        else:
            out.append(parts[i])
            i += 1
    return out, cut


def transcribe_checked(primary: Any, fallback: Any, audio: np.ndarray, caption_text: str = "",
                       on_stage: Callable[[str], None] = lambda m: None,
                       caption_min_words: int = 5) -> TranscriptionResult:
    """Transcribe with the silence gate and the words-per-second rule.

    primary / fallback: objects with .name, .describe(), .transcribe(audio); either may be None.
    Order: primary; if its word rate is implausible, primary again; then the fallback when it
    is a different transcriber. The first plausible non-empty text wins; otherwise the longest
    non-empty text; otherwise the live captions (at least caption_min_words words); else "".
    """
    stats = speech_stats(audio)
    seconds = stats["seconds"]
    if stats["peak_rms"] <= NOISE_FLOOR_RMS:
        note = (f"no speech detected: the level never rose above the noise floor "
                f"(peak {stats['peak_rms']:.4f} over {seconds:.0f} s); the transcriber was not called")
        logger.warning("Transcription skipped: %s", note)
        return TranscriptionResult("", "none", no_speech=True, notes=[note], stats=stats)

    result = TranscriptionResult("", "none", stats=stats)
    plan = []
    if primary is not None:
        plan.append((primary, "primary"))
        plan.append((primary, "retry"))
    if fallback is not None and (primary is None or getattr(fallback, "name", "") != getattr(primary, "name", "")):
        plan.append((fallback, "fallback"))

    texts: List[tuple] = []
    for t, role in plan:
        if role == "retry":
            last = result.attempts[-1] if result.attempts else None
            if not last or last.get("error") or last["words"] == 0 or last["plausible"]:
                continue        # retry only an implausible result, never an error or an empty one
            on_stage(f"Transcript looks incomplete ({last['words']} words in {seconds:.0f} s); retrying with {t.describe()}")
        elif role == "fallback" and result.attempts and not any(a.get("error") or not a["plausible"] for a in result.attempts if not a.get("error")) and any(a["plausible"] and a["words"] for a in result.attempts):
            break               # the primary already gave a plausible transcript
        elif role == "fallback":
            on_stage(f"Retrying transcription with {t.describe()}")
        t0 = time.perf_counter()
        attempt = {"role": role, "transcriber": t.describe(), "seconds": seconds}
        try:
            text = t.transcribe(audio) or ""
            text, loop_note = collapse_repeats(text)
            if loop_note:
                result.notes.append(f"{t.describe()}: {loop_note}")
            words = len(text.split())
            attempt.update(words=words, plausible=plausible_rate(words, seconds) and words > 0,
                           took_s=round(time.perf_counter() - t0, 2))
            if words == 0:
                result.notes.append(f"{t.describe()} heard no speech")
            elif not attempt["plausible"]:
                result.notes.append(f"{t.describe()} returned {words} words for {seconds:.0f} s of audio "
                                    f"({words / seconds:.2f} words/s): implausible")
            texts.append((attempt["plausible"], words, text, t.describe()))
        except TranscriptionError as e:
            attempt.update(error=str(e), words=0, plausible=False, took_s=round(time.perf_counter() - t0, 2))
            result.notes.append(str(e))
        result.attempts.append(attempt)
        if attempt.get("plausible"):
            result.text, result.who = texts[-1][2], texts[-1][3]
            return result

    usable = [x for x in texts if x[1] > 0]
    if usable:
        _, _, text, who = max(usable, key=lambda x: x[1])
        result.text, result.who = text, who
        result.notes.append(f"kept the longest transcript ({len(text.split())} words, {who}) after the checks")
        return result
    if len(caption_text.split()) >= caption_min_words:
        on_stage("Using the live captions as the transcript")
        result.text, result.who = caption_text, "live captions"
        result.notes.append("using the live captions as the transcript")
        return result
    result.notes.append("no transcript")
    return result
