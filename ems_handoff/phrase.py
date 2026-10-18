"""
phrase.py — find and remove the hands-free stop phrase ("confirm report ready").

Purpose
    One matcher, shared by the live listener (to trigger DONE) and the report
    pipeline (to drop the phrase from the transcript before the LLM sees it).

How matching works
    Text is lower-cased and stripped of punctuation. The phrase matches when
    some run of words in the text has the same number of words as the phrase
    and an overall similarity of at least ``threshold`` (difflib ratio).
    This tolerates small mis-hearings ("confirm the port ready" scores about 0.8)
    without firing on unrelated speech. A substring test alone used to fire on
    "report ready" anywhere in the transcript.

Threads
    Pure functions, safe from any thread.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

_WORD = re.compile(r"[a-z0-9']+")


def words(text: str) -> List[str]:
    return _WORD.findall((text or "").lower())


def _best_window(text_words: List[str], phrase_words: List[str]) -> Tuple[float, int]:
    """Best similarity of any window (len(phrase) +/- 1 words). Returns (score, start index)."""
    n = len(phrase_words)
    target = " ".join(phrase_words)
    best, best_i = 0.0, -1
    for size in (n - 1, n, n + 1):
        if size < 1 or size > len(text_words):
            continue
        for i in range(len(text_words) - size + 1):
            score = SequenceMatcher(None, " ".join(text_words[i:i + size]), target).ratio()
            if score > best:
                best, best_i = score, i
    return best, best_i


def contains_phrase(text: str, phrase: str, threshold: float = 0.82) -> bool:
    pw, tw = words(phrase), words(text)
    if not pw or not tw:
        return False
    score, _ = _best_window(tw, pw)
    return score >= threshold


def strip_phrase(text: str, phrase: str, threshold: float = 0.82,
                 tail_words: Optional[int] = 12) -> str:
    """Remove the stop phrase if it appears near the end of the transcript.

    Only the last ``tail_words`` words are searched, so a phrase-like run of
    words earlier in a real report is never deleted.
    """
    pw = words(phrase)
    raw = (text or "").strip()
    if not pw or not raw:
        return raw
    tokens = raw.split()
    lowered = [(_WORD.findall(t.lower()) or [""])[0] for t in tokens]
    start = max(0, len(tokens) - (tail_words or len(tokens)))
    tail = lowered[start:]
    score, i = _best_window(tail, pw)
    if score < threshold or i < 0:
        return raw
    # drop from the phrase start to the end; what follows the phrase is noise
    cut = start + i
    return " ".join(tokens[:cut]).rstrip(" ,;:-")
