"""
transcript_corrector.py
========================
Precision-first post-ASR correction for EMS radio transcripts.

This module repairs low-frequency medical vocabulary *before* the transcript reaches the extraction LLM, 
reusing the terminology already encoded in ``lexicon.py``.

Classes of errors:

  * Misspellings of distinctive terms** (drug / medication names) that are
    NOT real English words are auto-corrected. These are safe: a generic drug
    name is long, distinctive, and never collides with ordinary prose.

  * Real-word / valid-acronym confusions** -- "bronchi" for "rhonchi",
    "POST" for "POLST", "DNR before" for "DNR but" -- are NEVER auto-corrected.
    Only context disambiguates them, so they are left in place and surfaced as
    ``flagged`` suspects for the context-aware LLM extraction pass to resolve.

Safety boundaries:
  1. Only categories in ``AUTOCORRECT_CATEGORIES`` are ever rewritten.
  2. A token is only rewritten if it is NOT a known English word.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .lexicon import LEXICON

logger = logging.getLogger(__name__)

try:
    import jellyfish # phonetic 
    _HAVE_PHONETIC = True
except ImportError: 
    _HAVE_PHONETIC = False
    logger.info("jellyfish not installed; phonetic matching disabled")

try:
    from spellchecker import SpellChecker
    _SPELL = SpellChecker()
    _HAVE_SPELL = True
except ImportError:
    _SPELL = None
    _HAVE_SPELL = False
    logger.warning("pyspellchecker not installed; real-word safety gate "
                   "unavailable -- fuzzy auto-correction disabled "
                   "(deterministic known-misspelling and context fixes still apply)")

# =============================================================================
# TUNING
# =============================================================================
# Only categories with long, distinctive, non-word terms where a fuzzy + phonetic match is safe. 
# Every other category can only ever be flagged for the LLM.
AUTOCORRECT_CATEGORIES = {"drugs", "home_medications"}

MIN_TOKEN_LEN = 5 # ignore short tokens / acronyms as targets
AUTOCORRECT_THRESHOLD = 0.86 # high bar to actually rewrite text
FLAG_THRESHOLD = 0.78 # lower bar to merely surface a suspect
MIN_MARGIN = 0.05 # best match must beat the runner-up by this much
PHONETIC_BONUS = 0.15 # added when metaphone keys match (ratio-gated)

_TARGET_RE = re.compile(r"[a-z][a-z\-']+$")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']*|[^A-Za-z]+")

# =============================================================================
# STOPLIST  (tokens that must never be a correction/flag target)
# =============================================================================
# These are ordinary words that happen to sit within FLAG_THRESHOLD of a
# clinical term and would otherwise be surfaced as noisy suspects. The clearest
# offenders are the words in the live-recording stop phrase ("confirm report
# ready") — "ready" fuzzy-matches "brady" (bradycardia) — which are an artifact
# of the capture mechanism, not something the paramedic said about the patient.
# Left in the transcript verbatim; simply never treated as a correction target.
NEVER_CORRECT: set = {
    "confirm", "report", "ready",   # live-mode stop phrase
}

# =============================================================================
# KNOWN MISSPELLINGS  (unambiguous variant -> canonical)
# =============================================================================
# A hand-curated correction lexicon for recurring ASR/OCR errors whose target is
# the SAME regardless of context. Authored as {canonical: [variants...]}
KNOWN_MISSPELLINGS: Dict[str, List[str]] = {
    # EMS-administered drugs
    "furosemide":  ["ferosemide", "ferosamide", "furosemid", "phurosemide"],
    "metoprolol":  ["metoprol", "metaprolol", "metroprolol"],
    "ondansetron": ["ondansitron", "ondancetron"],
    "midazolam":   ["midazalam", "midozolam"],
    "epinephrine": ["epinephrin", "epinephrene"],
    "amiodarone":  ["amioderone", "amiodarohn"],
    # heard on the MoCo radio corpus (CS014, a 3-month-old with croup)
    "dexamethasone": ["texamethasone", "dexamethazone", "dexamethasome"],
    "racemic epinephrine": ["cma-kepi", "racemic-epi", "racemicepi"],
    # home / maintenance meds. Keys are the generic name the lexicon uses, so "aliquis"
    # becomes apixaban, as furosemide and the others do; the assumption note keeps
    # what was heard ("Apixaban (assumed from 'aliquis')").
    "donepezil":   ["donepizil", "donepazil", "donapazil", "donipizil"],
    "apixaban":    ["aliquis", "eliguis", "elequis", "eloquis", "eliquist", "aliquist"],
    "lisinopril": ["lysonoprol"],
    "rivaroxaban": ["rivaroxiban"],
    "atorvastatin": ["atorvastatan", "atorvastin"],
}

# =============================================================================
# DATA MODEL
# =============================================================================
@dataclass
class _Target:
    surface: str
    canonical: str
    category: str
    metaphone: Optional[str] = None

@dataclass
class Correction:
    """A change that was applied to the transcript."""
    original: str
    replacement: str
    canonical: str
    category: str
    score: float

@dataclass
class Flag:
    """A suspect left in place for the LLM to resolve from context.

    NOTE: flags are inert on their own. To have any effect they must be passed
    through ``format_suspects_for_llm`` and injected into the extraction prompt
    by the caller (see that function's docstring). If nothing consumes
    ``CorrectionResult.flagged`` the whole flagging path is dead weight.
    """
    token: str
    best_guess: str
    canonical: str
    category: str
    score: float
    reason: str

@dataclass
class CorrectionResult:
    text: str
    corrections: List[Correction] = field(default_factory=list)
    flagged: List[Flag] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.corrections)

# =============================================================================
# CONTEXT RULES
# =============================================================================
# Explicit, auditable fixes for the confusions the fuzzy layer deliberately
# refuses to touch: short alphanumeric abbreviations, and real-word / phrase
# mishears that only context disambiguates. Every rule is narrow and every
# firing is logged, so nothing changes unless a human wrote a rule for it.
#
# Each rule fires ONLY in a context that makes the intended term near-certain:
#   * "SbO2" / "SP02"         -> "SpO2"        (no English word looks like this)
#   * "coarse bronchi" etc.   -> "coarse rhonchi"  (a *sound* adjective, not anatomy)
#   * "POST form"             -> "POLST form"  (advance-directive phrasing)
#   * "glow-by oxygen"        -> "blow-by oxygen" (no glow in an ambulance; heard on CS014)
#
# To teach it a new confusion, append a (name, pattern, replacement) row. The
# replacement may use backreferences to preserve the surrounding real words.
_CONTEXT_RULES: List[Tuple[str, "re.Pattern", str]] = [
    ("SpO2 abbreviation",
     re.compile(r"(?i)\bs[bp][o0]2\b"),
     "SpO2"),
    ("rhonchi (lung-sound context)",
     re.compile(r"(?i)\b(coarse|fine|scattered|diffuse|expiratory|inspiratory)"
                r"(\s+)bronchi\b"),
     r"\1\2rhonchi"),
    ("POLST form",
     re.compile(r"(?i)\bpost(\s+form)\b"),
     r"POLST\1"),
    ("blow-by oxygen",
     re.compile(r"(?i)\bglow([- ]by)\b"),
     r"blow\1"),
]

def _apply_context_rules(text: str) -> Tuple[str, List["Correction"]]:
    """Run the explicit rule layer. Returns (new_text, corrections_applied)."""
    applied: List[Correction] = []

    def _make_sub(name: str, repl: str):
        def _sub(m: "re.Match") -> str:
            new = m.expand(repl)
            if new == m.group(0): # already correct
                return m.group(0)
            applied.append(Correction(
                original=m.group(0), replacement=new, canonical=new,
                category=name, score=1.0))
            return new
        return _sub

    for name, rx, repl in _CONTEXT_RULES:
        text = rx.sub(_make_sub(name, repl), text)
    return text, applied

# =============================================================================
# CORRECTOR
# =============================================================================
class TranscriptCorrector:
    """Alias-aware, precision-first corrector built once over LEXICON."""

    def __init__(self) -> None:
        self._targets: List[_Target] = []
        self._all_surfaces: set = set() # every surface, any length -> "known good"
        self._build()
        self._auto = [t for t in self._targets
                      if t.category in AUTOCORRECT_CATEGORIES]
        # invert {canonical: [variants]} -> {variant: canonical}
        self._misspellings: Dict[str, str] = {}
        for canonical, variants in KNOWN_MISSPELLINGS.items():
            for v in variants:
                self._misspellings[v.lower()] = canonical
        logger.info(
            "TranscriptCorrector ready: %d single-word targets "
            "(%d auto-correctable), phonetic=%s, spellcheck=%s",
            len(self._targets), len(self._auto), _HAVE_PHONETIC, _HAVE_SPELL,
        )

    def _build(self) -> None:
        seen = set()
        for category, terms in LEXICON.items():
            for canonical, aliases in terms.items():
                for surface in (canonical, *aliases):
                    low = surface.lower()
                    self._all_surfaces.add(low)
                    # only single alpha words qualify as correction targets
                    if not _TARGET_RE.match(low) or len(low) < MIN_TOKEN_LEN:
                        continue
                    key = (low, canonical, category)
                    if key in seen:
                        continue
                    seen.add(key)
                    self._targets.append(_Target(
                        surface=low,
                        canonical=canonical,
                        category=category,
                        metaphone=(jellyfish.metaphone(low)
                                   if _HAVE_PHONETIC else None),
                    ))

    # Scoring
    @staticmethod
    def _phonetic_bonus(token: str, target_meta: Optional[str], ratio: float) -> float:
        # gate on a decent string ratio so phonetics can't rescue pure noise
        if not _HAVE_PHONETIC or target_meta is None or ratio < 0.60:
            return 0.0
        return PHONETIC_BONUS if jellyfish.metaphone(token) == target_meta else 0.0

    def _score(self, token: str, target: _Target) -> float:
        ratio = SequenceMatcher(None, token, target.surface).ratio()
        return min(1.0, ratio + self._phonetic_bonus(token, target.metaphone, ratio))

    def _best(self, token: str, pool: List[_Target]) -> Tuple[Optional[_Target], float, float]:
        best, best_score, second = None, 0.0, 0.0
        for t in pool:
            s = self._score(token, t)
            if s > best_score:
                best, best_score, second = t, s, best_score
            elif s > second:
                second = s
        return best, best_score, second

    @staticmethod
    def _is_real_word(token: str) -> bool:
        if not _HAVE_SPELL:
            return False
        return bool(_SPELL.known([token]))

    @staticmethod
    def _match_case(src: str, repl: str) -> str:
        if src.isupper():
            return repl.upper()
        if src[:1].isupper():
            return repl[:1].upper() + repl[1:]
        return repl

    def correct(self, text: str) -> CorrectionResult:
        if not text or not text.strip():
            return CorrectionResult(text=text or "")

        # explicit context rules run first, on the raw text
        text, rule_hits = _apply_context_rules(text)
        result = CorrectionResult(text="")
        result.corrections.extend(rule_hits)
        out: List[str] = []

        for tok in _TOKEN_RE.findall(text):
            if not tok[:1].isalpha(): 
                out.append(tok)
                continue

            low = tok.lower()

            # exact known-misspelling -> canonical: context-free, zero-risk, and takes priority over every heuristic below
            canonical = self._misspellings.get(low)
            if canonical:
                repl = self._match_case(tok, canonical)
                out.append(repl)
                result.corrections.append(Correction(
                    original=tok, replacement=repl, canonical=canonical,
                    category="known_misspelling", score=1.0))
                continue

            if (len(low) < MIN_TOKEN_LEN
                    or low in self._all_surfaces
                    or low in NEVER_CORRECT):
                out.append(tok)                # too short, already correct, or stoplisted
                continue

            real = self._is_real_word(low)

            # auto-correct: non-word near-miss of a drug/med name, requires the spell gate 
            if _HAVE_SPELL and not real:
                cand, score, second = self._best(low, self._auto)
                if (cand and score >= AUTOCORRECT_THRESHOLD
                        and (score - second) >= MIN_MARGIN):
                    repl = self._match_case(tok, cand.canonical)
                    out.append(repl)
                    result.corrections.append(Correction(
                        original=tok, replacement=repl, canonical=cand.canonical,
                        category=cand.category, score=round(score, 3)))
                    continue

            # otherwise flag as a suspect for the LLM, text unchanged
            cand, score, _ = self._best(low, self._targets)
            if cand and score >= FLAG_THRESHOLD and low != cand.surface:
                result.flagged.append(Flag(
                    token=tok, best_guess=cand.surface, canonical=cand.canonical,
                    category=cand.category, score=round(score, 3),
                    reason="real-word ambiguity" if real else "low-confidence match"))
            out.append(tok)

        result.text = "".join(out)

        for c in result.corrections:
            logger.info("ASR correction: %r -> %r (%.2f, %s)",
                        c.original, c.replacement, c.score, c.canonical)
        if result.flagged:
            logger.info("%d transcription suspect(s) flagged for LLM review",
                        len(result.flagged))
        return result

# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================
_corrector: Optional[TranscriptCorrector] = None

def correct_transcript(text: str) -> CorrectionResult:
    """Correct a transcript using a lazily-built shared corrector."""
    global _corrector
    if _corrector is None:
        _corrector = TranscriptCorrector()
    return _corrector.correct(text)

def assumption_notes(result: CorrectionResult) -> List[str]:
    """
    Human-readable provenance for every applied correction, e.g.
    ``"Apixaban (assumed from 'aliquis')"``. Feed these straight into the
    ``data_quality.assumptions`` list so the receiving ED can see what was
    inferred rather than heard.
    """
    notes = []
    for c in result.corrections:
        rep = c.replacement[:1].upper() + c.replacement[1:] if c.replacement else c.replacement
        notes.append(f"{rep} (assumed from '{c.original}')")
    return notes

def format_suspects_for_llm(flagged: List[Flag]) -> str:
    """
    Render flagged suspects as a prompt block for the extraction LLM. Returns
    "" when nothing was flagged, so the prompt degrades cleanly.

    IMPORTANT: this is the only thing that makes ``CorrectionResult.flagged``
    do anything. The caller must take ``result.flagged`` from
    ``correct_transcript(...)``, pass it here, and concatenate the returned
    block into the extraction prompt. See the wiring note in the review.
    """
    if not flagged:
        return ""
    lines = ["TRANSCRIPTION SUSPECTS (possible mis-hearings -- "
             "correct from clinical context only if warranted):"]
    for f in flagged:
        lines.append(f'  "{f.token}" may be "{f.best_guess}" '
                     f"({f.canonical}) [{f.category}]")
    return "\n".join(lines) + "\n"