"""
Retrieves the subset of the medical lexicon that is actually relevant to a
given EMS report, and formats it for injection into the extraction prompt.

    lexicon side:   lexicon.py (data)      -> lexicon_rag.py (LexiconStore, whisper_hint)

The lexicon layer runs BEFORE/DURING extraction (terminology for Whisper and the
extraction prompt). 
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .lexicon import LEXICON, CLINICAL_NOTES

logger = logging.getLogger(__name__)

# Aliases too short or too common to match safely as standalone tokens
_ALIAS_STOPLIST = {"t", "in", "sc", "cc", "hf", "mi", "od", "dm", "pe", "pr", "ns"}

# Categories whose hits matter most when trimming to top_k
_CATEGORY_PRIORITY = {
    "alert_types": 0,
    "vitals": 1,
    "code_status": 2,
    "drugs": 3,
    "cardiac": 3,
    "neurologic": 4,
    "respiratory": 5,
    "interventions": 6,
    "conditions": 7,
    "routes": 8,
    "assessment_tools": 9,
    "anatomy": 10,
}

# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class RetrievedTerm:
    """One lexicon term found in a report."""
    canonical: str
    category: str
    matched_as: List[str] = field(default_factory=list)
    note: Optional[str] = None
    ambiguous_with: List[str] = field(default_factory=list)

    def render(self) -> str:
        """One-line rendering for prompt injection."""
        surface = ", ".join(sorted(set(self.matched_as)))
        line = f"  {surface} = {self.canonical} [{self.category}]"
        if self.ambiguous_with:
            line += f" (AMBIGUOUS: could also mean {'; '.join(self.ambiguous_with)} -- disambiguate from context)"
        if self.note:
            line += f"\n      {self.note}"
        return line

class LexiconStore:
    """Alias-aware retrieval over LEXICON."""
    def __init__(
        self,
        categories: Optional[List[str]] = None,
        client=None,
        use_embeddings: bool = False,
    ):
        """
        Args:
            categories: restrict retrieval to these lexicon categories.
                        None = all categories.
            client: OpenAI-compatible client, only used if use_embeddings.
            use_embeddings: opt-in semantic expansion. Requires a provider
                        with an embeddings endpoint (NOT Groq).
        """
        self.categories = categories
        self.client = client
        self.use_embeddings = use_embeddings and client is not None

        if use_embeddings and client is None:
            logger.warning("use_embeddings requested but no client supplied; falling back to alias matching")

        # surface form -> list of (canonical, category)
        self._surface_map: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        # (compiled pattern, surface form)
        self._patterns: List[Tuple[re.Pattern, str]] = []

        self._build()

    def _build(self) -> None:
        for category, terms in LEXICON.items():
            if self.categories and category not in self.categories:
                continue
            for canonical, aliases in terms.items():
                for surface in [canonical, *aliases]:
                    if surface.lower() in _ALIAS_STOPLIST or len(surface) < 2:
                        continue
                    pair = (canonical, category)
                    if pair not in self._surface_map[surface]:
                        self._surface_map[surface].append(pair)

        for surface in self._surface_map:
            self._patterns.append((self._compile(surface), surface))

        self._patterns.sort(key=lambda p: len(p[1]), reverse=True)

        n_ambig = sum(1 for v in self._surface_map.values() if len(v) > 1)
        logger.info(
            "LexiconStore built: %d surface forms, %d ambiguous",
            len(self._surface_map), n_ambig
        )

    @staticmethod
    def _compile(surface: str) -> re.Pattern:
        """
        Word-boundary pattern for one surface form.

        Short all-caps forms (BP, HR, IV, SL) are matched case-SENSITIVELY so
        they do not fire on ordinary lowercase prose. Longer or lowercase
        forms match case-insensitively.
        """
        escaped = re.escape(surface)
        left = r"\b" if surface[0].isalnum() else r"(?<!\S)"
        right = r"\b" if surface[-1].isalnum() else r"(?!\S)"
        pattern = f"{left}{escaped}{right}"

        case_sensitive = len(surface) <= 3 and any(c.isupper() for c in surface)
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.compile(pattern, flags)

    def retrieve(self, ems_report: str, top_k: int = 20) -> List[RetrievedTerm]:
        """
        Find lexicon terms present in the report.

        Args:
            ems_report: raw transcript / pasted report text
            top_k: cap on returned terms (0 or None = no cap)

        Returns:
            List of RetrievedTerm, ordered by category priority.
        """
        if not ems_report or not ems_report.strip():
            return []

        found: Dict[str, RetrievedTerm] = {}

        for pattern, surface in self._patterns:
            if not pattern.search(ems_report):
                continue

            candidates = self._surface_map[surface]
            others = [c for c, _ in candidates]

            for canonical, category in candidates:
                entry = found.get(canonical)
                if entry is None:
                    entry = RetrievedTerm(
                        canonical=canonical,
                        category=category,
                        note=CLINICAL_NOTES.get(canonical),
                    )
                    found[canonical] = entry
                entry.matched_as.append(surface)
                for other in others:
                    if other != canonical and other not in entry.ambiguous_with:
                        entry.ambiguous_with.append(other)

        results = sorted(
            found.values(),
            key=lambda e: (_CATEGORY_PRIORITY.get(e.category, 99), e.canonical),
        )
        if top_k:
            results = results[:top_k]

        logger.info("Retrieved %d lexicon terms from report", len(results))
        return results

    def to_prompt_context(self, terms: List[RetrievedTerm]) -> str:
        """
        Render retrieved terms as a prompt block. Returns "" when nothing was
        found, so the prompt degrades cleanly to its original form.
        """
        if not terms:
            return ""

        by_category: Dict[str, List[RetrievedTerm]] = defaultdict(list)
        for term in terms:
            by_category[term.category].append(term)

        lines = [
            "MEDICAL LEXICON (terms detected in this report -- use these readings):",
        ]
        for category in sorted(by_category, key=lambda c: _CATEGORY_PRIORITY.get(c, 99)):
            lines.append(f"  [{category.replace('_', ' ').upper()}]")
            for term in by_category[category]:
                lines.append(term.render())
        lines.append("")
        return "\n".join(lines)

    def build_context(self, ems_report: str, top_k: int = 20) -> str:
        """Convenience: retrieve + format in one call."""
        return self.to_prompt_context(self.retrieve(ems_report, top_k=top_k))

def whisper_hint(categories: Optional[List[str]] = None, max_chars: int = 800, vocab: str = "") -> str:
    """
    Static terminology hint for Whisper's `prompt` parameter, biasing ASR
    toward EMS abbreviations and drug names.

    This is deliberately independent of LexiconStore: transcription quality
    and extraction quality are separate concerns, so disabling lexicon
    retrieval should not degrade the audio pipeline.

    Args:
        categories: lexicon categories to draw from. Defaults to the ones
                    most prone to mis-transcription.
        max_chars: soft cap. Whisper only attends to the tail of a long
                    prompt, so keep this short.

    Returns:
        "The following is a paramedic radio report. Terminology: BP, HR, ..."
    """
    if categories is None:
        categories = ["vitals", "alert_types", "cardiac", "routes", "drugs"]

    surfaces: List[str] = []
    for category in categories:
        for canonical, aliases in LEXICON.get(category, {}).items():
            surfaces.extend(aliases[:2])  
            surfaces.append(canonical)

    seen, ordered = set(), []
    for s in surfaces:
        if s.lower() not in seen:
            seen.add(s.lower())
            ordered.append(s)

    prefix = "The following is a paramedic radio report. Terminology: "
    body = ""
    tail = RADIO_PHRASES + (f" {vocab.strip().rstrip('.')}." if vocab and vocab.strip() else "")
    budget = max_chars - len(tail) - 1
    for s in ordered:
        candidate = f"{body}{s}, "
        if len(prefix) + len(candidate) > budget:
            break
        body = candidate
    # Whisper attends to the tail of the prompt: the radio phrases go last
    return prefix + body.rstrip(", ") + ". " + tail


# How the crew talks on the radio, for Whisper's prompt. "Priority 2" came out as "priority to"
# and "How do you copy?" as "Howdy, copy" on the real calls (CS008, CS014, 2026-10-25).
RADIO_PHRASES = ("Priority 1, priority 2, priority 3. How do you copy? Copy. ETA ten minutes. "
                 "12-lead unremarkable, normal sinus with PVCs. Blow-by oxygen, dexamethasone, racemic epinephrine, "
                 "nitroglycerin sublingual, aspirin 324. LAMS, Cincinnati, last known well.")
