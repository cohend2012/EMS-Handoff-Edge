"""
textbook_rag.py
===============
Semantic retrieval over the embedded-textbook VECTOR DATABASE.

    database side:  vector_store.py (store) -> textbook_rag.py (TextbookRAG)   

This layer runs AFTER extraction: it takes the structured
handoff, retrieves relevant passages from the embedded textbooks, and formats
them so an LLM can suggest courses of action grounded in real references.

The vector store (embedding + storage + cosine search) lives in vector_store.py
and is built by embed_documents.py. This module only orchestrates *querying* it:
turning a handoff into queries, merging results, and rendering a prompt block.

If no store has been built, ``TextbookRAG.available`` is False and callers skip
guidance — nothing here fails hard on a missing database.
"""

from __future__ import annotations

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

# how many chars of a retrieved chunk to show the LLM / user (keeps prompts bounded)
_CHUNK_PREVIEW_CHARS = 600


class TextbookRAG:
    """
    Query layer over the embedded textbook vector database.

    Lazily loads the store on first use, so importing this module (and running
    the pipeline without a built store) costs nothing. Check ``.available``
    before relying on it.

    Typical use::

        rag = TextbookRAG()                       # points at database/
        if rag.available:
            context = rag.guidance_context(handoff_data, top_k=6)
            # -> inject `context` into a recommendation prompt
    """

    def __init__(self, store_dir: str = "database", embedder=None):
        self.store_dir = store_dir
        self._embedder = embedder     
        self._store = None            

    @property
    def available(self) -> bool:
        """True if a built store exists on disk (cheap filesystem check)."""
        from .vector_store import TextbookStore
        return TextbookStore.exists(self.store_dir)

    def _get_store(self):
        if self._store is None:
            from .vector_store import TextbookStore
            self._store = TextbookStore.load(self.store_dir, embedder=self._embedder)
        return self._store

    # -- query construction ---------------------------------------------------
    @staticmethod
    def build_queries(handoff_data: Dict) -> List[str]:
        """
        Turn extracted handoff data into a small set of retrieval queries.

        One query captures the syndrome/management picture; additional queries
        target administered interventions (so the pharmacology textbook is hit
        for dosing/indications, not just the management textbook). Blank/unknown
        fields are skipped.
        """
        queries: List[str] = []

        alert = str(handoff_data.get("alert_type", "")).replace("_", " ").strip()
        cc = str(handoff_data.get("chief_complaint", "")).strip()
        onset = str(handoff_data.get("mechanism_or_onset", "")).strip()
        primary = " ".join(p for p in (
            f"Emergency management of {alert}." if alert else "",
            f"Chief complaint: {cc}." if cc and cc.lower() != "unknown" else "",
            onset if onset and onset.lower() != "unknown" else "",
        ) if p).strip()
        if primary:
            queries.append(primary)

        summary = handoff_data.get("handoff_summary") or {}
        concerns = [c for c in summary.get("key_concerns", []) if c]
        if concerns:
            queries.append("Clinical concerns: " + "; ".join(concerns))

        for iv in (handoff_data.get("ems_interventions") or [])[:3]:
            name = str(iv.get("intervention", "")).strip()
            details = str(iv.get("details", "")).strip()
            q = f"{name} {details}".strip()
            if q:
                queries.append(f"{q} — indications, dosing, and considerations")

        # de-dupe while preserving order
        seen, ordered = set(), []
        for q in queries:
            key = q.lower()
            if key not in seen:
                seen.add(key)
                ordered.append(q)
        return ordered

    # -- retrieval ------------------------------------------------------------
    def retrieve(self, queries, top_k: int = 6, per_query: int = 4):
        """
        Run each query against the store and merge results.

        A chunk retrieved by several queries keeps its best score. Returns up to
        ``top_k`` distinct chunks (list of vector_store.Retrieved), best first.
        """
        if isinstance(queries, str):
            queries = [queries]
        store = self._get_store()

        best: Dict[str, object] = {}
        for q in queries:
            for r in store.search(q, top_k=per_query):
                cid = r.chunk.id
                if cid not in best or r.score > best[cid].score:
                    best[cid] = r

        merged = sorted(best.values(), key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    # -- formatting -----------------------------------------------------------
    @staticmethod
    def to_prompt_context(retrieved) -> str:
        """
        Render retrieved chunks as a reference block for the recommendation
        prompt. Returns "" when nothing was retrieved, so callers degrade
        cleanly.
        """
        if not retrieved:
            return ""
        lines = [
            "CLINICAL REFERENCE (retrieved from EMS textbooks -- base recommendations "
            "ONLY on these excerpts and cite the source):",
        ]
        for r in retrieved:
            text = r.chunk.text.strip().replace("\n", " ")
            if len(text) > _CHUNK_PREVIEW_CHARS:
                text = text[:_CHUNK_PREVIEW_CHARS].rsplit(" ", 1)[0] + " …"
            lines.append(f"  [{r.chunk.citation()}]")
            lines.append(f"    {text}")
        lines.append("")
        return "\n".join(lines)

    def guidance_context(self, handoff_data: Dict, top_k: int = 6) -> str:
        """Convenience: build queries -> retrieve -> format, in one call."""
        queries = self.build_queries(handoff_data)
        if not queries:
            return ""
        retrieved = self.retrieve(queries, top_k=top_k)
        logger.info("Textbook RAG: %d queries -> %d reference chunks",
                    len(queries), len(retrieved))
        return self.to_prompt_context(retrieved)