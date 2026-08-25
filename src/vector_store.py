"""
vector_store.py
===============
Local vector database for the EMS handoff pipeline.

Two responsibilities:

  1. **Embed text** — an :class:`OllamaEmbedder` turns text into vectors via a
     local Ollama embedding model (``nomic-embed-text`` by default). The *same*
     embedder must be used to build the store and to query it, or the vectors
     live in different spaces and similarity is meaningless. To guarantee that,
     the model name is written into ``meta.json`` and :meth:`TextbookStore.load`
     rebuilds a matching embedder automatically.

  2. **Persist + search** — :class:`TextbookStore` holds the embedded chunks and
     does brute-force cosine similarity with numpy. For a couple of textbooks
     this is a few thousand vectors; an exact search is effectively instant and
     needs no extra database dependency (Chroma / FAISS / etc.).

Storage layout (a directory)::
    database/
        embeddings.npy   float32 [N, D], L2-normalized
        chunks.json      [{id, text, source, page_start, page_end, index}, ...]
        meta.json        {model, dim, count, normalized, created_at}
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-8


# =============================================================================
# DATA MODEL
# =============================================================================
@dataclass
class Chunk:
    """One embedded slice of a source document."""
    id: str
    text: str
    source: str # filename
    page_start: int      
    page_end: int     
    index: int          

    def citation(self) -> str:
        """Human-readable provenance, e.g. ``Emergency Care (p. 211)``."""
        stem = os.path.splitext(os.path.basename(self.source))[0]
        if self.page_start == self.page_end:
            return f"{stem} (p. {self.page_start})"
        return f"{stem} (pp. {self.page_start}-{self.page_end})"


@dataclass
class Retrieved:
    """A chunk plus the cosine score it earned for a query."""
    chunk: Chunk
    score: float


# =============================================================================
# EMBEDDING BACKEND
# =============================================================================
def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization. Zero rows are left as zeros (safe for dot)."""
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        n = np.linalg.norm(mat)
        return mat / n if n > _EPS else mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > _EPS, norms, 1.0)
    return (mat / norms).astype(np.float32)


class OllamaEmbedder:
    """
    Embeds text via a local Ollama model through its OpenAI-compatible endpoint.

    Setup once::

        ollama pull nomic-embed-text

    ``nomic-embed-text`` returns 768-dim vectors and is a solid, lightweight
    default. ``mxbai-embed-large`` (1024-dim) is a heavier, higher-quality
    alternative — just pass ``model="mxbai-embed-large"``.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        api_key: str = "ollama",
        batch_size: int = 16,
    ):
        from openai import OpenAI

        self.model = model
        self.batch_size = max(1, batch_size)
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """One request for a batch, falling back to per-item on failure.

        Ollama's OpenAI-compatible ``/v1/embeddings`` accepts a list on recent
        versions; older ones only take a single string. We try the batch first
        and degrade gracefully so this works across versions.
        """
        try:
            resp = self.client.embeddings.create(model=self.model, input=list(texts))
            # data may come back out of order in theory; sort by index to be safe
            ordered = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
            return [d.embedding for d in ordered]
        except Exception as e:
            logger.debug("Batch embed failed (%s); falling back to per-item.", e)
            out: List[List[float]] = []
            for t in texts:
                resp = self.client.embeddings.create(model=self.model, input=t)
                out.append(resp.data[0].embedding)
            return out

    def embed(self, texts: Sequence[str], show_progress: bool = False) -> np.ndarray:
        """Embed a list of strings -> float32 array ``[len(texts), dim]``."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        vectors: List[List[float]] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
            if show_progress:
                done = min(start + self.batch_size, total)
                print(f"    embedding… {done}/{total}", end="\r", flush=True)
        if show_progress:
            print()  # newline after the progress line
        return np.asarray(vectors, dtype=np.float32)


# =============================================================================
# STORE
# =============================================================================
class TextbookStore:
    """An in-memory, disk-backed store of embedded chunks with cosine search."""

    EMB_FILE = "embeddings.npy"
    CHUNK_FILE = "chunks.json"
    META_FILE = "meta.json"

    def __init__(
        self,
        chunks: List[Chunk],
        embeddings: np.ndarray,
        model: str,
        embedder: Optional[object] = None,
    ):
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "length mismatch"
            )
        self.chunks = chunks
        self.matrix = l2_normalize(embeddings) # store normalized -> dot == cosine
        self.model = model
        self.embedder = embedder # only needed for querying

    # persistence  
    def save(self, store_dir: str) -> None:
        os.makedirs(store_dir, exist_ok=True)
        np.save(os.path.join(store_dir, self.EMB_FILE), self.matrix)
        with open(os.path.join(store_dir, self.CHUNK_FILE), "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in self.chunks], f, ensure_ascii=False)
        meta = {
            "model": self.model,
            "dim": int(self.matrix.shape[1]) if self.matrix.size else 0,
            "count": len(self.chunks),
            "normalized": True,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(os.path.join(store_dir, self.META_FILE), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved vector store: %d chunks -> %s", len(self.chunks), store_dir)

    @classmethod
    def exists(cls, store_dir: str) -> bool:
        return os.path.exists(os.path.join(store_dir, cls.EMB_FILE)) and \
               os.path.exists(os.path.join(store_dir, cls.CHUNK_FILE))

    @classmethod
    def load(cls, store_dir: str, embedder: Optional[object] = None) -> "TextbookStore":
        """
        Load a store from disk.

        If ``embedder`` is None, an :class:`OllamaEmbedder` is created using the
        model recorded in ``meta.json`` — so queries automatically use the same
        embedding space the store was built with. If you pass an embedder whose
        ``.model`` differs from the stored one, you get a loud warning because
        the results will be garbage.
        """
        if not cls.exists(store_dir):
            raise FileNotFoundError(
                f"No vector store at {store_dir}. Build it first with:\n"
                f"    python -m src.embed_documents --pdf-dir <your_pdfs>"
            )

        matrix = np.load(os.path.join(store_dir, cls.EMB_FILE))
        with open(os.path.join(store_dir, cls.CHUNK_FILE), "r", encoding="utf-8") as f:
            chunks = [Chunk(**c) for c in json.load(f)]
        with open(os.path.join(store_dir, cls.META_FILE), "r", encoding="utf-8") as f:
            meta = json.load(f)

        stored_model = meta.get("model", "")
        if embedder is None:
            embedder = OllamaEmbedder(model=stored_model)
        elif getattr(embedder, "model", stored_model) != stored_model:
            logger.warning(
                "Query embedder model %r != store model %r. Similarity will be "
                "meaningless; rebuild the store or match the model.",
                getattr(embedder, "model", "?"), stored_model,
            )

        store = cls.__new__(cls)         
        store.chunks = chunks
        store.matrix = matrix.astype(np.float32)
        store.model = stored_model
        store.embedder = embedder
        logger.info("Loaded vector store: %d chunks (model=%s) from %s",
                    len(chunks), stored_model, store_dir)
        return store

    # retrieval 
    def search(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[Retrieved]:
        """Return the ``top_k`` chunks most similar to ``query`` (cosine)."""
        if self.embedder is None:
            raise RuntimeError("This store has no embedder; cannot embed a query.")
        if not query or not query.strip() or self.matrix.size == 0:
            return []

        q = self.embedder.embed([query])[0]
        q = l2_normalize(q)
        sims = self.matrix @ q 

        if source_filter:
            mask = np.array([source_filter.lower() in c.source.lower()
                             for c in self.chunks])
            sims = np.where(mask, sims, -np.inf)

        k = min(top_k, len(self.chunks))
        if k <= 0:
            return []
        # argpartition for the top-k, then sort those descending
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        results = []
        for i in top_idx:
            score = float(sims[i])
            if not np.isfinite(score):
                continue
            results.append(Retrieved(chunk=self.chunks[i], score=score))
        return results
