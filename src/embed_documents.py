"""
embed_documents.py
==================
Build (and incrementally update) the textbook vector store from PDFs.

    1. Put PDFs in a folder (default: ``dataset/``).
    2. Build the store (default output: ``database/``):

           ollama pull nomic-embed-text # one-time
           python -m src.embed_documents

    3. Add more PDFs to ``dataset/`` later, then run the SAME command again.
       Only the new PDFs are embedded; files already in the store are skipped.

So ``dataset/`` holds your source PDFs and ``database/`` is the generated store
that ``textbook_rag.py`` reads at runtime. Re-running keeps them in sync.

    * default          -> incremental: embed only PDFs not already in the store
    * --rebuild        -> wipe and re-embed every PDF in the folder from scratch

Note: incremental adds new files; it does NOT remove chunks for a PDF you delete
from ``dataset/``. Use --rebuild after deleting a source to fully resync.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import re
from typing import List, Tuple

import numpy as np
from pypdf import PdfReader

from .vector_store import Chunk, OllamaEmbedder, TextbookStore, l2_normalize

logger = logging.getLogger(__name__)

# Sentence splitter: break after . ! ? when followed by whitespace + a capital
# letter or digit. Deliberately simple; textbook prose doesn't need more.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_WS = re.compile(r"\s+")

# -- boilerplate (running header / footer) detection --------------------------
# Lines matching these patterns are always junk, wherever they appear:
_URL_RE = re.compile(r"https?://", re.I)
_TS_RE = re.compile(r"\d{1,2}/\d{1,2}/\d{2,4},?\s+\d{1,2}:\d{2}\s*[AP]M", re.I)  # "8/18/26, 5:05 PM"
_PAGEMARK_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*/\s*\d+)?\s*$", re.I)       # "3", "1/6", "Page 3"
_DIGITS_RE = re.compile(r"\d+")

# Repeated edge lines: how many lines at the top/bottom of a page to consider,
# and on how many pages a (digit-normalized) line must recur to be boilerplate.
_EDGE_LINES = 3
_MIN_REPEAT_PAGES = 2
_MAX_BOILERPLATE_LEN = 200


def _normalize_line(ln: str) -> str:
    """Lowercase + collapse digit runs, so '1/6' and '2/6' hash to one footer."""
    return _DIGITS_RE.sub("#", ln.strip().lower())


def _is_pattern_junk(line: str) -> bool:
    s = line.strip()
    if not s:
        return True
    return bool(_URL_RE.search(s) or _TS_RE.search(s) or _PAGEMARK_RE.match(s))


def _find_repeated_edges(pages_lines: List[List[str]]) -> set:
    """
    Return the set of digit-normalized lines that recur at page edges.

    Only the first/last few lines of each page are candidates (running headers
    and footers live there), so body text is never mistaken for boilerplate.
    """
    from collections import Counter
    counts: Counter = Counter()
    for lines in pages_lines:
        edge = lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]
        # a set() so a line repeated within one page counts as one page
        keys = {
            _normalize_line(ln) for ln in edge
            if ln.strip() and len(ln.strip()) <= _MAX_BOILERPLATE_LEN
        }
        counts.update(keys)
    return {k for k, c in counts.items() if c >= _MIN_REPEAT_PAGES}


def _clean_page(lines: List[str], repeated: set) -> str:
    """Drop pattern-junk + repeated edge lines, then collapse whitespace."""
    kept = [
        ln for ln in lines
        if not _is_pattern_junk(ln) and _normalize_line(ln) not in repeated
    ]
    return _WS.sub(" ", " ".join(kept)).strip()


# =============================================================================
# PDF -> TEXT
# =============================================================================
def extract_pages(pdf_path: str, strip_boilerplate: bool = True) -> List[Tuple[int, str]]:
    """
    Return ``[(page_number, text), ...]`` (1-based pages, empty pages dropped).

    When ``strip_boilerplate`` is True (default), running headers/footers —
    URLs, export timestamps, page numbers, and short lines that repeat at page
    edges — are removed before chunking, so those don't pollute the embeddings.
    Pass False to keep the raw extracted text verbatim.
    """
    reader = PdfReader(pdf_path)

    raw: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as e:  # a single bad page shouldn't kill the run
            logger.warning("Failed to extract page %d of %s: %s", i, pdf_path, e)
            text = ""
        if text.strip():
            raw.append((i, text))
    if not raw:
        return []

    pages_lines = [t.split("\n") for _, t in raw]
    repeated = _find_repeated_edges(pages_lines) if strip_boilerplate else set()

    pages: List[Tuple[int, str]] = []
    for (page_no, _), lines in zip(raw, pages_lines):
        cleaned = (_clean_page(lines, repeated) if strip_boilerplate
                   else _WS.sub(" ", " ".join(lines)).strip())
        if cleaned:
            pages.append((page_no, cleaned))
    return pages


def _sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


# =============================================================================
# TEXT -> CHUNKS
# =============================================================================
def chunk_pages(
    pages: List[Tuple[int, str]],
    source: str,
    target_chars: int = 1200,
    overlap_chars: int = 200,
    min_chars: int = 100,
) -> List[Chunk]:
    """
    Pack sentences into overlapping chunks near ``target_chars``.

    Chunks respect sentence boundaries (never split mid-sentence) and carry a
    trailing overlap into the next chunk so a concept that straddles a boundary
    is still retrievable. Each chunk records the page range it spans.
    """
    stream: List[Tuple[str, int]] = []
    for page_no, page_text in pages:
        for sent in _sentences(page_text):
            stream.append((sent, page_no))

    chunks_raw: List[Tuple[str, int, int]] = []  # (text, page_start, page_end)
    cur: List[Tuple[str, int]] = []
    cur_len = 0

    def flush() -> None:
        if not cur:
            return
        text = " ".join(s for s, _ in cur).strip()
        if len(text) >= min_chars:
            pgs = [p for _, p in cur]
            chunks_raw.append((text, min(pgs), max(pgs)))

    for sent, page_no in stream:
        cur.append((sent, page_no))
        cur_len += len(sent) + 1
        if cur_len >= target_chars:
            flush()
            tail: List[Tuple[str, int]] = []
            tail_len = 0
            for item in reversed(cur):
                tail.insert(0, item)
                tail_len += len(item[0]) + 1
                if tail_len >= overlap_chars:
                    break
            cur = tail
            cur_len = sum(len(s) + 1 for s, _ in cur)
    flush()

    return [
        Chunk(id=f"{source}#{i}", text=text, source=source,
              page_start=ps, page_end=pe, index=i)
        for i, (text, ps, pe) in enumerate(chunks_raw)
    ]


# =============================================================================
# BUILD (incremental or full)
# =============================================================================
def build_store(
    pdf_dir: str,
    out_dir: str,
    model: str,
    base_url: str,
    target_chars: int,
    overlap_chars: int,
    rebuild: bool = False,
    limit: int = 0,
    strip_boilerplate: bool = True,
) -> TextbookStore:
    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        raise SystemExit(f"[X] No PDFs found in {pdf_dir!r}. "
                         f"Put your PDFs there and re-run.")

    # Load an existing store to append to (unless a full rebuild was requested).
    existing = None
    existing_sources = set()
    if TextbookStore.exists(out_dir) and not rebuild:
        existing = TextbookStore.load(out_dir)  # embedder rebuilt from stored model
        existing_sources = {c.source for c in existing.chunks}
        if model != existing.model:
            logger.warning("Store was built with %r; ignoring --model %r to keep "
                           "one embedding space.", existing.model, model)
        model = existing.model  # never mix embedding spaces
        print(f"[*] Existing database: {len(existing.chunks)} chunks "
              f"from {len(existing_sources)} file(s)")

    # Decide which PDFs are new.
    todo, skipped = [], []
    for p in pdf_paths:
        (skipped if os.path.basename(p) in existing_sources else todo).append(p)
    for p in skipped:
        print(f"[=] Already embedded, skipping: {os.path.basename(p)}  "
              f"(use --rebuild to redo)")

    if not todo:
        print("[OK] Nothing new to embed — database is already up to date.")
        return existing

    # Chunk the new PDFs.
    print(f"[*] {len(todo)} new PDF(s) to embed")
    new_chunks: List[Chunk] = []
    for path in todo:
        name = os.path.basename(path)
        print(f"[*] Reading {name} …")
        pages = extract_pages(path, strip_boilerplate=strip_boilerplate)
        chunks = chunk_pages(pages, source=name,
                             target_chars=target_chars, overlap_chars=overlap_chars)
        print(f"    {len(pages)} pages -> {len(chunks)} chunks")
        new_chunks.extend(chunks)

    if not new_chunks:
        if existing is not None:
            print("[!] New PDFs had no extractable text (scanned images? OCR needed). "
                  "Store unchanged.")
            return existing
        raise SystemExit("[X] No extractable text found. Are these scanned PDFs? "
                         "(OCR needed — pypdf only reads embedded text.)")

    if limit:
        new_chunks = new_chunks[:limit]
        print(f"[!] --limit set: embedding only the first {len(new_chunks)} new chunks")

    print(f"[*] Embedding {len(new_chunks)} new chunk(s) with '{model}' (Ollama)…")
    embedder = OllamaEmbedder(model=model, base_url=base_url)
    new_vectors = embedder.embed([c.text for c in new_chunks], show_progress=True)

    # Combine with any existing chunks.
    if existing is not None:
        all_chunks = existing.chunks + new_chunks
        all_vectors = np.vstack([existing.matrix, l2_normalize(new_vectors)])
    else:
        all_chunks = new_chunks
        all_vectors = new_vectors

    # Global positional index (ids stay "source#local", so they remain unique).
    for i, c in enumerate(all_chunks):
        c.index = i

    store = TextbookStore(all_chunks, all_vectors, model=model, embedder=embedder)
    store.save(out_dir)
    return store


# =============================================================================
# MAIN
# =============================================================================
def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Embed textbook PDFs into a vector store.")
    p.add_argument("--pdf-dir", default="dataset",
                   help="folder containing the source PDFs (default: dataset)")
    p.add_argument("--out-dir", default="database",
                   help="where to write the store (default: database)")
    p.add_argument("--model", default="nomic-embed-text",
                   help="Ollama embedding model (default: nomic-embed-text)")
    p.add_argument("--base-url", default="http://localhost:11434/v1",
                   help="Ollama OpenAI-compatible endpoint")
    p.add_argument("--chunk-size", type=int, default=1200,
                   help="target characters per chunk (default: 1200)")
    p.add_argument("--overlap", type=int, default=200,
                   help="overlap characters between chunks (default: 200)")
    p.add_argument("--rebuild", action="store_true",
                   help="wipe and re-embed every PDF (default is incremental)")
    p.add_argument("--limit", type=int, default=0,
                   help="embed only the first N new chunks (smoke test)")
    p.add_argument("--keep-boilerplate", action="store_true",
                   help="keep page headers/footers (default: strip them)")
    args = p.parse_args(argv)

    store = build_store(
        pdf_dir=args.pdf_dir,
        out_dir=args.out_dir,
        model=args.model,
        base_url=args.base_url,
        target_chars=args.chunk_size,
        overlap_chars=args.overlap,
        rebuild=args.rebuild,
        limit=args.limit,
        strip_boilerplate=not args.keep_boilerplate,
    )
    print(f"\n[OK] Database ready: {len(store.chunks)} chunks in {args.out_dir}/")
    print("     textbook_rag.py retrieves from it automatically at runtime.")


if __name__ == "__main__":
    main()