# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

from __future__ import annotations
import csv
import hashlib
import json
import logging
import re
import time
from pathlib import Path

from llm_client import chat as _llm_chat
from config import source_key

import embedder
import store
from chunker import chunk_document

log = logging.getLogger("ingestor")

SIMILARITY_REJECT = 0.95
SIMILARITY_PASS   = 0.75

_chunk_hashes: set[str] = set()


def _rel(path: Path) -> str:
    """Return a portable source key confined to the knowledge root."""
    return source_key(path)


# ── File loaders ──────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    for enc in ("utf-8", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""


def _load_pdf(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except Exception:
        return ""


def _load_csv(path: Path) -> str:
    """Load CSV: header injected once at top of each chunk block.
    Returns a text where chunks are separated by blank lines,
    each block starts with [columns: ...] followed by N rows.
    """
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            rows = list(reader)
        if not rows:
            return ""
        header_line = "[columns: " + ", ".join(rows[0]) + "]"
        data_rows   = [", ".join(row) for row in rows[1:] if any(row)]
        if not data_rows:
            return header_line

        # group rows so each block fits within CHILD_CHUNK_SIZE (400 chars)
        hlen       = len(header_line) + 1   # +1 for \n
        max_block  = max(50, 400 - hlen)    # chars available for data rows (wide headers must not go negative)
        blocks     = []
        buf:  list[str] = []
        buf_len = 0
        for row in data_rows:
            rlen = len(row) + 1             # +1 for \n
            if buf and buf_len + rlen > max_block:
                blocks.append(header_line + "\n" + "\n".join(buf))
                buf, buf_len = [], 0
            buf.append(row)
            buf_len += rlen
        if buf:
            blocks.append(header_line + "\n" + "\n".join(buf))

        # join blocks with blank line → chunker sees paragraph breaks
        return "\n\n".join(blocks)
    except Exception:
        return ""


def _collect_schema(obj, prefix: str = "", paths: set | None = None) -> set[str]:
    """Collect all unique key paths (without array indices) for schema header."""
    if paths is None:
        paths = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix} > {k}" if prefix else k
            paths.add(p)
            _collect_schema(v, p, paths)
    elif isinstance(obj, list):
        for v in obj:
            _collect_schema(v, prefix, paths)
    return paths


def _flatten_json(obj, prefix: str = "") -> str:
    lines = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            lines.append(_flatten_json(v, f"{prefix} > {k}" if prefix else k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            lines.append(_flatten_json(v, f"{prefix} > [{i}]"))
    else:
        lines.append(f"{prefix}: {obj}")
    return "\n".join(lines)


def _load_json(path: Path) -> str:
    """Load JSON: schema header injected at the start of every chunk block."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        schema      = _collect_schema(data)
        header_line = "[json schema: " + ", ".join(sorted(schema)) + "]"
        flat_lines  = _flatten_json(data).splitlines()

        hlen      = len(header_line) + 1
        max_block = max(50, 400 - hlen)  # wide schema headers must not go negative
        blocks: list[str] = []
        buf:    list[str] = []
        buf_len = 0
        for line in flat_lines:
            llen = len(line) + 1
            if buf and buf_len + llen > max_block:
                blocks.append(header_line + "\n" + "\n".join(buf))
                buf, buf_len = [], 0
            buf.append(line)
            buf_len += llen
        if buf:
            blocks.append(header_line + "\n" + "\n".join(buf))

        return "\n\n".join(blocks)
    except Exception:
        return ""


def load_file(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return _load_txt(path)
    if ext == ".pdf":
        return _load_pdf(path)
    if ext == ".csv":
        return _load_csv(path)
    if ext == ".json":
        return _load_json(path)
    return ""


# ── Chunk dedup ───────────────────────────────────────────────────────────────

def _chunk_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _llm_novelty(candidate: str, similar: str) -> bool:
    """Return True if candidate has meaningful new info."""
    try:
        prompt = (
            "Does text A contain meaningful new information not in text B?\n"
            f"A: {candidate[:400]}\nB: {similar[:400]}\n"
            "Reply YES or NO only."
        )
        answer = _llm_chat(prompt, temperature=0, max_tokens=4)
        return answer.strip().upper().startswith("Y")
    except Exception:
        # Availability failures must not silently discard a candidate chunk.
        log.warning("[ingestor] novelty check unavailable — preserving candidate")
        return True


def _should_store(child_text: str, vec: list[float]) -> bool:
    """3-gate dedup. Returns True if chunk should be stored.

    `vec` is the chunk's embedding, already computed by the caller's batch
    encode() — reusing it here avoids re-embedding every chunk a second time.
    """
    # Gate 1: exact hash
    h = _chunk_hash(child_text)
    if h in _chunk_hashes:
        return False

    # Gate 2: vector cosine
    neighbours = store.dense_search(vec, top_k=1)
    if neighbours:
        sim = neighbours[0]["score"]  # already 1 - cosine_distance
        if sim >= SIMILARITY_REJECT:
            return False
        if sim >= SIMILARITY_PASS:
            # Gate 3: LLM novelty
            if not _llm_novelty(child_text, neighbours[0]["child_text"]):
                return False

    _chunk_hashes.add(h)
    return True


# ── Main ingest ───────────────────────────────────────────────────────────────

def ingest_file(path: Path, on_progress=None) -> int:
    """Ingest a single file. Returns number of chunks stored."""
    fname = path.name
    start = time.time()

    log.info(f"[ingestor] ─── {fname} ───")
    _chunk_hashes.clear()  # Gate 1 is per-file scope — cross-file dup detection is Gate 2 (cosine vs. persisted store)

    # [1/4] load
    log.info(f"[ingestor]   [1/4] load file ({path.suffix})...")
    text = load_file(path)
    if not text.strip():
        log.warning(f"[ingestor]   {fname}: ไม่มีเนื้อหา — ข้าม")
        return 0
    log.info(f"[ingestor]   [1/4] loaded {len(text):,} chars")

    # [2/4] chunk
    log.info(f"[ingestor]   [2/4] chunking (parent/child)...")
    source = _rel(path)
    chunks = chunk_document(text)
    parents_n  = len(set(c["parent_index"] for c in chunks))
    children_n = len(chunks)
    log.info(f"[ingestor]   [2/4] parents={parents_n} children={children_n}")

    # [3/4] embed
    log.info(f"[ingestor]   [3/4] embedding {children_n} chunks...")
    child_texts = [c["child_text"] for c in chunks]
    embeddings  = embedder.encode(child_texts)
    log.info(f"[ingestor]   [3/4] embedding done")

    # [4/4] dedup + store
    log.info(f"[ingestor]   [4/4] dedup + store...")
    stored = 0
    skipped = 0
    try:
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
            if not _should_store(chunk["child_text"], vec):
                skipped += 1
                continue
            cid = store.upsert_chunk(
                child_text  = chunk["child_text"],
                parent_text = chunk["parent_text"],
                embedding   = vec,
                source      = source,
                parent_idx  = chunk["parent_index"],
                child_idx   = chunk["child_index"],
            )
            store.bm25_add(chunk["child_text"], cid, source, flush=False)
            stored += 1
            if on_progress:
                on_progress(i + 1, len(chunks))
        store.bm25_flush()
    except Exception:
        # Chroma and BM25 are separate stores. Roll back best-effort so a retry
        # cannot see its own partial Chroma rows as semantic duplicates and
        # permanently skip the missing BM25 writes.
        try:
            store.delete_by_source(source)
            store.bm25_delete_by_source(source)
        except Exception as rollback_error:
            log.error(f"[ingestor] rollback failed for {source}: {rollback_error}")
        raise

    elapsed = time.time() - start
    log.info(f"[ingestor]   [4/4] stored={stored} skipped(dedup)={skipped}")
    log.info(f"[ingestor] '{fname}' done — {elapsed:.1f}s")
    return stored


def delete_file(path: Path) -> int:
    """Remove all chunks for a file from all stores."""
    source = _rel(path)
    n = store.delete_by_source(source)
    store.bm25_delete_by_source(source)
    # Clear session hash cache so same-process re-ingest is not blocked by Gate 1
    _chunk_hashes.clear()
    return n
