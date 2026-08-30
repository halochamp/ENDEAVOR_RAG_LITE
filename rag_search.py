# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

from __future__ import annotations
from pathlib import Path

from langchain_core.tools import tool

from config import MEMORY_PATH, source_path
import retriever
import kb_operations
from expander import expand
from llm_client import chat as _llm_chat

MAX_CHUNKS    = 5   # child chunks sent to reranker
KEEP_CHUNKS   = 3   # parent chunks returned
QUALITY_HIGH  = 0.62
QUALITY_LOW   = 0.35

def _to_abs(s: str) -> str:
    """Resolve and validate a source path inside the knowledge root."""
    return str(source_path(s))


def _rerank(query: str, chunks: list[dict]) -> list[dict]:
    """Ask LLM to select best 2-3 chunks by index. Returns selected chunks."""
    if not chunks:
        return []
    if len(chunks) <= 2:
        return chunks

    numbered = "\n\n".join(
        f"[{i+1}] {c['child_text'][:400]}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"Query: {query}\n\n{numbered}\n\n"
        f"Which 2-3 chunks most directly answer the query? "
        f"Reply with numbers only, e.g. '1, 3'"
    )
    try:
        raw = _llm_chat(prompt, temperature=0.0, max_tokens=16)
        idxs = [int(x.strip()) - 1 for x in raw.replace(",", " ").split()
                if x.strip().isdigit()]
        selected = [chunks[i] for i in idxs if 0 <= i < len(chunks)]
        if selected:
            return selected[:KEEP_CHUNKS]
    except Exception:
        pass
    return chunks[:KEEP_CHUNKS]  # fallback: top by RRF


def _compute_quality(chunks: list[dict]) -> str:
    if not chunks:
        return "low"
    first = chunks[0]
    top = first.get("dense_score")
    if top is None and first.get("retriever") in {"dense", "hybrid"}:
        top = first.get("score", 0.0)
    lexical_hits = int(first.get("bm25_hits", 0))
    n = len(chunks)
    if top is None:
        return "medium" if lexical_hits else "low"
    if top > 0.78 and n >= 3 and lexical_hits:
        return "high"
    if top < QUALITY_LOW and not lexical_hits:
        return "low"
    if top > QUALITY_HIGH or n >= 2:
        return "medium"
    return "medium"


def _format_result(chunks: list[dict], quality: str) -> str:
    sources = list(dict.fromkeys(c["source"] for c in chunks))
    sources_line = "SOURCES: " + " | ".join(
        _to_abs(s) for s in sources
    )
    body = "\n\n".join(
        f"[{i+1}] (source: {Path(c['source']).name} | {c.get('retriever','?')}={c.get('score', 0):.3f})\n{c['parent_text']}"
        for i, c in enumerate(chunks)
    )
    prefix = "[low_quality]\n" if quality == "low" else ""
    return prefix + sources_line + "\n" + body


_SAMPLE_PER_ROUND = 30
_FILES_PER_ROUND  = 200
_MAX_ROUNDS       = 5


def _summarize_batch(names: str, batch_size: int, total: int, round_info: str) -> str:
    prompt = (
        f"ชื่อไฟล์ {batch_size} ไฟล์ ({round_info}):\n{names}\n\n"
        f"ระบุหมวดหมู่ความรู้ที่พบ (bullet points สั้นๆ ภาษาไทย ไม่เกิน 5 ข้อ)"
    )
    return _llm_chat(prompt, temperature=0.0, max_tokens=200)


@tool
def list_knowledge() -> str:
    """Summarize what topics are stored in the knowledge base.

    Use when the user asks what topics, documents, or files are in the knowledge base.
    Scales sampling rounds by file count, then aggregates summaries. No query needed.
    """
    try:
        import math
        import random

        paths = kb_operations.registered_paths()
        if not paths:
            return "ยังไม่มีไฟล์ใน knowledge base ครับ"

        total   = len(paths)
        rounds  = min(_MAX_ROUNDS, math.ceil(total / _FILES_PER_ROUND))
        n_sample = min(rounds * _SAMPLE_PER_ROUND, total)
        sampled  = random.sample(paths, n_sample)
        batches  = [sampled[i * _SAMPLE_PER_ROUND:(i + 1) * _SAMPLE_PER_ROUND]
                    for i in range(rounds)]

        mini_summaries = []
        for i, batch in enumerate(batches):
            names = "\n".join(f"- {Path(p).name}" for p in batch)
            round_info = f"รอบ {i+1}/{rounds} สุ่มจาก {total} ไฟล์"
            mini_summaries.append(_summarize_batch(names, len(batch), total, round_info))

        if rounds == 1:
            summary = mini_summaries[0]
        else:
            combined = "\n\n".join(f"[รอบ {i+1}]\n{s}" for i, s in enumerate(mini_summaries))
            agg_prompt = (
                f"สรุปรวมจาก {rounds} รอบการสุ่มตรวจไฟล์ใน knowledge base:\n\n{combined}\n\n"
                f"สรุปภาพรวมว่ามีความรู้เกี่ยวกับอะไรบ้าง (3-6 bullet points ภาษาไทย)"
            )
            summary = _llm_chat(agg_prompt, temperature=0.1, max_tokens=300)

        return (f"พบไฟล์ทั้งหมด {total} ไฟล์ "
                f"(สุ่มตรวจ {n_sample} ไฟล์ใน {rounds} รอบ) "
                f"เจอความรู้ประมาณนี้:\n{summary}")
    except Exception as e:
        return f"[error] {e}"


_MAX_SHOW_FILES = 30


@tool
def search_files(query: str) -> str:
    """Search for files in the knowledge base whose filename contains the query keyword.

    Use when the user asks to find files by name or topic keyword (e.g. 'หาไฟล์ที่มีคำว่า X').
    Returns up to 30 matching filenames. Shows remaining count if more exist.
    """
    try:
        paths = kb_operations.registered_paths()
        if not paths:
            return "ยังไม่มีไฟล์ใน knowledge base ครับ"
        matched = kb_operations.search_registered_paths(query)
        if not matched:
            return f"ไม่พบไฟล์ที่มีคำว่า '{query}' ในชื่อไฟล์"
        total = len(matched)
        lines = "\n".join(f"  - {Path(p).name}" for p in matched[:_MAX_SHOW_FILES])
        suffix = f"\n  ... ยังมีอีก {total - _MAX_SHOW_FILES} ไฟล์" if total > _MAX_SHOW_FILES else ""
        return f"พบ {total} ไฟล์ที่มีคำว่า '{query}':\n{lines}{suffix}"
    except Exception as e:
        return f"[error] {e}"


@tool
def read_file(filename: str) -> str:
    """Read and return the raw content of a file from the knowledge base.

    Use when the user says 'เปิดไฟล์ X', 'อ่านไฟล์ X', or explicitly asks to see file content.
    Accept partial filename, exact filename, or absolute path.
    If multiple files match, ask user to be more specific.
    """
    try:
        try:
            _, text = kb_operations.read_registered_file(filename)
            return text
        except kb_operations.RegisteredFileNotFound:
            return f"[error] ไม่พบไฟล์ '{filename}' ใน knowledge base"
        except kb_operations.RegisteredFileAmbiguous as exc:
            names = "\n".join(f"  - {Path(p).name}" for p in exc.matches[:10])
            more = f"\n  ... และอีก {len(exc.matches)-10} ไฟล์" if len(exc.matches) > 10 else ""
            return f"[error] พบหลายไฟล์ที่ตรงกับ '{filename}':\n{names}{more}\nกรุณาระบุชื่อให้ชัดเจนขึ้น"
    except Exception as e:
        return f"[error] {e}"


@tool
def save_memory(text: str) -> str:
    """Save a note to persistent local memory under the configured state directory.

    Use when user says 'จำไว้', 'บันทึกไว้', 'จำไว้ว่า', or 'remember'.
    IMPORTANT: pass the user's EXACT words as text — do NOT paraphrase, translate, or summarize.
    Memory is loaded automatically on every session start.
    """
    try:
        from datetime import datetime
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        entry = f"- [{timestamp}] {text.strip()}\n"
        with open(MEMORY_PATH, "a", encoding="utf-8") as f:
            f.write(entry)
        return f"บันทึกแล้วครับ: {text.strip()}"
    except Exception as e:
        return f"[error] {e}"


@tool
def rag_search(query: str) -> str:
    """Search the local knowledge base and return relevant document chunks.

    Use when the user asks about topics that may be in the knowledge base.
    Returns 2-3 parent chunks with source files.
    Returns [low_quality] prefix if results are weak — caller should retry with rephrased query.
    Returns [error] prefix on failure.
    Do NOT use for general knowledge, math, small talk, or coding questions.
    """
    try:
        import _progress as P
        P.report("🔍 ค้นหาใน knowledge base…", "📝 สร้าง Q1+Q2+Q3…")
        q1, q2, q3 = expand(query)
        queries = [q for q in [q1, q2, q3] if q.strip()]

        P.report("🔍 ค้นหาใน knowledge base…", "🔎 Dense search + BM25…")
        chunks = retriever.search(queries, top_k=10, top_fused=MAX_CHUNKS)

        P.report("🔍 ค้นหาใน knowledge base…", "⚖️  RRF fusion…")
        chunks = retriever.fetch_parents(chunks)
        if not chunks:
            return "[low_quality]\nNo results found in knowledge base."

        P.report("🔍 ค้นหาใน knowledge base…", "🎯 Rerank chunks…")
        selected = _rerank(q1, chunks)
        quality  = _compute_quality(selected)
        return _format_result(selected, quality)
    except Exception as e:
        return f"[error] {e}"
