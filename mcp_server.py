#!/usr/bin/env python3
"""Pipe C: a small stdio MCP adapter for the public ENDEAVOR_RAG release.

The dependency direction is intentionally one-way::

    MCP client -> Pipe C -> Pipe B (rag_retrieve) -> retriever/index

Pipe C is only a protocol boundary.  It never imports ``main.py``,
``rag_search.py``, or ``llm_client.py`` and therefore never starts or calls the
standalone Pipeline A model.  Pipe B remains the single implementation of the
external-agent retrieval contract.
"""

from __future__ import annotations

import datetime as _datetime
import threading
from typing import TypeAlias

from mcp.server.fastmcp import FastMCP

from rag_retrieve import rag_retrieve as _pipe_b_rag_retrieve


Query: TypeAlias = str | list[str]

_ALLOWED_MODES = frozenset({"chunks", "files", "source_first"})
_MAX_QUERY_VARIANTS = 8
_MAX_QUERY_CHARS = 4_000
_MAX_FILTER_CHARS = 256
_MAX_OUTPUT_CHARS = 50_000

# Chroma/BM25 clients are shared by Pipe B.  Keep calls serialized until a
# dedicated concurrent-read test proves that every backend is safe to share.
_RETRIEVE_LOCK = threading.Lock()


mcp = FastMCP(
    "ENDEAVOR_RAG Pipe C",
    instructions=(
        "Pipe C is a local, read-only MCP adapter. It delegates the rag_retrieve "
        "tool to Pipe B and never starts the standalone Pipeline A LLM."
    ),
)


def _bounded_text(name: str, value: object, *, limit: int, allow_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    text = value.strip()
    if not allow_empty and not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"{name} must be <= {limit} characters")
    return text


def _validate_query(query: object) -> Query:
    if isinstance(query, str):
        return _bounded_text("query", query, limit=_MAX_QUERY_CHARS, allow_empty=False)
    if not isinstance(query, list):
        raise ValueError("query must be a string or an array of strings")
    if not query:
        raise ValueError("query must contain at least one string")
    if len(query) > _MAX_QUERY_VARIANTS:
        raise ValueError(f"query may contain at most {_MAX_QUERY_VARIANTS} variants")
    normalized: list[str] = []
    for index, item in enumerate(query):
        normalized.append(
            _bounded_text(f"query[{index}]", item, limit=_MAX_QUERY_CHARS, allow_empty=False)
        )
    return normalized


def _validate_date(name: str, value: object) -> str:
    text = _bounded_text(name, value, limit=10)
    if not text:
        return ""
    try:
        parsed = _datetime.date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"{name} must use YYYY-MM-DD") from None
    if parsed.isoformat() != text:
        raise ValueError(f"{name} must use YYYY-MM-DD")
    return text


def _validate_request(
    query: object,
    mode: object,
    tags: object,
    filename_contains: object,
    created_after: object,
    created_before: object,
    source_type: object,
) -> dict[str, object]:
    validated_query = _validate_query(query)
    normalized_mode = _bounded_text("mode", mode, limit=32).lower()
    if normalized_mode not in _ALLOWED_MODES:
        allowed = ", ".join(sorted(_ALLOWED_MODES))
        raise ValueError(f"mode must be one of: {allowed}")

    after = _validate_date("created_after", created_after)
    before = _validate_date("created_before", created_before)
    if after and before and after > before:
        raise ValueError("created_after must be <= created_before")

    return {
        "query": validated_query,
        "mode": normalized_mode,
        "tags": _bounded_text("tags", tags, limit=_MAX_FILTER_CHARS),
        "filename_contains": _bounded_text(
            "filename_contains", filename_contains, limit=_MAX_FILTER_CHARS
        ),
        "created_after": after,
        "created_before": before,
        "source_type": _bounded_text("source_type", source_type, limit=_MAX_FILTER_CHARS),
    }


def _cap_output(result: str) -> str:
    cap = _MAX_OUTPUT_CHARS
    if cap <= 0:
        return ""
    if len(result) <= cap:
        return result
    marker = (
        f"\n\n[truncated] Pipe C output is capped at {cap} characters; "
        "use narrower filters or a more specific query."
    )
    if len(marker) >= cap:
        return marker[:cap]
    return result[: cap - len(marker)] + marker


def _invoke_pipe_b(arguments: dict[str, object]) -> object:
    """Call Pipe B's LangChain wrapper in one patchable boundary."""
    return _pipe_b_rag_retrieve.invoke(arguments)


def _call_pipe_b(arguments: dict[str, object]) -> str:
    """Delegate one validated request to Pipe B while preserving its contract."""
    with _RETRIEVE_LOCK:
        try:
            result = _invoke_pipe_b(arguments)
        except Exception as exc:
            # Do not retain the backend exception as a chained cause: MCP hosts
            # may render exception chains/tracebacks and expose local details.
            raise RuntimeError(f"Pipe B rag_retrieve failed ({type(exc).__name__})") from None

    if not isinstance(result, str):
        raise RuntimeError("Pipe B rag_retrieve returned a non-text result")
    if result.startswith("[error]"):
        # Pipe B errors can contain backend paths or exception details.  Keep
        # MCP failure semantics without forwarding those details.
        raise RuntimeError("Pipe B rag_retrieve returned an error")
    return _cap_output(result)


@mcp.tool()
def rag_retrieve(
    query: str | list[str],
    mode: str = "chunks",
    tags: str = "",
    filename_contains: str = "",
    created_after: str = "",
    created_before: str = "",
    source_type: str = "",
) -> str:
    """Retrieve local knowledge through Pipe B.

    Pipe C validates the request, then delegates unchanged retrieval semantics
    to Pipe B's ``rag_retrieve`` implementation.  ``query`` is one question
    string or up to eight bounded variants of that same question.  ``mode`` is
    ``chunks``, ``files``, or ``source_first``.  Date filters use ``YYYY-MM-DD``.

    This is read-only.  It does not call Pipeline A's LLM, read arbitrary paths,
    or expose shell/Python/memory-write tools.
    """
    arguments = _validate_request(
        query,
        mode,
        tags,
        filename_contains,
        created_after,
        created_before,
        source_type,
    )
    return _call_pipe_b(arguments)


if __name__ == "__main__":
    mcp.run(transport="stdio")
