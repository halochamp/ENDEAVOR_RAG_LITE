# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""Deterministic shared knowledge-base operations for Pipeline A and Pipe C.

This module intentionally has no dependency on ``main.py``, ``rag_search.py``,
``llm_client.py``, or the local model runtime. It owns the small read-only KB
primitives that both the local RAG agent and the MCP adapter can safely reuse.
All registered paths are reconstructed through ``file_registry`` and therefore
remain inside the configured knowledge root enforced by ``config.source_path``.
"""
from __future__ import annotations

from pathlib import Path

import file_registry


_SUPPORTED_READ_EXT = frozenset({".txt", ".md", ".pdf", ".csv", ".json"})


class RegisteredFileNotFound(LookupError):
    """Requested file is not present in the KB registry."""


class RegisteredFileAmbiguous(LookupError):
    """A partial filename matched more than one registered KB file."""

    def __init__(self, query: str, matches: list[str]):
        super().__init__(query)
        self.query = query
        self.matches = tuple(matches)


def registered_paths() -> list[str]:
    """Return registered KB paths in deterministic filename/path order."""
    paths = file_registry.all_registered()
    return sorted(paths, key=lambda value: (Path(value).name.casefold(), value.casefold()))


def search_registered_paths(query: str) -> list[str]:
    """Return registered paths whose filename contains ``query`` case-insensitively."""
    needle = query.strip().casefold()
    if not needle:
        return []
    return [path for path in registered_paths() if needle in Path(path).name.casefold()]


def resolve_registered_path(filename: str) -> Path:
    """Resolve an exact path/name or unique partial filename inside the registry only."""
    requested = filename.strip()
    if not requested:
        raise RegisteredFileNotFound(filename)

    paths = registered_paths()
    for raw in paths:
        path = Path(raw)
        if str(path) == requested or path.name == requested:
            return path

    matches = [raw for raw in paths if requested.casefold() in Path(raw).name.casefold()]
    if not matches:
        raise RegisteredFileNotFound(requested)
    if len(matches) > 1:
        raise RegisteredFileAmbiguous(requested, matches)
    return Path(matches[0])


def _read_pdf(path: Path) -> str:
    import pypdf

    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_registered_file(filename: str) -> tuple[Path, str]:
    """Read one registered KB document without allowing arbitrary filesystem paths."""
    path = resolve_registered_path(filename)
    if not path.is_file():
        raise FileNotFoundError(f"registered file is unavailable: {path.name}")
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_READ_EXT:
        raise ValueError(f"unsupported registered file type: {ext or '(none)'}")
    if ext == ".pdf":
        text = _read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    return path, text


def health_snapshot() -> dict[str, object]:
    """Return shared Chroma/BM25/registry health signals without mutating the KB."""
    import store

    issues = list(store.health_check())
    ghosts = list(file_registry.ghost_files())
    return {
        "issues": issues,
        "ghost_files": ghosts,
    }
