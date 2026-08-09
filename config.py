# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""Central runtime configuration for the standalone public release."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def _path_setting(name: str, default: Path) -> Path:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default.resolve()
    value = Path(raw).expanduser()
    return (value if value.is_absolute() else ROOT_DIR / value).resolve()


WORKSPACE_DIR = _path_setting("RAGMAX_WORKSPACE", ROOT_DIR / "workspace")
KNOWLEDGE_DIR = _path_setting("RAGMAX_KNOWLEDGE_DIR", WORKSPACE_DIR / "knowledge")
STATE_DIR = _path_setting("RAGMAX_STATE_DIR", WORKSPACE_DIR / ".rag_state")
CHROMA_DIR = STATE_DIR / "chroma"
BM25_PATH = STATE_DIR / "bm25.pkl"
REGISTRY_PATH = STATE_DIR / "file_hashes.json"
MEMORY_PATH = STATE_DIR / "memory.md"
LOG_DIR = STATE_DIR / "logs"

MLX_HOST = os.getenv("RAGMAX_MLX_HOST", "127.0.0.1")
MLX_PORT = int(os.getenv("RAGMAX_MLX_PORT", "8092"))
MLX_MODEL = os.getenv(
    "RAGMAX_MLX_MODEL", "mlx-community/Qwen3.5-2B-OptiQ-4bit"
)
MLX_API_KEY = os.getenv("RAGMAX_MLX_API_KEY", "x")
MLX_PREFILL_STEP_SIZE = int(os.getenv("RAGMAX_MLX_PREFILL_STEP_SIZE", "512"))
MLX_SERVER_PYTHON = Path(
    os.getenv("RAGMAX_MLX_PYTHON", str(ROOT_DIR / ".venv" / "bin" / "python"))
).expanduser()
if not MLX_SERVER_PYTHON.is_absolute():
    MLX_SERVER_PYTHON = ROOT_DIR / MLX_SERVER_PYTHON
MLX_SERVER_PYTHON = MLX_SERVER_PYTHON.resolve()
NO_AUTO_START = os.getenv("RAGMAX_NO_AUTO_START", "").strip().lower() in {
    "1", "true", "yes", "on"
}


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def source_key(path: str | Path) -> str:
    """Return a portable key for a file inside the configured knowledge root."""
    candidate = _resolved(path)
    try:
        return candidate.relative_to(KNOWLEDGE_DIR).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"File is outside the configured knowledge directory: {candidate}"
        ) from exc


def source_path(source: str | Path) -> Path:
    """Resolve a stored key and reject path/symlink escapes."""
    raw = Path(source).expanduser()
    candidate = _resolved(raw if raw.is_absolute() else KNOWLEDGE_DIR / raw)
    try:
        candidate.relative_to(KNOWLEDGE_DIR)
    except ValueError as exc:
        raise ValueError(
            f"Source is outside the configured knowledge directory: {candidate}"
        ) from exc
    return candidate


def ensure_runtime_dirs() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
