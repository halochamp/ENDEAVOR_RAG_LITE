# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""Build the configured private knowledge index without starting the LLM server."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config


def main() -> int:
    config.ensure_runtime_dirs()
    import embedder
    from main import _run_build

    print(f"Knowledge directory: {config.KNOWLEDGE_DIR}")
    print(f"State directory: {config.STATE_DIR}")
    print("Loading the local embedding model if it is not cached...")
    embedder._get_model()
    _run_build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
