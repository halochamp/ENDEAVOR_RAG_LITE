# Developer: Poomwat Jarussri
# Email: champoomwat@gmail.com
# GitHub: https://github.com/halochamp

"""Read-only installation and local-runtime diagnostics."""
from __future__ import annotations

import argparse
import importlib.util
import platform
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

REQUIRED_MODULES = {
    "chromadb": "chromadb",
    "langgraph": "langgraph",
    "mlx_vlm": "mlx_vlm",
    "sentence_transformers": "sentence_transformers",
}


def _check_server() -> bool:
    try:
        with socket.create_connection((config.MLX_HOST, config.MLX_PORT), timeout=1.0):
            return True
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="probe the configured local MLX endpoint; never starts it",
    )
    args = parser.parse_args()

    failures = 0

    def report(label: str, ok: bool, detail: str) -> None:
        nonlocal failures
        print(f"{'OK' if ok else 'FAIL':4} {label}: {detail}")
        failures += not ok

    report("platform", platform.system() == "Darwin" and platform.machine() == "arm64",
           f"{platform.system()} {platform.machine()} (Apple Silicon required)")
    report("python", sys.version_info[:2] == (3, 11),
           f"{sys.version.split()[0]} (Python 3.11 required)")

    for label, module in REQUIRED_MODULES.items():
        report(f"package {label}", importlib.util.find_spec(module) is not None, module)

    report("knowledge", config.KNOWLEDGE_DIR.exists() or config.KNOWLEDGE_DIR.parent.exists(),
           str(config.KNOWLEDGE_DIR))
    if config.KNOWLEDGE_DIR.exists():
        report("knowledge writable", config.KNOWLEDGE_DIR.is_dir() and config.KNOWLEDGE_DIR.stat().st_mode & 0o200 != 0,
               str(config.KNOWLEDGE_DIR))
    else:
        print(f"INFO knowledge: will be created at {config.KNOWLEDGE_DIR}")

    if config.STATE_DIR.exists():
        report("state", config.STATE_DIR.is_dir(), str(config.STATE_DIR))
    else:
        print(f"INFO state: will be created at {config.STATE_DIR}")

    report("server python", config.MLX_SERVER_PYTHON.exists(), str(config.MLX_SERVER_PYTHON))

    if args.check_server:
        report("MLX server", _check_server(), f"{config.MLX_HOST}:{config.MLX_PORT}")
    else:
        print("INFO server: skipped (use --check-server; no process was started)")

    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
