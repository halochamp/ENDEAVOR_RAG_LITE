#!/usr/bin/env bash
# Install ENDEAVOR_RAG into a project-local Python 3.11 environment.
# This script installs dependencies only; it never starts a server or loads a model.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_DIR/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

echo "=== ENDEAVOR_RAG installer ==="
echo
echo "[1/4] Checking platform"
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "[error] This release requires macOS because MLX uses Apple Metal."
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "[error] Apple Silicon is required; detected: $(uname -m)"
  exit 1
fi
echo "macOS Apple Silicon: OK"

echo
echo "[2/4] Checking Python 3.11"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] Could not find $PYTHON_BIN. Install Python 3.11 or set PYTHON_BIN to its full path."
  exit 1
fi
if ! "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
then
  echo "[error] $PYTHON_BIN is not Python 3.11: $($PYTHON_BIN --version)"
  exit 1
fi
echo "Using: $($PYTHON_BIN --version)"

echo
echo "[3/4] Creating project virtual environment"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  echo "Created: $VENV_DIR"
else
  echo "Reusing: $VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install --require-hashes -r "$PROJECT_DIR/requirements.txt"

echo
echo "[4/4] Installation complete"
cat <<EOF

Next steps:

  1. Put Markdown, text, PDF, CSV, or JSON files under:
       $PROJECT_DIR/workspace/knowledge/

  2. Check the installation:
       cd "$PROJECT_DIR"
       source .venv/bin/activate
       python tools/doctor.py

  3. Build the private index:
       python tools/build_index.py

  4. Start the local agent:
       python main.py

Optional manual server (Terminal 1):
  python -m mlx_vlm.server \\
    --model "\${RAGMAX_MLX_MODEL:-mlx-community/Qwen3.5-2B-OptiQ-4bit}" \\
    --host "\${RAGMAX_MLX_HOST:-127.0.0.1}" --port "\${RAGMAX_MLX_PORT:-8092}" \\
    --api-key "\${RAGMAX_MLX_API_KEY:-x}" \\
    --prefill-step-size "\${RAGMAX_MLX_PREFILL_STEP_SIZE:-512}"

The model download starts only when the MLX server is first launched.
EOF
