# Contributing

Use macOS Apple Silicon and Python 3.11 for development:

```bash
bash install_library/install.sh
source .venv/bin/activate
python -m pytest tests -q
```

Do not commit `.venv/`, `workspace/.rag_state/`, model caches, logs,
screenshots, or real personal documents. Keep the MLX server local-only.

Changes to configuration, path confinement, ingestion, retrieval, MCP Pipe C,
or tool behavior must include deterministic regression coverage. The Pipe C
contract test must continue to prove that only `rag_retrieve` is exposed, calls
delegate to Pipe B, invalid input is rejected before retrieval, and the stdio
handshake does not start Pipeline A. Live-model tests are optional and must be
called out separately from the normal test result.

Before opening a pull request, run the deterministic test suite, inspect
`git status --short`, and verify that no credentials or absolute personal paths
are included.
