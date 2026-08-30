# ENDEAVOR_RAG_LITE project rules

- This public repository is self-contained; do not depend on a private parent repository.
- Read [`AGENT.md`](AGENT.md) and [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) before substantial work.
- Preserve the three-pipe architecture: Pipe A is the standalone RAG app, Pipe B is `rag_retrieve`, Pipe C is a thin stdio MCP adapter that delegates to Pipe B.
- Pipe C must remain read-only, expose only `rag_retrieve`, validate input before retrieval, serialize calls as designed, cap output, and never import/start Pipe A or a second LLM/retrieval implementation.
- Keep the MLX server local (`127.0.0.1`) and do not put the stdio MCP adapter behind a public network endpoint without a deliberate authentication/policy redesign.
- Preserve workspace/state confinement and do not commit `.venv/`, `workspace/.rag_state/`, model caches, logs, screenshots, credentials, private documents, or personal absolute paths.
- Changes to configuration, ingestion, retrieval, path confinement, Pipe C, or tool behavior require deterministic regression coverage.
- Standard deterministic suite: `python -m pytest tests -q`.
- Keep live-model tests separate from deterministic retrieval/protocol tests.
