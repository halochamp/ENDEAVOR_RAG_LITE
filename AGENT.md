# ENDEAVOR_RAG_LITE — Agent Overview

Quick operating map for agents working directly in this public repository.

Read order:

1. [`CLAUDE.md`](CLAUDE.md) — hard constraints.
2. **This file** — architecture and quick workflow.
3. [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) — full procedure.
4. [`README.md`](README.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), [`SECURITY.md`](SECURITY.md) — product/contributor/security contracts.

## Architecture mental model

```text
Pipe A: standalone interactive RAG app
        owns its chat/LLM experience

Pipe B: rag_retrieve
        retrieval/tool interface used by external callers

Pipe C: stdio MCP adapter
        validates + delegates to Pipe B
        no second retrieval implementation
        no Pipe A startup
        no extra LLM
```

The most important invariant is that Pipe C remains a **thin protocol adapter**.

## Start every task

Before editing:

1. inspect Git status;
2. identify which pipe is affected;
3. read the relevant README/CONTRIBUTING/SECURITY section;
4. identify the exact retrieval/protocol/path invariant;
5. run or add deterministic coverage for that boundary.

## Pipe C hard rules

Pipe C must continue to:

- expose exactly one read-only MCP tool: `rag_retrieve`;
- delegate retrieval semantics to Pipe B;
- reject invalid query/mode/date/filter bounds before retrieval;
- serialize calls where required by the existing implementation;
- cap returned output;
- map Pipe B failures into bounded execution errors;
- avoid importing or starting `main.py`, `rag_search.py`, or `llm_client.py` as part of the MCP server path;
- complete a stdio MCP handshake without starting Pipeline A.

## Retrieval and data rules

- Keep workspace/state paths bounded and configurable through the documented settings.
- Do not index or commit private documents unintentionally.
- Retrieval correctness must be tested independently of answer-generation quality.
- Do not hide a retrieval regression by tuning the LLM.
- Preserve source/path metadata semantics expected by trusted local callers.

## Network/security rule

- MLX server remains local by default (`127.0.0.1`).
- Pipe C is stdio-only and intended for a trusted local MCP host.
- Do not expose Pipe C through a public network endpoint without adding an explicit authentication and policy design.

## Testing

Standard deterministic suite:

```bash
python -m pytest tests -q
```

For Pipe C changes, the contract tests must still prove the one-tool surface, Pipe B delegation, input validation, and stdio handshake/no-Pipe-A-start behavior.

Live-model tests are optional and separate from deterministic retrieval/protocol verification.

## Git/release hygiene

Do not commit `.venv/`, `workspace/.rag_state/`, logs, model caches, screenshots, credentials, private documents, or machine-specific absolute paths.

Full workflow: [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md).
