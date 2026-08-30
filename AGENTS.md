# Agent entry point

This repository is self-contained.

1. Read [`CLAUDE.md`](CLAUDE.md) for mandatory constraints.
2. Read [`AGENT.md`](AGENT.md) for the architecture and quick workflow.
3. Use [`AGENT_PROCEDURE.md`](AGENT_PROCEDURE.md) for the full repository-agent procedure.
4. Use [`README.md`](README.md), [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`SECURITY.md`](SECURITY.md) for product, contributor, and security contracts.

Preserve the Pipe A/B/C separation: Pipe C is a thin stdio MCP adapter over Pipe B and must not become a second retrieval or LLM pipeline.
