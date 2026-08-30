# ENDEAVOR_RAG_LITE — Agent Procedure

Complete repository-agent workflow for this standalone public project.

Document roles:

- `AGENTS.md` — discovery entry point.
- `CLAUDE.md` — hard constraints.
- `AGENT.md` — quick architecture/workflow map.
- **This file** — full execution procedure.
- `README.md` / `CONTRIBUTING.md` / `SECURITY.md` — authoritative product, contributor, and security documentation.

## 1. Start-of-task procedure

Before changing code:

1. Read `CLAUDE.md` and `AGENT.md`.
2. Inspect Git status.
3. Classify the change as Pipe A, Pipe B, Pipe C, ingestion/indexing, retrieval/ranking, configuration/path confinement, UI/server, or docs.
4. Read the relevant README/CONTRIBUTING/SECURITY section.
5. State the exact invariant and smallest deterministic test that can falsify the change.

## 2. Pipe architecture

### Pipe A

Standalone interactive RAG application. It may own its own local LLM/chat flow.

### Pipe B

`rag_retrieve` is the external retrieval/tool interface and is the semantic source of truth for retrieval behavior exposed to other agents.

### Pipe C

The MCP server is a thin stdio adapter:

```text
MCP client -> Pipe C -> Pipe B (rag_retrieve) -> retriever/index
```

Do not duplicate Pipe B logic in Pipe C. Do not turn Pipe C into a second agent or retrieval pipeline.

## 3. Pipe C change procedure

For any `mcp_server.py` or MCP-visible change, verify all of these:

1. Exactly one tool is exposed: `rag_retrieve`.
2. The tool is read-only.
3. Query/mode/date/filter bounds are validated before calling retrieval.
4. Calls delegate to Pipe B rather than an alternate retrieval path.
5. Existing serialization/concurrency behavior is preserved.
6. Output is bounded/capped.
7. Pipe B errors become bounded execution errors rather than protocol corruption.
8. Starting/handshaking Pipe C does not import/start Pipe A or local answer-generation code.
9. MCP protocol stays on stdio; no public listener is introduced accidentally.

Any change to one of these requires deterministic regression coverage.

## 4. Pipe B/retrieval change procedure

When changing `rag_retrieve`, retriever, store, chunking, expansion, filters, or ranking:

- inspect all callers that rely on the returned schema;
- preserve stable field/source/path semantics unless intentionally versioned;
- validate filter/date/mode behavior independently of the LLM;
- keep result/output sizes bounded;
- test empty/no-match/error cases;
- test Unicode/Thai/English queries where relevant;
- avoid silently changing ranking semantics while claiming only a protocol change.

Retrieval bugs must be fixed in retrieval code, not hidden by prompt/LLM changes.

## 5. Ingestion/index changes

For ingestion, registry, chunking, embeddings, or state changes:

1. Keep source/workspace/state paths inside the documented configuration boundary.
2. Handle duplicate, changed, moved, and deleted files deterministically where the current design supports them.
3. Avoid indexing secrets/private documents in tests; use synthetic fixtures.
4. Preserve transactional/consistency assumptions of the existing store.
5. Add regression coverage for changed lifecycle/state behavior.
6. Do not commit generated index/state data.

## 6. Local model/network boundary

The MLX server is intended to remain local and bind to `127.0.0.1` by default.

Pipe C is intended for a trusted local MCP client over stdio.

Do not expose either surface to an untrusted/public network merely by changing a host/transport flag. A network-exposed design requires explicit authentication, authorization, request limits, privacy review, and documentation.

## 7. Path/privacy boundary

Before completion of path/config/data changes, verify:

- workspace and `.rag_state` paths remain intentional and bounded;
- credentials/private keys/private documents are not committed or echoed unnecessarily;
- source paths returned by retrieval match documented trusted-local semantics;
- logs/errors do not leak document contents beyond what is required;
- `.venv/`, state/indexes, model caches, logs, screenshots, and real personal documents remain out of Git.

## 8. Testing procedure

Standard deterministic regression suite:

```bash
python -m pytest tests -q
```

Run targeted tests first when useful, then the full deterministic suite before completion.

For Pipe C, always retain a contract test that proves:

- only `rag_retrieve` is exposed;
- valid calls delegate to Pipe B;
- invalid input is rejected before retrieval;
- stdio handshake works;
- Pipe A/LLM pipeline is not started by the MCP adapter.

Live-model tests, if used, are separate and must not substitute for deterministic retrieval/protocol tests.

## 9. Configuration changes

When changing env/config defaults:

- verify the README table remains accurate;
- preserve localhost defaults;
- preserve workspace/state derivation;
- reject invalid bounds early;
- update tests for new defaults/validation;
- do not hardcode a developer-specific absolute path.

## 10. Documentation changes

Keep document roles distinct:

- `AGENT.md` = quick architecture and rules;
- this file = detailed agent procedure;
- `CLAUDE.md` = hard constraints;
- README = user/product documentation;
- CONTRIBUTING = contributor setup/change expectations;
- SECURITY = public security boundary.

Update cross-links when headings/files change. Do not import private parent-repository workflows into this public repo.

## 11. Git/release hygiene

Before commit/push:

1. inspect status/diff;
2. stage only intended files;
3. ensure no `.venv/`, `.rag_state`, indexes, caches, logs, screenshots, credentials, private docs, or absolute personal paths are staged;
4. run applicable deterministic tests;
5. use a focused commit message;
6. never force-push unless explicitly requested and appropriate.

## 12. Completion criteria

A task is complete when applicable items hold:

- requested behavior is implemented;
- Pipe boundaries remain correct;
- retrieval/protocol behavior has deterministic coverage;
- full deterministic suite passes;
- localhost/privacy/path boundaries remain intact;
- final diff contains only intended changes;
- claims distinguish deterministic verification from optional live-model observations.

**Decision rule:** keep Pipe C thin, keep Pipe B semantic, keep retrieval deterministic, keep the network local by default.
