# Security policy

Please do not publish credentials, private documents, model tokens, or
destructive proof-of-concept files in a public issue.

For a suspected vulnerability, send a private report to
`champoomwat@gmail.com` with the affected version, reproduction steps, impact,
and suggested mitigation. Do not include real credentials or personal files.

## Security boundaries

- The MLX server is intended to bind to `127.0.0.1` only.
- The indexer reads from the configured knowledge root and rejects path and
  symlink escapes.
- Runtime state and user documents are ignored by Git, but users must still
  check `git status` before committing.
- These are defense-in-depth controls, not a guarantee that untrusted files or
  prompts are safe. Review important answers and keep the local machine secure.
