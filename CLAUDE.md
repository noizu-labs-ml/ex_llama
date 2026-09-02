# CLAUDE.md — ex_llama

Guidance for Claude Code. Monorepo ops → `../../../../../CLAUDE.md` (trl-infra root).

## Identity

Elixir bindings for local LLM inference (llama.cpp family), part of the GenAI libs family (`ai/genai-core`, `ai/genai`). Native/NIF-heavy: expect build sensitivity to compiler/toolchain versions.

## Stack & Commands

Elixir + native deps. `mix deps.get && mix compile`; `mix test` (unit only unless local models configured); `mix format`, `mix credo`.

## Publishing

Hex-published — bump `version` + CHANGELOG on release; hex publish discipline (2FA).

## Universal Rules (compressed)

- **Trinity Protocol REQUIRED**: Orientation → Friction → Response (full text: monorepo `protocols/the-trinity-protocol.md`).
- **No shell in main thread** — delegate to taskers.
- **Worktrees**: all work on worktrees; `epic.<group>` consolidation branches off `develop`; squash-PR provenance into epics.
- MAIN checkout owns `deps/_build`; worktrees symlink deps (absolute path).
