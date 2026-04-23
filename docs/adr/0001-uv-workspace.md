# 0001 — Adopt a uv Workspace Monorepo

- **Status**: Accepted
- **Date**: 2026-04-21
- **Tags**: architecture, packaging, tooling

## Context

The repository started as a single `llm-patch` package. We now need to
ship multiple downstream artifacts (a wiki-specialized agent, a future
web chat component, a future deployer) that all build on the same
engine. We also want shared cross-cutting utilities to live in one place
without leaking into the engine itself.

Options considered: separate git repositories, a single package with
sub-modules, or a monorepo with a workspace tool.

## Decision

Adopt a **uv workspace monorepo** rooted at the repository root, with
project members under `projects/*`. The engine becomes
`projects/llm-patch/` (Python import name unchanged: `llm_patch`).
Shared primitives live in `projects/shared-utils/`
(`llm_patch_shared`). The first downstream use-case lives in
`projects/wiki-agent/` (`llm_patch_wiki_agent`).

A single `uv.lock` at the repo root governs the whole workspace.

## Consequences

### Positive

- One clone, one `uv sync`, atomic cross-project changes.
- Shared lockfile guarantees the engine and use-cases are tested
  together against the same dependency graph.
- New use-cases scaffold with `tools/scaffold_project.py` and inherit
  the standard layout, AGENTS.md template, and CI gates automatically.

### Negative / Trade-offs

- All projects share a Python version floor (3.11+). Bumping it affects
  every project at once.
- Repo root has both a workspace `pyproject.toml` and per-project ones
  — contributors must learn which file owns which setting.

### Neutral

- Existing import path `from llm_patch import ...` is preserved.
- Existing test baseline `216 passed, 3 skipped` is preserved.

## Alternatives Considered

### Alternative A — Separate git repos

Highest isolation but breaks atomic refactors and forces version-pin
gymnastics for cross-project changes. Rejected.

### Alternative B — Keep core at root, add `projects/` for use-cases only

Less churn, but creates an asymmetric layout where the engine is
"special" and standardized layout/AGENTS rules don't apply uniformly.
Rejected — uniformity is more valuable than the small migration cost.

### Alternative C — Poetry / Hatch / Pants

Considered briefly. uv is the project's existing package manager and
ships first-class workspace support; switching would add complexity for
no clear benefit.

## References

- [uv workspaces docs](https://docs.astral.sh/uv/concepts/workspaces/)
- [SPEC.md §1 Architectural Layering](../../SPEC.md#1-architectural-layering)
