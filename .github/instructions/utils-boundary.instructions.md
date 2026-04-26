---
applyTo: "projects/utils/**"
description: "Utils boundary rules: stdlib-only; no imports from engine or use-cases; new runtime deps require an ADR."
---

# Utils Boundary Rules — `llm-patch-utils`

These rules apply to every file under `projects/utils/`. They implement
Layer 1 (Utilities) of the architecture in
[SPEC.md §1](../../SPEC.md#1-architectural-layering),
[ADR-0002](../../docs/adr/0002-layered-architecture.md), and
[ADR-0009](../../docs/adr/0009-monorepo-structural-unification.md).

## Hard Rules

- **Stdlib only**. Adding any third-party runtime dependency requires
  an ADR.
- **Never import** from `llm_patch` or any `llm_patch_<usecase>`
  package. Dependencies flow downward only (utils ← engine ← use-cases).
- **No domain logic, no I/O at import time, no network**. This package
  hosts genuinely cross-cutting primitives only (logging adapters,
  config helpers, telemetry hooks, common error types).
- **Public API is sacred**: anything re-exported from
  `src/llm_patch_utils/__init__.py` is part of the published surface.
  Removing or renaming a public symbol requires a major version bump,
  an ADR, and a deprecation cycle.

## When Adding a New Symbol

1. Confirm two or more downstream projects actually need it. If only
   one needs it, keep the helper inside that project instead.
2. Add a unit test in `tests/`.
3. Re-export the symbol from `src/llm_patch_utils/__init__.py` and add
   it to `__all__`.
4. Bump the version in `pyproject.toml` per SemVer and add a
   `[Unreleased]` entry to `CHANGELOG.md`.

## Test

```pwsh
uv run --package llm-patch-utils pytest -q
```
