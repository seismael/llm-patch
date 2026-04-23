---
applyTo: "projects/llm-patch/**"
description: "Engine boundary rules: no imports from use-cases; preserve public API; preserve test baseline 216 passed / 3 skipped."
---

# Engine Boundary Rules

These rules apply to every file under `projects/llm-patch/`. They
implement the engine layer of the architecture in
[SPEC.md §1](../../SPEC.md#1-architectural-layering) and
[ADR-0002](../../docs/adr/0002-layered-architecture.md).

## Hard Rules

- **Do not import** from `llm_patch_wiki_agent` or any other use-case
  package. Dependency direction is one-way (use-cases → engine).
- **Do not import** from `llm_patch_shared` until that dependency is
  explicitly added via ADR. Today the engine has no shared-utils dep.
- **Do not edit** existing ABCs in `core/interfaces.py` to add a new
  capability. Add a new ABC or a new Strategy implementation instead
  (Open/Closed).
- **Preserve the public API**: anything currently re-exported from
  `src/llm_patch/__init__.py` stays re-exported (additions OK, removals
  require a major bump + ADR + deprecation).
- **Preserve the test baseline**: `216 passed, 3 skipped`. Any PR that
  changes the count must justify the change in its description.

## When Adding a New Source / Generator / Storage / Provider / Runtime

1. Implement the corresponding ABC from `core/interfaces.py` in a new
   module under the appropriate subpackage (`sources/`, `generators/`,
   `storage/`, `attach/`, `runtime/`).
2. Add a unit test in `tests/unit/`.
3. If the new class is part of the public API, re-export it from
   `src/llm_patch/__init__.py` and add to `__all__`.
4. Update `docs/ARCHITECTURE.md` if the addition expands the architecture
   story.
