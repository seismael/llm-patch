# 0009 — Monorepo Structural Unification

- **Status**: Accepted
- **Date**: 2026-04-26
- **Deciders**: llm-patch maintainers
- **Tags**: architecture | layering | governance | naming

## Context

After v1.0.0rc1 the workspace had drifted in three small ways:

1. **Naming inconsistency.** The shared library was named
   `shared-utils` (folder), `llm-patch-shared` (distribution),
   `llm_patch_shared` (import). Every other project uses the
   `llm-patch[-<usecase>]` / `llm_patch[_<usecase>]` family. The odd
   one out caused friction in docs, scaffolding, and onboarding.
2. **Multiple AGENTS.md files.** Every project carried its own
   `AGENTS.md` in addition to the root one. The per-project files
   mostly reiterated the root rules with minor scope notes. With
   `.github/instructions/*.instructions.md` already providing
   `applyTo`-scoped guidance to VS Code Copilot, the per-project
   AGENTS files were redundant and risked drift.
3. **Use-case template was implicit.** `wiki-agent` is the canonical
   example of a downstream project, but the contract was nowhere
   stated: a use-case must own a `projects/<name>/{src,tests,
   pyproject.toml,README.md,CHANGELOG.md}` tree, depend on the engine
   via the public API, and never reach into engine internals.

## Decision

Unify the workspace structure with three concrete changes:

1. **Rename** `projects/shared-utils/` → `projects/utils/`,
   distribution `llm-patch-shared` → `llm-patch-utils`, import
   `llm_patch_shared` → `llm_patch_utils`. Stdlib-only policy is
   unchanged.
2. **Collapse AGENTS.md.** Delete every per-project `AGENTS.md`. The
   single root `AGENTS.md` is the only human-facing agent contract.
   Scoped, machine-loaded rules continue to live in
   `.github/instructions/*.instructions.md`, which VS Code applies
   automatically via `applyTo` globs.
3. **Codify the use-case template.** Every new project under
   `projects/<name>/` follows the same flat shape as `wiki-agent`:

   ```
   projects/<name>/
     pyproject.toml      # depends on llm-patch (+ optionally llm-patch-utils)
     README.md
     CHANGELOG.md
     src/<import_pkg>/   # llm_patch_<name> by convention
     tests/
   ```

   Per-project `AGENTS.md` files are **not** generated.
   `tools/scaffold_project.py` enforces this template.

The dependency direction stays one-way:
`use-cases → engine → utils`. `tools/check_layering.py` is updated
to reflect the rename but the rule is unchanged.

## Consequences

### Positive

- One naming family across folder / dist / import; predictable for
  newcomers and tooling.
- One AGENTS.md to read; no risk of stale per-project copies.
- Use-case authors have an unambiguous template (and `wiki-agent` as
  a working reference implementation).

### Negative / Trade-offs

- One-time mass rename touching ~50 import sites and ~30 doc/tooling
  references.
- Pre-commit hook ID changes (`mypy-shared-utils` → `mypy-utils`)
  invalidate cached results; contributors must run
  `pre-commit clean` once after pulling.
- Older external references to `llm-patch-shared` / `llm_patch_shared`
  (none known publicly at v1.0.0rc1) will break.

### Neutral

- ADR-0001, ADR-0002 and ADR-0005 keep their original titles and
  filenames as historical record. A short callout at the top of each
  references this ADR for the new names.
- Engine version unchanged (`1.0.0rc1`). Utils bumps `0.1.0 → 0.2.0`
  because the distribution name changed (an observable event).

## Alternatives Considered

### Alternative A — Keep `shared-utils` as-is

Cheapest. Rejected: the inconsistency with the `llm_patch_*` family
is a recurring source of confusion in docs and CI matrices.

### Alternative B — Rename to bare `utils` (folder/dist/import)

Shortest, but `utils` as a top-level Python package shadows many
common names and risks import collisions in downstream environments.
Rejected.

### Alternative C — Fold utils into the engine namespace
(`llm_patch.utils`)

Removes one project. Rejected: blurs the layering boundary that
ADR-0002 establishes, and `llm-patch-utils` must remain installable
without `torch`/`peft`.

### Alternative D — Per-project AGENTS.md as thin pointers

Keep `projects/<name>/AGENTS.md` as a 3-line file pointing to root.
Rejected: the user explicitly asked for a single AGENTS.md, and a
pointer file adds noise without value.

### Alternative E — Backwards-compat shim (`llm_patch_shared`
re-exports from `llm_patch_utils` with `DeprecationWarning`)

Considered. Rejected: the engine just hit `1.0.0rc1`, no public
consumers exist yet, and a shim doubles maintenance for the next
release cycle.

### Alternative F — Nested `projects/usecases/<name>/`

Adds a layer that tools and docs would have to learn. Rejected;
flat `projects/<name>/` mirrors the existing wiki-agent layout and
keeps the workspace shallow.

## References

- [0001](0001-uv-workspace.md), [0002](0002-layered-architecture.md),
  [0003](0003-public-api-policy.md), [0005](0005-engine-shared-errors.md)
- [SPEC.md §Layered Architecture, §Project Shape](../../SPEC.md)
- [tools/scaffold_project.py](../../tools/scaffold_project.py)
- [tools/check_layering.py](../../tools/check_layering.py)
- [projects/wiki-agent/](../../projects/wiki-agent/) — reference
  use-case implementation.
