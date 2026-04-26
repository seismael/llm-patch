# 0002 — Three-Layer Architecture (use-cases → engine → utils)

- **Status**: Accepted (Layer 1 renamed `shared-utils` → `utils` / `llm_patch_shared` → `llm_patch_utils` by [ADR-0009](0009-monorepo-structural-unification.md); rule unchanged)
- **Date**: 2026-04-21
- **Tags**: architecture, layering, dependency-direction

## Context

With multiple use-cases consuming the engine, we need an unambiguous
rule for which package may import from which. Without one, use-case
glue inevitably leaks into the engine and the engine grows
use-case-specific code paths, killing reusability.

## Decision

Adopt a strict three-layer architecture with **one-way** dependency
flow:

```
use-cases  ───►  engine  ───►  shared-utils
```

- A package in layer N must not import from any package in a higher
  layer.
- Use-cases consume the engine only through symbols re-exported from
  `llm_patch.__init__` (see [ADR-0003](0003-public-api-policy.md)).
- The engine may eventually depend on `llm_patch_shared` (no such
  dependency today).
- `llm_patch_shared` depends only on the Python standard library.

Enforced mechanically by [tools/check_layering.py](../../tools/check_layering.py)
in pre-commit and CI.

### Grandfathering of legacy engine code

The engine's existing code base predates the strict ruff rules added at
the workspace root. Until a follow-up cleanup pass, the engine's
`pyproject.toml` keeps its narrower ruff ruleset; any per-file ignores
introduced for legacy files must reference this ADR in a
`# noqa: <code>  # ADR-0002` comment so they can be tracked and removed.

## Consequences

### Positive

- The engine remains a pure, reusable framework.
- Use-cases stay thin; cross-cutting concerns naturally land in `shared`.
- Layering violations fail fast at commit time, not at review time.

### Negative / Trade-offs

- Refactors that need to "punch through" layers require either an
  abstraction in the engine or an ADR exception.
- Two ruff rulesets in flight (engine grandfathered, new code strict)
  until the cleanup pass lands.

## Alternatives Considered

### Alternative A — Free-form imports, rely on review

Tried in past projects; consistently fails as the team scales. Rejected.

### Alternative B — Use `import-linter` with full contract files

More expressive but adds a dependency and YAML config. We start with
the lightweight AST scan in `tools/check_layering.py`; we can graduate
to `import-linter` if rules grow.

## References

- [SPEC.md §1 Architectural Layering](../../SPEC.md#1-architectural-layering)
- [SPEC.md §2 SOLID Per-Layer Responsibilities](../../SPEC.md#2-solid--per-layer-responsibilities)
