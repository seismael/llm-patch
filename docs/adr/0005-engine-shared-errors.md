# 0005 — Engine dependency on shared-utils for error types

- **Status**: Accepted
- **Date**: 2026-04-23
- **Deciders**: llm-patch maintainers
- **Tags**: architecture | layering | dependency | errors

## Context

The monorepo already defines `llm-patch-shared` as the home for cross-cutting
concerns such as common error types, but the engine still raised raw
third-party and built-in exceptions at several runtime boundaries. That made
use-case projects like `llm-patch-wiki-agent` responsible for translating
inconsistent engine failures into stable user-facing messages.

We need a minimal, controlled adoption of shared-utils inside the engine for
v1 so compile and chat flows can expose a coherent failure contract without
pulling broader logging or configuration abstractions into the engine.

## Decision

The engine may depend on `llm-patch-shared` for the shared exception hierarchy
only. Concrete engine integrations that cross external boundaries (model
loading, adapter attachment, generator initialization) should raise
`llm_patch_shared` exceptions rather than leaking raw dependency/import errors.

This ADR does not authorize broader engine adoption of shared-utils helpers.
Logging, telemetry, and configuration utilities remain outside the engine until
separately justified.

## Consequences

### Positive

- Use-case projects can catch one stable error hierarchy across engine and
  downstream CLI boundaries.
- The dependency remains narrow and aligned with `SPEC.md` rule R-1.4.
- v1 runtime failures become easier to document and test.

### Negative / Trade-offs

- The engine now has a direct runtime dependency on `llm-patch-shared`.
- Contributors must avoid using this ADR as cover for unrelated cross-cutting
  utility creep.

### Neutral

- Existing internal exception behavior remains largely unchanged; the shift is
  concentrated at public runtime boundaries.

## Alternatives Considered

### Alternative A — Keep engine exceptions local

Leave the engine independent of shared-utils and translate errors only in the
wiki-agent. Rejected because it would duplicate boundary handling across
use-cases and weaken the purpose of the shared layer.

### Alternative B — Move broader shared utilities into the engine now

Adopt logging and configuration helpers at the same time as shared errors.
Rejected because it expands scope beyond the v1 closeout and mixes unrelated
behavioral changes into one step.

## References

- [0002](0002-layered-architecture.md)
- [0003](0003-public-api-policy.md)
- [SPEC.md](../../SPEC.md)
