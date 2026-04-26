---
applyTo: "**/*.py"
description: "Always-on Python style rules for the llm-patch monorepo (SOLID, OOP, type hints, OOD). Aligned with SPEC.md."
---

# Python Style — Always On

These rules apply to every `.py` file in the workspace. They derive
from [SPEC.md](../../SPEC.md). Where this file and `SPEC.md` disagree,
`SPEC.md` wins.

## Type Hints

- All public functions, methods, and class attributes have full type
  annotations. Use `from __future__ import annotations` at the top of
  modules that need forward references.
- Prefer `collections.abc` over `typing` for ABCs (e.g.,
  `Iterable`, `Mapping`). Use `|` union syntax (Python 3.11+).
- No `Any` in public signatures unless documented and justified.

## OOP / OOD

- Use `abc.ABC` for new interfaces. Prefix interface names with `I`
  (e.g., `IWeightGenerator`).
- Prefer **composition** over inheritance. Inherit only to satisfy an
  ABC or to share invariants.
- Constructors take dependencies as parameters (DI). Don't reach out
  to module-level singletons.
- Data classes for value objects: `@dataclass(frozen=True, slots=True)`
  unless the field requires Pydantic validation; in that case use
  `pydantic.BaseModel`.

## Module Hygiene

- **No module-level side effects** — no I/O, no network, no sleep, no
  registration of singletons at import time.
- Every package has an `__init__.py` with an explicit `__all__` listing
  the public symbols (see [ADR-0003](../../docs/adr/0003-public-api-policy.md)).
- New packages ship a `py.typed` marker.

## Errors

- Derive new exception types from `llm_patch_utils.errors.LlmPatchError`
  (or one of its subclasses). Never raise bare `Exception`.
- Catch the narrowest exception possible. Don't swallow exceptions
  silently — log or re-raise.

## Naming

- Modules and packages: `snake_case`.
- Classes: `PascalCase`. Interfaces: `IPascalCase`.
- Constants: `UPPER_SNAKE_CASE`.
- Test files: `test_<unit-under-test>.py`.

## Imports

- One import per line where the import system allows.
- Group order: stdlib, third-party, first-party (handled by ruff isort).
- **No** imports from another project's internal modules — see
  [ADR-0002](../../docs/adr/0002-layered-architecture.md) and
  [ADR-0003](../../docs/adr/0003-public-api-policy.md).
