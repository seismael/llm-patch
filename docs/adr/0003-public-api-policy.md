# 0003 — Public API = Top-Level `__init__.py` Re-exports Only

- **Status**: Accepted
- **Date**: 2026-04-21
- **Tags**: api, stability, semver

## Context

Without an explicit public API contract, every internal symbol becomes
a de-facto public symbol the moment one external caller imports it.
That makes the engine impossible to refactor without breaking users.

## Decision

A symbol in any project of this monorepo is **public** if and only if
it is re-exported from that project's top-level `__init__.py` (and
listed in `__all__`). Everything else is internal and may change
without warning.

Use-cases (including the in-tree `llm_patch_wiki_agent`) must import
from the top-level package:

```python
# OK
from llm_patch import CompilePipeline, MarkdownDataSource, IDataSource

# NOT OK in a use-case
from llm_patch.pipelines.compile import CompilePipeline
from llm_patch.core.interfaces import IDataSource
```

Public API changes follow [SemVer](https://semver.org/) at the
**per-project** level (each project owns its version). Removing or
changing the signature of a public symbol requires a major bump and a
`CHANGELOG.md` entry.

## Consequences

### Positive

- The engine can refactor internals freely between major versions.
- Use-case code is forced to depend on the smallest possible surface.
- IDE autocompletion at `llm_patch.<TAB>` documents the public surface.

### Negative / Trade-offs

- Adding a new feature now requires updating `__init__.py` and `__all__`
  in addition to the implementation file.
- Internal symbols needed by tests sometimes must be imported via their
  internal path; that's acceptable because tests are not external API.

## Alternatives Considered

### Alternative A — `_private` underscore convention

Less explicit and not enforceable. Rejected.

### Alternative B — Separate `public/` subpackage

Heavier ceremony for the same outcome. Rejected.

## References

- [SPEC.md §4 Public API Stability](../../SPEC.md#4-public-api-stability)
- [PEP 8 — Public and Internal Interfaces](https://peps.python.org/pep-0008/#public-and-internal-interfaces)
