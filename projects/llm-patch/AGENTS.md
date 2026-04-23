# AGENTS — `llm-patch` (engine)

Per-project agent contract for the **engine**. Read together with the
root [SPEC.md](../../SPEC.md) and root [AGENTS.md](../../AGENTS.md).

## Goal

Provide the generic, source-/model-/storage-agnostic
**Ingest → Compile → Attach → Use** framework. The engine is the
foundation every use-case project builds on.

## Public API

The public surface is everything re-exported from
[`src/llm_patch/__init__.py`](src/llm_patch/__init__.py). Anything not
re-exported is **internal** and may change without a major-version bump.

When adding new public symbols:

1. Add them to `__init__.py`'s explicit `__all__`.
2. Add a unit test that imports the symbol from `llm_patch` (the
   top-level), not the submodule. This locks the public path.
3. Update [`docs/USAGE.md`](../../docs/USAGE.md) and
   [`docs/ARCHITECTURE.md`](../../docs/ARCHITECTURE.md) if the addition
   changes the architecture story.

## Allowed Dependencies

- Anything declared in `pyproject.toml` `[project.dependencies]` and
  `[project.optional-dependencies]`.
- **Never** import from `llm_patch_wiki_agent` or any other use-case
  package. Dependency direction is one-way (use-cases → engine).
- May import from `llm_patch_shared` once such a dependency is added
  via ADR; today the engine has no shared-utils dependency.

## Do

- Maintain the existing 7 ABCs in `core/interfaces.py` as the single
  extension surface; new variation goes through new interfaces or
  Strategy implementations, not by editing existing ones (OCP).
- Preserve test baseline: `216 passed, 3 skipped`. Any change that
  alters that count must be justified in the PR description.
- Add new sources / generators / storages / providers / runtimes by
  implementing the corresponding ABC, never by extending an unrelated
  one.

## Don't

- Don't import use-case packages.
- Don't break legacy import shims (`KnowledgeFusionOrchestrator`,
  `WikiKnowledgeSource`) without a deprecation cycle and ADR.
- Don't introduce module-level side effects (no I/O at import time).

## Test

```pwsh
uv run --package llm-patch pytest -q
# expect: 216 passed, 3 skipped
```
