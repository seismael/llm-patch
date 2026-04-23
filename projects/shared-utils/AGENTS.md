# AGENTS — `llm-patch-shared`

Per-project agent contract for the shared-utils library. Read together
with the root [SPEC.md](../../SPEC.md) and root [AGENTS.md](../../AGENTS.md).

## Goal

Provide a tiny, stable, **stdlib-only** library of utilities used by two
or more projects in the workspace.

## Public API

Anything exported from `llm_patch_shared.__init__` is public. Subpackages
are public unless their `__init__.py` is empty / explicitly marked private.

## Allowed Dependencies

- Python standard library only.
- Adding any third-party runtime dependency requires an ADR.
- **Never** import from `llm_patch` or any use-case package.

## Do

- Keep modules small, single-responsibility, dependency-free.
- Add a unit test for every new public symbol.
- Bump version in `pyproject.toml` per SemVer when public API changes.

## Don't

- Don't add domain types from the engine (DocumentContext, AdapterManifest, etc.).
- Don't add I/O backends or framework integrations — those belong in the engine or a use-case.
- Don't catch broad `Exception`; use the `LlmPatchError` hierarchy.

## Test

```pwsh
uv run --package llm-patch-shared pytest -q
```
