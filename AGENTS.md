# AGENTS — Root Contract

This file is the **single, authoritative agent contract** for the entire
repository. It is intentionally short. Authoritative engineering rules
live in [SPEC.md](SPEC.md). Scoped, machine-loaded rules live under
[.github/instructions/](.github/instructions/) and are applied
automatically by VS Code / Copilot via `applyTo` globs (see ADR-0009).

There are **no per-project `AGENTS.md` files** — the same checklist and
hard rules apply to every project under [projects/](projects/).

## Repository Shape

This is a **uv workspace monorepo** with a unified per-project layout:

```
projects/
  llm-patch/         engine — the generic Ingest → Compile → Attach → Use framework
  utils/             cross-project, stdlib-only utilities (llm_patch_utils)
  wiki-agent/        first downstream use-case (wiki-specialized agent)
docs/
  adr/               Architecture Decision Records (MADR format)
  ARCHITECTURE.md, USAGE.md, E2E_WALKTHROUGH.md
tools/               cross-project scripts (scaffolding, layering, coverage)
.github/
  instructions/      scoped agent instructions (applyTo)
  workflows/         CI / release pipelines
SPEC.md              the engineering specification (binding)
AGENTS.md            this file
```

Every `projects/<name>/` follows the same flat layout — see
[SPEC.md §1.1 Project Shape](SPEC.md):

```
projects/<name>/
  pyproject.toml      # name = "llm-patch[-<usecase>]"
  README.md
  CHANGELOG.md
  src/<import_pkg>/   # llm_patch[_<usecase>]
  tests/
```

## Mandatory Pre-Change Checklist

Before changing any file, the agent must:

1. **Read this file** and the relevant scoped instructions under
   [.github/instructions/](.github/instructions/) (engine boundary,
   utils boundary, use-case boundary, python style, tests).
2. **Re-read the relevant section of [SPEC.md](SPEC.md)** — especially
   Dependency Direction (§1) and Public API Stability (§3).
3. **Write or update tests first** — the project follows TDD for new
   public API. See SPEC §Testing.
4. **Run `uv run --package <project> pytest -q`** for the affected
   project(s) and confirm baselines hold:
   - `llm-patch` engine: `464 passed, 12 skipped`.
   - `llm-patch-utils`, `llm-patch-wiki-agent`: smoke tests pass.
5. **Add an ADR** under [docs/adr/](docs/adr/) for any cross-cutting,
   architectural, or dependency-direction-affecting decision. Use
   [docs/adr/0000-template.md](docs/adr/0000-template.md).

## Hard Rules (do not violate without an ADR)

- **Dependency direction is one-way**: use-cases → engine → utils.
  No reverse imports. Enforced by [tools/check_layering.py](tools/check_layering.py).
- **Public API is sacred**: only symbols re-exported from a project's
  top-level `__init__.py` are public. Use-cases must consume only the
  public API of the engine.
- **No behavior changes alongside structural changes** in the same PR.
- **No module-level side effects** (no I/O, no network) at import time.
- **No per-project `AGENTS.md`**: scoped rules belong in
  `.github/instructions/*.instructions.md` with an `applyTo` glob.

## Adding a New Use-Case Project

```pwsh
uv run python tools/scaffold_project.py <name>
```

This materializes the unified layout above. The new project consumes
the engine via its public API and may depend on `llm-patch-utils`. See
[SPEC.md §9](SPEC.md) and use [projects/wiki-agent/](projects/wiki-agent/)
as the reference implementation.

## Where to Look for What

| If you want to… | Read… |
|---|---|
| Understand the system architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Understand the engineering rules | [SPEC.md](SPEC.md) |
| Add a new use-case project | [tools/scaffold_project.py](tools/scaffold_project.py) + [SPEC.md §9](SPEC.md) |
| Change the engine | [.github/instructions/engine-boundary.instructions.md](.github/instructions/engine-boundary.instructions.md) |
| Change utilities | [.github/instructions/utils-boundary.instructions.md](.github/instructions/utils-boundary.instructions.md) |
| Change a use-case | [.github/instructions/usecase-boundary.instructions.md](.github/instructions/usecase-boundary.instructions.md) |
| Write Python | [.github/instructions/python-style.instructions.md](.github/instructions/python-style.instructions.md) |
| Write tests | [.github/instructions/tests.instructions.md](.github/instructions/tests.instructions.md) |
| See past architectural decisions | [docs/adr/README.md](docs/adr/README.md) |
