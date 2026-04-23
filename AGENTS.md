# AGENTS — Root Contract

This file is the **entry point** for any agent (human or AI) working in
this repository. It is intentionally short. Authoritative rules live in
[SPEC.md](SPEC.md). Per-project specifics live in each project's own
`AGENTS.md` under [projects/](projects/).

## Repository Shape

This is a **uv workspace monorepo**. The structure is:

```
projects/
  llm-patch/         engine — the generic Ingest → Compile → Attach → Use framework
  shared-utils/      cross-project, stdlib-only utilities
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

## Mandatory Pre-Change Checklist

Before changing any file, the agent must:

1. **Read the closest `AGENTS.md`** — the per-project file overrides /
   refines this root file. If editing a use-case, read the use-case's
   `AGENTS.md` and the engine's `AGENTS.md`.
2. **Re-read the relevant section of [SPEC.md](SPEC.md)** — especially
   Dependency Direction and Public API Stability.
3. **Write or update tests first** — the project follows TDD for new
   public API. See SPEC §Testing.
4. **Run `uv run --package <project> pytest -q`** for the affected
   project(s) and confirm baselines hold:
   - `llm-patch` engine: `216 passed, 3 skipped`
   - `llm-patch-shared`, `llm-patch-wiki-agent`: smoke tests pass.
5. **Add an ADR** under [docs/adr/](docs/adr/) for any cross-cutting,
   architectural, or dependency-direction-affecting decision. Use
   [docs/adr/0000-template.md](docs/adr/0000-template.md).

## Hard Rules (do not violate without an ADR)

- **Dependency direction is one-way**: use-cases → engine → shared-utils.
  No reverse imports. Enforced by [tools/check_layering.py](tools/check_layering.py).
- **Public API is sacred**: only symbols re-exported from a project's
  top-level `__init__.py` are public. Use-cases must consume only the
  public API of the engine.
- **No behavior changes alongside structural changes** in the same PR.
- **No module-level side effects** (no I/O, no network) at import time.

## Where to Look for What

| If you want to… | Read… |
|---|---|
| Understand the system architecture | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Understand the engineering rules | [SPEC.md](SPEC.md) |
| Add a new use-case project | [tools/scaffold_project.py](tools/scaffold_project.py) + this file |
| Change the engine | [projects/llm-patch/AGENTS.md](projects/llm-patch/AGENTS.md) |
| Change shared utilities | [projects/shared-utils/AGENTS.md](projects/shared-utils/AGENTS.md) |
| Change the wiki agent | [projects/wiki-agent/AGENTS.md](projects/wiki-agent/AGENTS.md) |
| See past architectural decisions | [docs/adr/README.md](docs/adr/README.md) |
