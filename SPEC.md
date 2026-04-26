# SPEC — Engineering Specification

> **Status**: binding. Deviations require an ADR under [docs/adr/](docs/adr/).
> **Audience**: every contributor (human or AI) to this monorepo.
> **Version**: 1.0.0 (2026-04-21).

This document is the authoritative engineering specification for the
**llm-patch** monorepo. It governs architecture, dependency direction,
SOLID/GoF expectations, TDD policy, public-API stability, versioning,
and quality gates. The root [AGENTS.md](AGENTS.md) and per-project
`AGENTS.md` files derive from this spec.

---

## 1. Architectural Layering

The repository is partitioned into three layers. Dependencies flow
**downward only**:

```
                ┌────────────────────────────────────────────┐
   Layer 3      │  Use-cases (wiki-agent, future chat-web,    │
   Use-cases    │  agent-deployer, ...)                       │
                │  → may depend on: engine + shared           │
                └─────────────────┬───────────────────────────┘
                                  │
                ┌─────────────────▼───────────────────────────┐
   Layer 2      │  Engine (llm-patch)                         │
   Engine       │  → may depend on: shared (via DI), 3rd-party│
                └─────────────────┬───────────────────────────┘
                                  │
                ┌─────────────────▼───────────────────────────┐
   Layer 1      │  Shared utilities (llm-patch-shared)        │
   Shared       │  → may depend on: stdlib only               │
                └─────────────────────────────────────────────┘
```

**Hard rules:**

- **R-1.1** A package in layer N must NOT import from any package in
  layer N+k (k > 0).
- **R-1.2** A use-case must consume the engine **only** through the
  public symbols re-exported from `llm_patch.__init__`.
- **R-1.3** Adding a runtime dependency to `llm-patch-shared` requires
  an ADR.
- **R-1.4** Cross-cutting concerns (logging, telemetry, common errors)
  belong in `llm-patch-shared`, not the engine.

Enforced by [tools/check_layering.py](tools/check_layering.py) (pre-commit + CI).

---

## 2. SOLID — Per-Layer Responsibilities

| Principle | Engine | Shared | Use-case |
|---|---|---|---|
| **SRP** — one reason to change | one ABC = one capability | one module = one utility | one class = one user-visible workflow |
| **OCP** — open/closed | extend via new `IXxx` impls; never edit existing ABCs | extend by adding modules | extend via composition of engine pieces |
| **LSP** — substitutability | every `IDataSource`/`IWeightGenerator`/etc. must satisfy the ABC contract under unit tests | n/a (mostly free functions) | n/a |
| **ISP** — interface segregation | prefer many small ABCs (current 7) over one fat one | n/a | depend on the narrowest engine ABC needed |
| **DIP** — depend on abstractions | pipelines accept ABCs, not concretes (constructor injection) | n/a | construct concretes, inject into engine pipelines |

---

## 3. Mandatory GoF Patterns Already in Use

These patterns are part of the engine's public design contract. Future
contributions must respect them.

| Pattern | Where | Notes |
|---|---|---|
| **Strategy** | `IWeightGenerator`, `IModelProvider`, `IAdapterLoader`, `IAgentRuntime` | Swap implementations by passing different concrete classes. |
| **Repository** | `IAdapterRepository` | Persistence is decoupled from domain logic. |
| **Composite** | `CompositeDataSource` | Combines multiple `IDataSource` instances under one interface. |
| **Pipeline / Composition** | `CompilePipeline`, `UsePipeline`, `WikiPipeline` | Compose the 7 ABCs into end-to-end flows. |
| **Provider / Factory** | `IModelProvider`, `HFModelProvider` | Centralizes object creation behind a stable interface. |
| **Observer** | `IKnowledgeStream.subscribe(...)` | Push-based ingestion via callback subscription. |
| **Adapter** | `MarkdownDataSource` ↔ `IDataSource`, `PeftAdapterLoader` ↔ `IAdapterLoader` | Bridges third-party APIs to engine ABCs. |
| **Template Method** | `WikiManager.ingest_*` flow | Fixed skeleton with overridable hooks. |
| **Repository (network)** | `IAdapterRegistryClient` | Same Repository pattern across the network boundary; concretes live outside the engine ([ADR-0006](docs/adr/0006-distributed-adapter-registry.md)). |
| **Decorator / Cache** | `IAdapterCache`, `LRUAdapterCache` | Bounded LRU cache layered over manifest resolution; pure-stdlib reference impl. |
| **Strategy (runtime control)** | `IRuntimeAdapterController`, `PeftRuntimeController` | Hot-swap on a live `ModelHandle`; serialized via lock per [SERVER_ARCHITECTURE.md](docs/SERVER_ARCHITECTURE.md). |

**Rule R-3.1**: Adding a new data source / generator / storage / model
provider / runtime is done by **implementing the existing ABC**, not by
modifying it. Modifying an ABC is a breaking change and requires an ADR
plus a major-version bump.

---

## 4. Public API Stability

- **R-4.1** A symbol is **public** if and only if it is re-exported from
  the project's top-level `__init__.py` (`__all__` is authoritative).
- **R-4.2** Public symbols follow [SemVer](https://semver.org/) at the
  per-project level. Removing or changing the signature of a public
  symbol → major bump.
- **R-4.3** Use-cases (and external callers) must import from the
  top-level package, not internal modules. Example:
  ```python
  # OK
  from llm_patch import CompilePipeline, MarkdownDataSource
  # NOT OK in a use-case
  from llm_patch.pipelines.compile import CompilePipeline
  ```
- **R-4.4** Legacy import shims (`KnowledgeFusionOrchestrator`,
  `WikiKnowledgeSource`) are kept until at least the next major version
  and removal requires an ADR + deprecation entry in `CHANGELOG.md`.

---

## 5. Testing — TDD Policy

- **R-5.1** New public APIs are **test-first**: write the failing test
  in `tests/unit/` (or `tests/integration/`), then implement.
- **R-5.2** Coverage thresholds (enforced in CI):
  - Engine `core/`: **≥ 85%** branch coverage.
  - Per-project overall: **≥ 75%** branch coverage.
  - New code in a PR: **≥ 80%** for the changed lines.
- **R-5.3** Test layout per project:
  - `tests/unit/` — fast, no I/O, no network. The default suite.
  - `tests/integration/` — marked `@pytest.mark.integration`. Run via
    `pytest -m integration`. May touch the filesystem under `tmp_path`.
- **R-5.4** Tests follow Arrange-Act-Assert and assert one behavior per
  test. Mock at the boundary, not in the middle.
- **R-5.5** A failing baseline is a release blocker. Engine baseline
  for `0.2.0`: `379 passed, 12 skipped` (engine), plus the new
  adapter-market unit tests under `tests/unit/test_adapter_manifest_v2.py`,
  `test_lru_cache.py`, `test_runtime_controller.py`,
  `test_cli_distribute.py`, `test_preflight.py`.

---

## 6. Versioning & Releases

- **R-6.1** Each project version is independent and lives in its own
  `pyproject.toml`.
- **R-6.2** Release tags follow `<project>-vX.Y.Z`. Example:
  `llm-patch-v0.2.0`, `llm-patch-wiki-agent-v0.1.0`.
- **R-6.3** Each project owns a `CHANGELOG.md` in
  [Keep a Changelog](https://keepachangelog.com) format. CI enforces an
  `Unreleased` entry update when files in the project change.

---

## 7. Code Quality Gates (CI-enforced)

| Gate | Tool | Where |
|---|---|---|
| Lint | `ruff check` | pre-commit + CI |
| Format | `ruff format --check` | pre-commit + CI |
| Static types | `mypy --strict` | CI per-project |
| Tests | `pytest -q` | CI per-project |
| Coverage threshold | `pytest --cov` + `coverage report --fail-under` | CI |
| Layering | [tools/check_layering.py](tools/check_layering.py) | pre-commit + CI |
| Changelog | [tools/check_changelog.py](tools/check_changelog.py) | CI on PR |

Legacy engine code is grandfathered against the strictest new ruff rules
via per-file-ignores tied to a tracking ADR
([0002](docs/adr/0002-layered-architecture.md)). New code does not get
the same exemption.

---

## 8. Naming Conventions

- **Packages**: `llm_patch`, `llm_patch_shared`, `llm_patch_<usecase>`.
  Project (distribution) names use hyphens; Python import names use
  underscores.
- **Interfaces (ABCs)**: prefix `I` (e.g., `IDataSource`).
  Implementations: descriptive concrete name (e.g., `MarkdownDataSource`).
- **Configs**: Pydantic `BaseModel` or frozen dataclass; named
  `XxxConfig` or `XxxSpec`.
- **Errors**: derive from `llm_patch_shared.errors.LlmPatchError`.

---

## 9. Adding a New Use-Case Project

1. Run `python tools/scaffold_project.py <name>` to materialize the
   standardized skeleton (`src/`, `tests/{unit,integration}/`, `docs/`,
   `data/`, `artifacts/`, `examples/`, `pyproject.toml`, `AGENTS.md`,
   `README.md`, `CHANGELOG.md`).
2. Add `<name>` as a member of the workspace (already covered by
   `members = ["projects/*"]`).
3. Write its `AGENTS.md` (allowed deps, public API surface, do/don't).
4. Open an ADR if the new use-case adds a new external dependency or
   touches a layering rule.

---

## 10. Glossary

- **Engine**: `projects/llm-patch/` — the generic framework.
- **Use-case**: any project under `projects/` other than the engine and
  shared-utils. Built on top of the engine's public API.
- **Public API**: symbols re-exported from a project's top-level
  `__init__.py`.
- **ADR**: Architecture Decision Record. See
  [docs/adr/0000-template.md](docs/adr/0000-template.md).
- **Layering check**: AST-based scan in
  [tools/check_layering.py](tools/check_layering.py) that fails on
  forbidden cross-project imports.
