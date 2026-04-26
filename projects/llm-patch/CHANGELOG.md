# Changelog — llm-patch

All notable changes to this package are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.0.0rc1] — 2026-04-26

First production-readiness release candidate. Public API and CLI verbs
are now frozen for the 1.x series — see
[docs/ROADMAP.md](../../docs/ROADMAP.md).

### Added
- **Project config**: `.llm-patch.toml` is now consumed end-to-end. New `llm_patch.core.project_config.ProjectConfig` walks upward from the working directory, parses `[project]` / `[compile]` / `[runtime]` / `[registry]` sections via stdlib `tomllib`, and is honored by `llm-patch compile`, `watch`, and `chat` (explicit flags > config > defaults).
- **Manifest v2 round-trip integration test** (`tests/integration/test_manifest_v2_roundtrip.py`) — exercises the full `IAdapterRegistryClient` contract through an in-memory fake: `publish` → `search` → `resolve` → `pull` with checksum verification.
- **Documentation**: [docs/ROADMAP.md](../../docs/ROADMAP.md) (versioning promise, frozen surface, deprecation timeline) and [docs/LIMITATIONS.md](../../docs/LIMITATIONS.md) (non-goals and known limits).

### Changed
- **Registry env var canonicalized**: `LLM_PATCH_PLUGIN_REGISTRY` is now the canonical name, aligned with the `LLM_PATCH_PLUGIN_<KIND>` family from [ADR-0008](../../docs/adr/0008-plugin-discovery.md). The legacy `LLM_PATCH_REGISTRY` name remains accepted as a deprecated alias and emits a `DeprecationWarning`; removal scheduled for **v2.0.0**. ADR-0006 amended accordingly.
- **Lazy public API**: `llm_patch.__init__` now resolves heavy attributes via :pep:`562` `__getattr__`. Importing `llm_patch.cli` no longer pulls in `torch`, `peft`, or `transformers`. The Phase-2 `xfail` for `--help` cold-start is removed; the test now passes unconditionally.
- **Deprecation message tightened**: `KnowledgeFusionOrchestrator` now names **v2.0.0** as its removal version and links to `docs/ROADMAP.md`.
- Engine version `0.3.0` → `1.0.0rc1`.

### Tests
- Baseline now **464 passed, 12 skipped, 0 xfailed** (was `442 passed, 12 skipped, 1 xfailed`). New: 12 unit tests for `ProjectConfig`, 5 integration tests for `init`+`compile`+config round-trip, 4 integration tests for manifest v2 round-trip; 1 prior `xfail` (lazy `--help`) is now a real passing assertion.

## [0.3.0] — 2026-XX-XX

### Added
- **CLI**: new primary verbs `init`, `doctor`, `version`. Composite `CommandRegistry` hides advanced groups (`adapter`, `model`, `source`, `wiki`) in `--help` unless `LLM_PATCH_ADVANCED=1`. Root group gains global `--quiet`, `--json`, `--no-color`.
- **Plugin discovery** — `llm_patch.core.plugins` exposes `PluginLoader`, `PluginKind`, `PluginSpec`. Env-var (`LLM_PATCH_PLUGIN_<KIND>`) + entry-point (`llm_patch.plugins`) channels; env-var wins. Documented in [docs/EXTENDING.md](../../docs/EXTENDING.md) and [ADR-0008](../../docs/adr/0008-plugin-discovery.md).

### Changed
- Public API re-exports `PluginLoader`, `PluginKind`, `PluginSpec`.
- Engine version `0.2.0` → `0.3.0`.

### Tests
- Baseline now `439 passed, 12 skipped, 1 xfailed` (was `415 passed, 12 skipped`). The xfail tracks a deliberate follow-up: trim torch/peft/transformers imports out of the `--help` cold path.

## [0.2.0] — 2026-04-26

### Added
- **Adapter Market scaffolding** — manifest v2 (`namespace`, `version`, `checksum_sha256`, `base_model_compatibility`, `tags`, `description`), `AdapterRef` value object, and the `hub://namespace/name:version` URI grammar ([ADR-0007](../../docs/adr/0007-adapter-manifest-v2.md)).
- **Three new engine ABCs**: `IAdapterRegistryClient`, `IAdapterCache`, `IRuntimeAdapterController` ([ADR-0006](../../docs/adr/0006-distributed-adapter-registry.md)).
- **Reference impls**: `LRUAdapterCache` (stdlib `OrderedDict` + `RLock`) and `PeftRuntimeController` (RLock-serialized hot-swap on `ModelHandle`).
- **CLI**: top-level verbs `compile`, `watch`, `chat`, `push`, `pull` and the `hub` group (`search`, `info`). `--json` / `--quiet` global flags. Heavy deps lazy-imported so `--help` is fast.
- **Server**: `POST /adapters/attach`, `POST /adapters/detach`, `GET /adapters/active`, `GET /cache/stats` serialized via a single `asyncio.Lock`.
- **MCP**: `search_knowledge_hub`, `pull_hub_adapter`, `load_hub_adapter`, `unload_hub_adapter`, `list_active_adapters`. Tools fail loudly with `RegistryUnavailableError` until `configure_hub(...)` is called.
- **Runtime preflight**: `runtime/preflight.py` with `PreflightReport.probe()` for VRAM/CUDA discovery; `torch` is imported lazily.
- **Public API**: `AdapterRef`, `IAdapterRegistryClient`, `IAdapterCache`, `IRuntimeAdapterController`, `LRUAdapterCache`, `PeftRuntimeController` re-exported from `llm_patch.__init__`.
- **Tests**: 36 new unit tests across manifest v2, LRU cache, runtime controller, CLI distribute, and preflight.

### Changed
- `PeftAgentRuntime` accepts an optional `controller: IRuntimeAdapterController`; when set, generation reads through `controller.handle` so hot-swaps are visible immediately.
- Shared error hierarchy gained `AdapterNotFoundError`, `ChecksumMismatchError`, `IncompatibleBaseModelError`, `RegistryUnavailableError`, `CapacityExceededError`.

### Notes
- No concrete `IAdapterRegistryClient` ships with the engine. Operators wire one via `LLM_PATCH_PLUGIN_REGISTRY="module:factory"` (canonical, since 1.0.0rc1; `LLM_PATCH_REGISTRY` remains accepted as a deprecated alias) (CLI/MCP) or `configure_hub(...)` (server). See [docs/AGENTIC_AI_INTEGRATION.md](../../docs/AGENTIC_AI_INTEGRATION.md) and [docs/REGISTRY_PROTOCOL.md](../../docs/REGISTRY_PROTOCOL.md).

## [0.1.0] — 2026-04-21

### Added
- Workspace-packaged engine project extracted into `projects/llm-patch/` while preserving the `llm_patch` import path.
- Generic Ingest → Compile → Attach → Use engine with sources, generators, storage, runtime, pipelines, CLI, server, MCP, and wiki modules.
