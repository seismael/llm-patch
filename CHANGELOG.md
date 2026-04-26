# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0rc1] — 2026-04-26

First production-readiness release candidate. Public API surface and
CLI verbs are now frozen for 1.x. See
[docs/ROADMAP.md](docs/ROADMAP.md) for the versioning promise and
[docs/LIMITATIONS.md](docs/LIMITATIONS.md) for non-goals.

### Added

- **Project-config end-to-end**: `llm-patch compile`, `watch`, and `chat` now consume `.llm-patch.toml` (`[compile] source/output`, `[runtime] base_model`, `[registry] plugin`). Explicit CLI flags > project config > defaults; env vars always win over the project's `[registry] plugin`.
- **Manifest v2 round-trip integration coverage**: `tests/integration/test_manifest_v2_roundtrip.py` runs an in-memory `IAdapterRegistryClient` fake through the full publish → search → resolve → pull cycle (with checksum verification) so plugin authors have a concrete reference.
- **Documentation**: new [docs/ROADMAP.md](docs/ROADMAP.md) (semver promise, frozen surface, v2.0 removals) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md) (non-goals, platform support, security caveats).

### Changed

- **Canonical registry env var**: `LLM_PATCH_PLUGIN_REGISTRY` is now canonical (aligned with the `LLM_PATCH_PLUGIN_<KIND>` family from [ADR-0008](docs/adr/0008-plugin-discovery.md)). The legacy `LLM_PATCH_REGISTRY` name remains accepted as a deprecated alias and emits `DeprecationWarning`; **removal in v2.0.0**. README, ARCHITECTURE, USAGE, AGENTIC_AI_INTEGRATION, ADR-0006, and the publish-workflow template were swept accordingly.
- **Lazy public-API loader**: `llm_patch.__init__` switched to :pep:`562` `__getattr__`. Importing `llm_patch.cli` no longer pulls in `torch`, `peft`, or `transformers`; `llm-patch --help` is now genuinely cold-start friendly. The Phase-2 `xfail` is retired.
- **Deprecation messages**: `KnowledgeFusionOrchestrator` now names **v2.0.0** as its removal version and points at `docs/ROADMAP.md`.
- Engine version `0.3.0` → `1.0.0rc1`.

### Tests

- Baseline **464 passed, 12 skipped, 0 xfailed** (was `442 / 12 / 1`). +21 new tests across `ProjectConfig`, init→compile round-trip, and manifest v2 round-trip; the lazy `--help` xfail is now a hard pass.

### Added — 0.3.0 (Community / CLI-first pivot)

- **CLI cleanup** — new primary verbs:
  - `llm-patch init` — interactive scaffold writing `.llm-patch.toml` (stdlib-only).
  - `llm-patch doctor` — Python / extras / torch / CUDA / registry-plugin probe (text + `--json` + `--quiet`).
  - `llm-patch version` — engine version (text + `--json`).
  - New `CommandRegistry` (Composite) hides advanced groups (`adapter`, `model`, `source`, `wiki`) in `--help` unless `LLM_PATCH_ADVANCED=1`.
  - Root group now accepts global `--quiet`, `--json`, `--no-color` flags.
- **Plugin discovery** — `llm_patch.core.plugins` (`PluginLoader`, `PluginKind`, `PluginSpec`):
  - Env-var channel: `LLM_PATCH_PLUGIN_<KIND>=module:factory`.
  - Entry-point channel: `[project.entry-points."llm_patch.plugins"]`.
  - Documented in [docs/EXTENDING.md](docs/EXTENDING.md) and [ADR-0008](docs/adr/0008-plugin-discovery.md).
- **Documentation pivot** — [docs/QUICKSTART.md](docs/QUICKSTART.md) (5-minute CLI walkthrough), [docs/COMMUNITY.md](docs/COMMUNITY.md) (channels + plugin gallery + roadmap labels), [examples/quickstart/](examples/quickstart/) (demo notes + run scripts powering the README cast).
- **Community infrastructure** — five GitHub issue templates (`bug_report`, `feature_request`, `new_source_plugin`, `new_registry_client`, `documentation`) plus issue-template `config.yml`.

### Changed — 0.3.0

- Engine version bumped `0.2.0` → `0.3.0`.
- Public API of `llm_patch` now re-exports `PluginLoader`, `PluginKind`, `PluginSpec`.
- README repositioned around the CLI (60-second quickstart at the top, framework story moved to ARCHITECTURE).
- `docs/USAGE.md` table-of-contents reordered to put `CLI Reference` and `Publishing & consuming adapters` above the Python API.
- PR template baseline note no longer hardcodes a specific test count.
- CONTRIBUTING.md gains a "Plugins (`pip install`-able add-ons)" section pointing to `docs/EXTENDING.md`.

### Notes

- Engine baseline: `439 passed, 12 skipped, 1 xfailed` (was `415 passed, 12 skipped` at 0.2.0). The new xfail tracks a deliberate Phase-2 follow-up: trim heavy imports out of the `--help` cold path.
- All earlier 0.2.0 entries below remain unchanged.

### Added — 0.2.0 (already shipped)

- **Adapter Market scaffolding** — distributed-knowledge-registry use case wired end-to-end through the engine's public API. Includes:
  - Manifest v2 (`namespace`, `version`, `checksum_sha256`, `base_model_compatibility`, `tags`, `description`) with backward-compatible v1 round-trip ([ADR-0007](docs/adr/0007-adapter-manifest-v2.md)).
  - `AdapterRef` value object with `hub://namespace/name:version` URI grammar.
  - Three new ABCs in `core/interfaces.py`: `IAdapterRegistryClient`, `IAdapterCache`, `IRuntimeAdapterController` ([ADR-0006](docs/adr/0006-distributed-adapter-registry.md)).
  - Reference impls: `LRUAdapterCache` (stdlib LRU) and `PeftRuntimeController` (RLock-serialized hot-swap).
  - Top-level CLI verbs `compile`, `watch`, `chat`, `push`, `pull`, plus the `hub` group (`search`, `info`). All distribute commands honor `--json` / `--quiet`.
  - FastAPI hot-swap endpoints `POST /adapters/attach`, `POST /adapters/detach`, `GET /adapters/active`, `GET /cache/stats` (serialized via a single `asyncio.Lock`).
  - Five MCP hub tools: `search_knowledge_hub`, `pull_hub_adapter`, `load_hub_adapter`, `unload_hub_adapter`, `list_active_adapters`.
  - `runtime/preflight.py` with `PreflightReport.probe()` for VRAM/CUDA discovery (lazy `torch` import).
  - New shared error types: `AdapterNotFoundError`, `ChecksumMismatchError`, `IncompatibleBaseModelError`, `RegistryUnavailableError`, `CapacityExceededError`.
- **Documentation** — [docs/AGENTIC_AI_INTEGRATION.md](docs/AGENTIC_AI_INTEGRATION.md) rewritten as the canonical Adapter Market use case with a Mermaid sequence and a per-requirement Status Matrix; new [docs/REGISTRY_PROTOCOL.md](docs/REGISTRY_PROTOCOL.md) and [docs/SERVER_ARCHITECTURE.md](docs/SERVER_ARCHITECTURE.md).

### Changed

- Engine version bumped `0.1.0` → `0.2.0` (additive minor; no breaking changes).
- Public API of `llm_patch` now re-exports `AdapterRef`, `IAdapterRegistryClient`, `IAdapterCache`, `IRuntimeAdapterController`, `LRUAdapterCache`, `PeftRuntimeController`.
- SPEC §3 (Patterns) extended with Repository-over-network, Cache/Decorator, and Strategy-on-runtime entries; SPEC §5 baseline updated to `379 passed, 12 skipped` plus new adapter-market unit tests.

## [0.2.0] - 2026-04-18

### Added

- **LiteLLM wiki agent** — `LiteLLMWikiAgent` supports any model via LiteLLM (Gemini, OpenAI, Anthropic, Ollama, etc.) with automatic API key resolution from environment variables.
- **Retry with exponential backoff** — `LiteLLMWikiAgent._call()` retries up to 5 times on rate limit errors (15s → 30s → 60s → 120s backoff), catching `RateLimitError`, `AuthenticationError`, and `BadRequestError` distinctly.
- **Pipeline composition layer** — `CompilePipeline`, `UsePipeline`, and `WikiPipeline` in `llm_patch.pipelines` replace legacy orchestrators.
- **Attach layer** — `HFModelProvider` (model loading), `PeftAdapterLoader` (LoRA attachment), `merge_into_base()` and `weighted_blend()` (adapter merging).
- **Agent runtime** — `PeftAgentRuntime` with generate/chat/stream and `ChatSession` for stateful multi-turn conversations.
- **HTTP API server** — FastAPI-based server (`llm-patch serve`) with adapter management and inference endpoints.
- **MCP server** — Model Context Protocol server exposing wiki tools for Claude Desktop, Cursor, and other MCP clients.
- **Additional data sources** — `PdfDataSource`, `HttpApiDataSource`, `JsonlDataSource`, `CompositeDataSource` with namespace support.
- **Obsidian vault integration** — `ObsidianVault` with graph export, `.obsidian/` config, CLI commands (`obsidian init | graph | status`).
- **CLI expansion** — `source`, `model`, `adapter`, `wiki`, `serve` command groups with `--agent` selection (mock, litellm, anthropic).
- **Gemini E2E comparison script** — `scripts/run_gemini_comparison.py` demonstrating before/after knowledge improvement with real Gemini API.
- **E2E walkthrough documentation** — `docs/E2E_WALKTHROUGH.md` with 6-step guide and Phase 10 before/after comparison.

### Changed

- **Test suite expanded** — 274 unit tests + integration tests (up from 69 in v0.1.0).
- **Legacy shims** — `KnowledgeFusionOrchestrator` and `WikiPipelineOrchestrator` preserved as backward-compat wrappers around new pipelines.
- **Package exports** — `__init__.py` now exports 69 symbols covering core interfaces, models, config, pipelines, and sources.

### Fixed

- **JSON extraction regex** — Changed non-greedy `.*?` to greedy `.*` inside code fence patterns in `BaseWikiAgent._extract_json_array()` and `_extract_json_object()`, fixing truncated JSON when content contains `]` or `}` characters.

### Validated — Gemini Before/After Comparison

Live comparison using `gemini/gemini-2.0-flash` on 2 research papers (Attention Is All You Need, LoRA):

| Metric | Before (raw LLM) | After (wiki-enhanced) |
|---|---|---|
| Answer length | 642 chars | 1,848 chars (+188%) |
| Domain terms | 3/5 | 5/5 (+67%) |
| Citations | 0 | 6 wiki pages |
| Math formulas | No | Yes (attention equation, LoRA decomposition) |
| Wiki links | No | Yes ([[Transformer]], [[Self-Attention]], etc.) |

## [0.1.0] - 2026-04-14

### Added

- **Core architecture** — SOLID-based pluggable pipeline with `IKnowledgeSource`, `IWeightGenerator`, and `IAdapterRepository` interfaces.
- **KnowledgeFusionOrchestrator** — Central facade coordinating the document-to-LoRA pipeline with context manager and batch/watch modes.
- **MarkdownDirectoryWatcher** — `IKnowledgeSource` implementation that watches directories for Markdown file changes using `watchdog`.
- **WikiKnowledgeSource** — Wiki-aware source with YAML frontmatter parsing, `[[wikilink]]` extraction, and cross-page content aggregation via `WikiDocumentAggregator`.
- **SakanaT2LGenerator** — `IWeightGenerator` implementation wrapping Sakana AI's Text-to-LoRA hypernetwork for single-pass weight generation.
- **LocalSafetensorsRepository** — `IAdapterRepository` implementation storing adapters as `safetensors` files with PEFT-compatible `adapter_config.json`.
- **Pydantic configuration models** — `GeneratorConfig`, `WatcherConfig`, `StorageConfig` with validation and defaults.
- **Domain models** — Frozen `DocumentContext` and `AdapterManifest` with automatic UTC timestamps.
- **End-to-end demo** — 5-step scripted scenario (`demo_e2e_scenario.py`) proving wiki changes progressively improve model answers.
- **Example pipeline** — `research_pipeline.py` with batch and watch modes, `run_e2e.py` full runner, `validate_adapter.py` GPU inference comparison.
- **Sample data** — 3 ML research paper summaries (Attention Is All You Need, LoRA, GPT-3).
- **Test suite** — 69 tests (unit + integration + E2E), all passing.
- **Type safety** — Full `mypy --strict` compliance with `py.typed` marker.
- **Code quality** — Ruff lint + format, pre-commit hooks.
- **Documentation** — README with use cases, architecture guide, examples tutorial, API reference in docstrings.

[Unreleased]: https://github.com/seismael/llm-patch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/seismael/llm-patch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/seismael/llm-patch/releases/tag/v0.1.0
