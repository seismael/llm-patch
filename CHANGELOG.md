# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
