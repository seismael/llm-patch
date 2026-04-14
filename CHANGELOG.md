# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/seismael/llm-patch/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/seismael/llm-patch/releases/tag/v0.1.0
