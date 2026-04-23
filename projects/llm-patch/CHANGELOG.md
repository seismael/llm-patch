# Changelog — llm-patch

All notable changes to this package are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Public top-level exports for the engine runtime and attach-layer concretes: `ChatSession`, `PeftAgentRuntime`, `HFModelProvider`, and `PeftAdapterLoader`.
- Public API tests covering the new top-level engine exports.

### Changed
- Legacy orchestrator shims now emit runtime deprecation warnings to guide callers toward `CompilePipeline` and `WikiPipeline`.
- Engine runtime integration boundaries now use the shared `llm_patch_shared` error hierarchy for dependency and model-attachment failures.

## [0.1.0] — 2026-04-21

### Added
- Workspace-packaged engine project extracted into `projects/llm-patch/` while preserving the `llm_patch` import path.
- Generic Ingest → Compile → Attach → Use engine with sources, generators, storage, runtime, pipelines, CLI, server, MCP, and wiki modules.
