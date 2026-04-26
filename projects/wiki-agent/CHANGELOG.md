# Changelog — llm-patch-wiki-agent

All notable changes to this package are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Living Wiki Copilot** end-to-end use case (per `NOTES.md`):
  - `AdapterMetadata` + `SidecarMetadataRegistry` — JSON-sidecar registry
    enriching engine `AdapterManifest` with `context_id`, `tags`, and
    `summary` (engine model untouched).
  - `IAdapterRouter` Strategy ABC + `MetadataExactMatchRouter` for
    deterministic context-driven adapter selection.
  - `WikiCompileDaemon` (Phase 1) — wraps `engine.CompilePipeline` with
    sidecar persistence; supports batch (`run_once`) and live modes.
  - FastAPI inference gateway (Phase 2) — `create_app(GatewayContext)`
    exposing `GET /health`, `GET /v1/adapters`, `POST /v1/route`,
    `POST /v1/chat`, with a thread-safe per-adapter runtime cache.
  - MCP server (Phase 3) — `build_server(GatewayContext)` with tools
    `internalize_knowledge`, `list_adapters`, `chat_with_adapter`.
  - `llm-patch-wiki-agent daemon`, `serve`, and `mcp` CLI subcommands.
  - Optional extras `[server]` (fastapi+uvicorn) and `[mcp]` (mcp).
- Real `WikiAgent` compile and one-shot chat workflows built on the public `llm_patch` API only.
- `llm-patch-wiki-agent compile`, `chat`, and enriched `info` CLI commands.
- Public `WikiAgentInfo` metadata type and unit tests covering compile/chat/info behavior.

## [0.1.0] — 2026-04-21

### Added
- Initial scaffold: `WikiAgent` + `WikiAgentConfig` placeholder, `llm-patch-wiki-agent` CLI stub.
