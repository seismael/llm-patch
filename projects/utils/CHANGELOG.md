# Changelog — llm-patch-utils

All notable changes to this package are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-04-22

### Changed
- **BREAKING**: Renamed distribution `llm-patch-shared` → `llm-patch-utils` and
  import package `llm_patch_shared` → `llm_patch_utils`. See
  [ADR-0009](../../docs/adr/0009-monorepo-structural-unification.md).
- Folder moved from `projects/shared-utils/` to `projects/utils/`.

## [0.1.0] — 2026-04-21

### Added
- `DependencyError` and `ResourceNotFoundError` to extend the shared monorepo error hierarchy.
- Top-level package exports for the shared configuration, integration, dependency, and resource error types.
- Initial scaffold with `LlmPatchError` hierarchy and version export.
