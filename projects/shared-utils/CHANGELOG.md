# Changelog — llm-patch-shared

All notable changes to this package are documented in this file. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `DependencyError` and `ResourceNotFoundError` to extend the shared monorepo error hierarchy.

### Changed
- Top-level package exports now expose the shared configuration, integration, dependency, and resource error types.

## [0.1.0] — 2026-04-21

### Added
- Initial scaffold with `LlmPatchError` hierarchy and version export.
