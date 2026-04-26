# Roadmap

This document tracks the **versioning promise**, the surface frozen at
v1.0, and the deprecations slated for v2.0. It is the canonical place
to look when deciding whether something is safe to depend on.

## Versioning Promise

`llm-patch` follows [Semantic Versioning](https://semver.org/):

| Component | Stability |
|-----------|-----------|
| Public API symbols re-exported from `llm_patch.__init__` | **Stable** — breaking changes only at major bumps. |
| CLI verbs (`init`, `doctor`, `compile`, `watch`, `chat`, `push`, `pull`, `hub`, `serve`, `version`) | **Stable** — flag removal only at major bumps. Additive flags allowed in minor releases. |
| Adapter manifest **v2** schema | **Stable** — additive evolution only; breaking changes ship a v3 schema and a converter. |
| `IAdapterRegistryClient` protocol | **Stable** — see [REGISTRY_PROTOCOL.md](REGISTRY_PROTOCOL.md). |
| Engine internals (`llm_patch.runtime._*`, `llm_patch.cli._registry`, etc.) | **Unstable** — may change in any release. |
| Advanced subgroups (`adapter`, `model`, `source`, `wiki`) hidden behind `LLM_PATCH_ADVANCED=1` | **Provisional** — may be reorganized in minor releases. |

## v1.0 Frozen Surface

The following are guaranteed not to break in any 1.x release:

- All names listed in [`llm_patch.__all__`](../projects/llm-patch/src/llm_patch/__init__.py).
- The CLI verbs above and their documented flags.
- The adapter manifest v2 JSON schema (see [ADR-0007](adr/0007-adapter-manifest-v2.md)).
- The `IAdapterRegistryClient` protocol.
- The `.llm-patch.toml` config keys: `[project] name`, `[compile] source/output`,
  `[runtime] base_model`, `[registry] plugin`.

## Deprecated in v1.x → Removed in v2.0

| Item | Deprecated in | Removal | Replacement |
|------|---------------|---------|-------------|
| `KnowledgeFusionOrchestrator` (legacy facade) | <= v0.3 | **v2.0.0** | `llm_patch.CompilePipeline` (drop-in for `compile_all`). |
| `LLM_PATCH_REGISTRY` env var (legacy alias) | v1.0.0rc1 | **v2.0.0** | `LLM_PATCH_PLUGIN_REGISTRY` — see [ADR-0008](adr/0008-plugin-discovery.md). |

Each deprecated symbol emits a `DeprecationWarning` at use time. The
warning text names the removal version explicitly so machine-readable
checks (e.g. `pytest -W error::DeprecationWarning`) work in CI.

## Planned Improvements (No Breaking Changes)

- **1.1** — first reference `IAdapterRegistryClient` plugin (community-maintained).
- **1.x** — `llm-patch doctor` profile with optional dependency probes
  (CUDA, MPS, registry connectivity).
- **1.x** — streaming output for `llm-patch chat` and `model generate`.
- **1.x** — additional manifest v2 metadata (license, signed checksums).

## Out of Scope for v1

See [LIMITATIONS.md](LIMITATIONS.md).

## Process

- Architectural decisions go through [ADRs](adr/README.md).
- All API additions ship with tests in
  [tests/unit/test_public_api.py](../projects/llm-patch/tests/unit/test_public_api.py)
  or an equivalent.
- Removal of deprecated items requires:
  1. Listed in this document at deprecation time.
  2. `DeprecationWarning` emitted for at least one minor version.
  3. CHANGELOG entry under the major-bump section.
