# ADR-0008: Plugin discovery via env vars + entry points

* **Status:** Accepted
* **Date:** 2025-01-XX
* **Deciders:** llm-patch maintainers
* **Supersedes / Superseded by:** —

## Context

`llm-patch` 0.2.0 shipped six pluggable ABCs (`IDataSource`,
`IWeightGenerator`, `IAdapterLoader`, `IAdapterRegistryClient`,
`IAdapterCache`, `IRuntimeAdapterController`). Until 0.3.0, the only way
to wire a custom implementation into the CLI was to construct it in
Python and call the public API directly. For the **Adapter Market**
use-case (hot-swappable registry clients via `llm-patch push` /
`pull`), and for community-contributed sources/generators, we need a
discovery mechanism that:

1. Works without modifying engine source.
2. Stays **lazy** — the `--help` fast path must not import plugin code.
3. Supports both ad-hoc setups (single user, one env var) and
   distributable plugins (a `pip install`-able package).
4. Has predictable precedence and clear error reporting.

## Decision

Add `llm_patch.core.plugins` with three types and a single loader:

* `PluginKind` (enum: SOURCE / GENERATOR / LOADER / RUNTIME / REGISTRY /
  CACHE / CONTROLLER)
* `PluginSpec` (frozen dataclass: `module:attribute` + origin tag)
* `PluginLoader` (env-var + entry-point discovery)

Two discovery channels:

1. **Environment variables** — `LLM_PATCH_PLUGIN_<KIND>=module:factory`.
   One spec per kind, lookup is O(1), no import side-effects.
2. **Entry points** — `[project.entry-points."llm_patch.plugins"]` in a
   plugin package's `pyproject.toml`. Discovered via
   `importlib.metadata.entry_points`.

**Precedence:** env-var beats entry-point. Rationale: env-var is the
explicit, per-process override; entry-points are the implicit default.

**Resolution returns the factory callable**, not an instance. The
caller decides when to invoke it (so plugin construction failures
surface in user code with full traceback, not deep inside the engine).

The new symbols are re-exported from the top-level `llm_patch` package
(`PluginLoader`, `PluginKind`, `PluginSpec`).

## Consequences

### Positive

* Plugin authors can ship a `pip install`-able package and have it
  picked up automatically — no engine PR required.
* The CLI's `doctor` command can report which plugins are configured.
* Entry-point lookup is lazy; `--help` stays fast.
* Backward compatible: callers that build instances by hand continue to
  work unchanged.

### Negative

* Two channels (env vs entry-points) means two places to look when
  debugging. Mitigated by `llm-patch doctor` showing the resolved spec.
* Entry points are not yet keyed by `PluginKind`. Future work may
  introduce a structured naming convention (e.g. group name suffix).

### Neutral

* The loader is intentionally minimal — caching and validation are
  deferred until a concrete need surfaces.

## Alternatives Considered

* **Config-file only** (`.llm-patch.toml [plugins]`): rejected because
  it requires the engine to read TOML during the CLI fast path.
* **CLI flag for plugin module** (`--registry-plugin module:factory`):
  rejected because it pushes the choice into every command instead of
  the environment.
* **Auto-discovery by import path scan**: rejected — too magical,
  unreliable across virtualenvs.
