# 0006 — Distributed Adapter Registry & Runtime Boundaries

- **Status**: Accepted
- **Date**: 2026-04-26
- **Deciders**: llm-patch maintainers
- **Tags**: architecture | layering | dependency | distribution | runtime

> **Amendment (2026-04-26, v1.0.0rc1)**: The configuration env var was
> renamed from `LLM_PATCH_REGISTRY` to `LLM_PATCH_PLUGIN_REGISTRY` to
> align with the `LLM_PATCH_PLUGIN_<KIND>` family introduced by
> [ADR-0008](0008-plugin-discovery.md). The legacy name remains accepted
> with a `DeprecationWarning`; removal is scheduled for **v2.0.0** (see
> [docs/ROADMAP.md](../ROADMAP.md)).

## Context

The "Distributed Knowledge Registry & Agentic Runtime" use case (a.k.a.
**Adapter Market**, see [docs/AGENTIC_AI_INTEGRATION.md](../AGENTIC_AI_INTEGRATION.md))
positions `llm-patch` as the substrate for treating LoRA adapters as
versioned, immutable artifacts that agents can fetch and hot-swap on
demand — analogous to Docker images or NPM packages.

That use case implies four cross-cutting capabilities:

1. **Remote distribution** — pull/push adapters from public hubs or
   private enterprise stores (HTTP, S3, HF Hub, …).
2. **Caching** — bound the on-disk and in-memory footprint when many
   adapters are in play.
3. **Hot-swap inference** — attach/detach adapters on a live model
   handle, safely under concurrency.
4. **Agentic discovery** — let agents query the hub through MCP and
   trigger their own knowledge upgrades.

The status quo only covers local storage (`LocalSafetensorsRepository`)
and one-shot attach (`PeftAdapterLoader`). Shipping concrete HTTP/S3/HF
clients inside the engine would (a) bloat dependencies, (b) lock
deployments to one transport, and (c) tangle the engine's stable layer
with experimental network code.

## Decision

The engine ships **interfaces only** for the distribution layer. It
does **not** ship concrete network/storage clients in this version.

Concretely, three new ABCs land in
[`core/interfaces.py`](../../projects/llm-patch/src/llm_patch/core/interfaces.py):

- **`IAdapterRegistryClient`** — `search`, `resolve`, `pull`, `push`.
  The contract for any remote registry (HTTP hub, HF Hub, S3 bucket,
  custom). Implementations live outside the engine, are wired in by
  use-cases, and reach the engine through the Strategy / Repository
  patterns.
- **`IAdapterCache`** — bounded, in-memory cache of
  `AdapterManifest`s. Reference impl `LRUAdapterCache` (stdlib-only).
- **`IRuntimeAdapterController`** — `attach`, `detach`, `active`. The
  contract for serialized hot-swap on a live handle. Reference impl
  `PeftRuntimeController` wraps an existing `IAdapterLoader` +
  `ModelHandle`.

CLI (`push`/`pull`/`hub`), MCP hub tools, and server hot-swap endpoints
are wired to these ABCs and **fail loudly with `RegistryUnavailableError`**
when no concrete client is configured. The integration mechanism is
constructor injection plus an opt-in environment variable
(`LLM_PATCH_PLUGIN_REGISTRY="module:factory"`) so that operators choose their
transport without engine code changes.

This ADR explicitly defers:

- `HTTPRegistryRepository`, `S3EnterpriseRepository`, `HFHubRegistryClient`
  concrete impls.
- LoRAX / batched multi-adapter inference. Concurrency uses a single
  `asyncio.Lock` (server) and `threading.RLock` (controller) until a
  dedicated ADR justifies LoRAX.
- Live VRAM measurement. A static estimator stub is acceptable.
- Registry-side server code. This repo defines the **protocol**
  ([docs/REGISTRY_PROTOCOL.md](../REGISTRY_PROTOCOL.md)), not the
  server.

## Consequences

### Positive

- Engine remains transport-agnostic (`Strategy` + `Repository`).
  Anyone can plug in HTTP, S3, HF, or a local-only test double.
- Public API stays additive (only new symbols). No breaking change.
- Dependency direction (R-1.1) preserved: engine ABCs → optional
  third-party clients in use-cases; never the reverse.
- Shipped reference impls (`LRUAdapterCache`, `PeftRuntimeController`)
  give immediate value without imposing transport choices.
- Concurrency contract is documented and enforced by reference impls,
  not left to each consumer to rediscover.

### Negative / Trade-offs

- Out-of-the-box adoption requires the user to register a concrete
  registry client (or set `LLM_PATCH_PLUGIN_REGISTRY`). The CLI is unusable
  for `push`/`pull` until that is done — by design.
- The hot-swap path is locked to a global lock for now; it does not
  scale to LoRAX-grade batched throughput. Documented in
  [docs/SERVER_ARCHITECTURE.md](../SERVER_ARCHITECTURE.md).

### Neutral

- Existing local pipelines (`CompilePipeline`, `UsePipeline`,
  `LocalSafetensorsRepository`) are untouched.
- Engine bumps to `0.2.0` (additive minor) per
  [0003](0003-public-api-policy.md).

## Alternatives Considered

### Alternative A — Ship a built-in HTTP client

Bundle an `HTTPRegistryClient` using `httpx` as a default extra. Pro:
zero-config UX. Con: ties the engine to a specific protocol version,
forces dep churn into core, conflicts with users who already operate
internal hubs with custom auth.

### Alternative B — Single `IAdapterRepository` extension

Add network methods directly to the existing `IAdapterRepository` ABC.
Rejected: violates ISP (one interface for both local persistence and
remote distribution), and would be a breaking change for every
existing implementer.

### Alternative C — LoRAX integration in v1

Replace the lock-based controller with LoRAX. Rejected: pulls in a
heavy GPU-bound dependency before the surface is shaken out. The lock
model is a portable stepping stone and was sized for ≤ tens of
adapters per node.

## References

- [0002](0002-layered-architecture.md), [0003](0003-public-api-policy.md), [0005](0005-engine-shared-errors.md)
- [SPEC.md §1, §3, §4](../../SPEC.md)
- [docs/AGENTIC_AI_INTEGRATION.md](../AGENTIC_AI_INTEGRATION.md)
- [docs/REGISTRY_PROTOCOL.md](../REGISTRY_PROTOCOL.md)
- [docs/SERVER_ARCHITECTURE.md](../SERVER_ARCHITECTURE.md)
- [0007](0007-adapter-manifest-v2.md)
