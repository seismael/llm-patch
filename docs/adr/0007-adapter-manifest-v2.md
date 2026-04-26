# 0007 — Adapter Identity & Manifest v2

- **Status**: Accepted
- **Date**: 2026-04-26
- **Deciders**: llm-patch maintainers
- **Tags**: architecture | public-api | data-model | distribution

## Context

`AdapterManifest` (v1) carried only the minimum fields needed for local
compile/attach: `adapter_id`, `rank`, `target_modules`, `storage_uri`,
`created_at`. The Adapter Market use case (see
[ADR-0006](0006-distributed-adapter-registry.md)) requires a manifest
that doubles as a **universal package descriptor** so that adapters can
be searched, resolved, fetched, and verified across hubs.

We need the v2 fields without breaking any v1 consumer (engine
internals, the `wiki-agent` use-case, and existing on-disk artifacts).

## Decision

Extend `AdapterManifest` with **optional** fields and introduce a
companion value object `AdapterRef` for hub-style URIs. All additions
are backwards-compatible: v1 manifests continue to load and round-trip.

### Fields added to `AdapterManifest`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `manifest_version` | `int` | `1` | Bumped to `2` by writers that emit v2 fields. |
| `namespace` | `str \| None` | `None` | `owner/name` slug (lower-kebab; validated). |
| `version` | `str \| None` | `None` | SemVer (`1.2.0` or `v1.2.0`, optional `-pre`). |
| `checksum_sha256` | `str \| None` | `None` | 64-char lowercase hex of the `.safetensors` payload. |
| `base_model_compatibility` | `list[str]` | `[]` | Compatible base-model ids (e.g. `google/gemma-2-2b-it`). |
| `tags` | `list[str]` | `[]` | Free-form discovery tags. |
| `description` | `str \| None` | `None` | Human-readable summary. |

Pydantic validators enforce SemVer, hex-64 checksums, and the
`owner/name` slug shape. Unknown extra fields remain rejected (strict
mode), so future v3 extensions will go through this same ADR process.

### URI grammar — `hub://`

```
hub-uri    := "hub://" namespace "/" name [":" version]
namespace  := <lower-kebab token>
name       := <lower-kebab token>
version    := SemVer | "latest"
```

Examples:

- `hub://acme/react-19:1.2.0`
- `hub://acme/react-19:v1.2.0`
- `hub://acme/react-19` (resolves to `:latest`)

`AdapterRef.parse(uri)` produces a frozen value object with
`namespace`, `name`, `version`, `to_uri()`, and a filesystem-safe
`adapter_id` property (`"acme__react-19__v1.2.0"`).

`hf://owner/repo` and `s3://bucket/key` are recognized at the CLI layer
and dispatched to whatever `IAdapterRegistryClient` is registered for
that scheme. The engine itself does **not** parse those — they are
client-defined per ADR-0006.

## Consequences

### Positive

- Manifest v2 is the same Pydantic class with optional fields → no
  call-site change for v1 callers.
- `AdapterRef` is a small, frozen, immutable value object; safe to
  share across threads and to use as dict keys.
- The checksum field enables anti-tampering at pull time
  (`ChecksumMismatchError`) without forcing a full signing system now.

### Negative / Trade-offs

- Two manifests in the same repo can have different field sets. UIs
  that expect uniform fields must guard with `if manifest.version`.
- Strict-mode Pydantic + extra fields means every legitimate v3
  extension must come through an ADR. Acceptable for stability.

### Neutral

- Engine version → `0.2.0` (additive minor, per
  [0003](0003-public-api-policy.md)).
- New public symbols re-exported from `llm_patch.__init__`:
  `AdapterRef`, three ABCs, `LRUAdapterCache`, `PeftRuntimeController`.

## Alternatives Considered

### Alternative A — A separate `AdapterPackage` model

Keep v1 `AdapterManifest` frozen and introduce `AdapterPackage` for
distribution. Rejected: doubles the type surface, forces every
adapter-aware function to accept either type, and bloats serialization
glue.

### Alternative B — Versioned manifests as separate classes

`AdapterManifestV1`, `AdapterManifestV2`, with a discriminator. Rejected:
violates OCP for trivially additive fields and pollutes the public
surface. Pydantic optional fields with a `manifest_version` integer
satisfy the same goal cheaper.

### Alternative C — Bake checksum into a sidecar file

Store checksum/signature in `manifest.sig` next to the safetensors.
Rejected for v2: adds a new artifact and a second loader. Carrying it
inside the manifest keeps the unit of distribution single-file from
the consumer's perspective. Signing (vs. plain checksum) is deferred.

## References

- [0003](0003-public-api-policy.md), [0006](0006-distributed-adapter-registry.md)
- [SPEC.md §3, §4](../../SPEC.md)
- [docs/REGISTRY_PROTOCOL.md](../REGISTRY_PROTOCOL.md)
- [`projects/llm-patch/src/llm_patch/core/models.py`](../../projects/llm-patch/src/llm_patch/core/models.py)
