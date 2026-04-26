# Adapter Registry Protocol

> **Status**: v1 — protocol contract for the **Adapter Market** use case.
> Authoritative for any registry that wants to be reachable through
> `IAdapterRegistryClient`. The engine itself does **not** ship a
> concrete HTTP client (see [ADR-0006](adr/0006-distributed-adapter-registry.md)).

This document defines the wire format and semantics that any
"llm-patch-compatible" adapter hub must honor. It complements
[`AdapterManifest` v2](adr/0007-adapter-manifest-v2.md) and the
[`IAdapterRegistryClient`](../projects/llm-patch/src/llm_patch/core/interfaces.py)
ABC.

---

## 1. Goals

- Let agents and CLIs **search**, **resolve**, **pull**, and **push**
  adapters across implementations they did not write.
- Stay transport-light: HTTP/JSON + multipart for uploads. No bespoke
  binary protocol.
- Be reusable for public community hubs, private enterprise mirrors,
  and even single-file static buckets (read-only subset).

## 2. URI Grammar

```
hub-uri    := "hub://" namespace "/" name [":" version]
namespace  := <lower-kebab token>      ; e.g. "acme", "seismael"
name       := <lower-kebab token>      ; e.g. "react-19-docs"
version    := SemVer | "latest"        ; e.g. "1.2.0", "v1.2.0", "latest"
```

`hf://`, `s3://`, and `oci://` are reserved for client-specific
implementations and dispatch through whatever `IAdapterRegistryClient`
the operator registers (see [ADR-0006](adr/0006-distributed-adapter-registry.md)).

## 3. REST Endpoints (v1)

All paths are under `/v1`. All requests/responses are `application/json`
unless noted. Authentication is `Authorization: Bearer <token>` for
write endpoints; reads MAY be public.

### 3.1 `GET /v1/health`

Liveness probe. Returns `{"status": "ok", "version": "<server-version>"}`.

### 3.2 `GET /v1/search?q=<query>&limit=<n>&tag=<t>`

Full-text + tag search.

**Response 200:**
```json
{
  "results": [
    {
      "adapter_id": "acme__react-19__v1.2.0",
      "namespace": "acme",
      "name": "react-19",
      "version": "1.2.0",
      "description": "React 19 hooks and Server Components reference.",
      "tags": ["react", "frontend"],
      "base_model_compatibility": ["google/gemma-2-2b-it"]
    }
  ]
}
```

### 3.3 `GET /v1/adapters/{namespace}/{name}/{version}`

Resolve a single adapter to its manifest. `version` accepts `latest`.

**Response 200:** a full [Manifest v2 document](#5-manifest-v2-schema).

**Response 404:** `{"error": "AdapterNotFoundError", "message": "..."}`.

### 3.4 `GET /v1/adapters/{namespace}/{name}/{version}/payload`

Download the safetensors blob.

- `Content-Type: application/octet-stream`
- `Digest: sha-256=<hex>` MUST match `manifest.checksum_sha256`.
- Clients MUST verify the checksum and raise
  `ChecksumMismatchError` on mismatch.

### 3.5 `POST /v1/adapters` (auth required)

Multipart upload.

- Part `manifest`: `application/json` — Manifest v2 document.
- Part `payload`: `application/octet-stream` — the `.safetensors` file.

The server MUST recompute SHA-256 over `payload`, compare with
`manifest.checksum_sha256`, and reject with `400` on mismatch.

**Response 201:** the canonicalized manifest (server may inject
`created_at`, `storage_uri`).

### 3.6 `DELETE /v1/adapters/{namespace}/{name}/{version}` (auth required)

Soft- or hard-delete is server-defined. Clients SHOULD treat `404` as
"already gone" (idempotent).

## 4. Error Model

All error responses use:

```json
{ "error": "<ErrorClassName>", "message": "<human-readable>", "details": {...} }
```

Stable error names mapped from `llm_patch_shared.errors`:

| HTTP | Error name | Meaning |
|---|---|---|
| 400 | `ValidationError` | Malformed manifest or URI. |
| 401 | `AuthenticationError` | Missing/invalid bearer. |
| 403 | `AuthorizationError` | Forbidden namespace. |
| 404 | `AdapterNotFoundError` | Unknown ref. |
| 409 | `IncompatibleBaseModelError` | Base-model mismatch. |
| 502 | `ChecksumMismatchError` | Payload digest disagreed. |
| 503 | `RegistryUnavailableError` | Registry down or unconfigured. |

## 5. Manifest v2 Schema

```jsonc
{
  "manifest_version": 2,
  "adapter_id": "acme__react-19__v1.2.0", // filesystem-safe
  "namespace": "acme",                    // lower-kebab
  "name": "react-19",                     // lower-kebab
  "version": "1.2.0",                     // SemVer (or "vX.Y.Z")
  "description": "React 19 reference.",
  "tags": ["react", "frontend"],
  "base_model_compatibility": ["google/gemma-2-2b-it"],
  "rank": 8,
  "target_modules": ["q_proj", "v_proj"],
  "storage_uri": "https://hub.example.com/v1/adapters/acme/react-19/1.2.0/payload",
  "checksum_sha256": "<64-hex>",
  "created_at": "2026-04-26T12:00:00Z"
}
```

Validation rules (enforced by Pydantic on the client side):

- `version` matches `^v?\d+\.\d+\.\d+(-[A-Za-z0-9]+)?$`.
- `checksum_sha256` matches `^[0-9a-f]{64}$`.
- `namespace` and `name` match `^[a-z0-9][a-z0-9-]*$`.

## 6. Checksum Verification Flow

```
1. client.pull(ref)
2. → resolve(ref) → manifest
3. → GET .../payload (stream to temp file)
4. → compute sha256(temp)
5. → if digest != manifest.checksum_sha256: raise ChecksumMismatchError
6. → move temp into IAdapterRepository under canonical adapter_id
7. → return manifest
```

The reference `IAdapterRegistryClient` contract requires step 5: a
client MUST NOT return a manifest whose payload it could not verify.

## 7. Compatibility Notes

- Servers MAY ignore unknown manifest fields on write but MUST round-trip
  them on read.
- Clients MUST tolerate v1 manifests on read (no `manifest_version`
  field, missing optional fields default to `None`/`[]`).
- The `latest` alias is resolved server-side; clients SHOULD record the
  resolved concrete version locally for reproducibility.

## 8. Out of Scope (deferred)

- Cryptographic signing (`manifest.signature`) — checksum only for now.
- Mutable tags ("staging", "prod") — versions are immutable.
- Federation between hubs — single registry per `IAdapterRegistryClient`
  instance.
- LoRAX-aware batch endpoints — see
  [SERVER_ARCHITECTURE.md](SERVER_ARCHITECTURE.md).
