"""Pydantic schemas for the HTTP API request/response bodies."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ── Adapters ──────────────────────────────────────────────────────────


class AdapterInfo(BaseModel):
    adapter_id: str
    rank: int
    target_modules: list[str]
    storage_uri: str
    namespace: str | None = None
    version: str | None = None
    checksum_sha256: str | None = None
    base_model_compatibility: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: str | None = None


# ── Adapter Market / Hot-swap ─────────────────────────────────────────


class AttachRequest(BaseModel):
    """Request body for ``POST /adapters/attach``."""

    ref: str = Field(..., description="hub://owner/name[:version] reference.")


class DetachRequest(BaseModel):
    """Request body for ``POST /adapters/detach``."""

    adapter_id: str


class ActiveAdapters(BaseModel):
    active: list[str]


class CacheStats(BaseModel):
    """Snapshot of the adapter cache (manifests-only)."""

    size: int
    capacity: int


class HubSearchResult(BaseModel):
    """Single hit returned by the registry search."""

    adapter_id: str
    namespace: str | None = None
    version: str | None = None
    description: str | None = None
    tags: list[str] = Field(default_factory=list)


class CompileRequest(BaseModel):
    """Request to compile a single document."""

    document_id: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


class CompileResponse(BaseModel):
    adapter_id: str
    storage_uri: str


class CompileAllResponse(BaseModel):
    compiled: int
    adapters: list[AdapterInfo]


# ── Inference ─────────────────────────────────────────────────────────


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True


class GenerateResponse(BaseModel):
    text: str


class ChatMessageSchema(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessageSchema]
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True


class ChatResponse(BaseModel):
    message: ChatMessageSchema


# ── Sources ───────────────────────────────────────────────────────────


class DocumentInfo(BaseModel):
    document_id: str
    content_length: int
    metadata: dict[str, object] = Field(default_factory=dict)


class DocumentDetail(BaseModel):
    document_id: str
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)


# ── Health ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
