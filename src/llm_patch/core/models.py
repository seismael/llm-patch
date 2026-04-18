"""Pydantic domain models for the llm_patch library."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DocumentContext(BaseModel):
    """Immutable representation of ingested knowledge.

    Attributes:
        document_id: Unique identifier derived from document filename stem.
        content: Raw document text content.
        metadata: Arbitrary metadata (source path, modified time, etc.).
    """

    model_config = {"frozen": True}

    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterManifest(BaseModel):
    """Tracks a generated LoRA adapter and its storage location.

    Attributes:
        adapter_id: Unique identifier matching the source document_id.
        rank: LoRA rank (r) used for the adapter.
        target_modules: List of model modules targeted by the adapter.
        storage_uri: Path or URI to the adapter directory.
        created_at: UTC timestamp of adapter creation.
    """

    adapter_id: str
    rank: int
    target_modules: list[str]
    storage_uri: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


# ── Model Handles ─────────────────────────────────────────────────────


class ModelHandle(BaseModel):
    """Lightweight wrapper around a loaded model + tokenizer.

    The ``model`` and ``tokenizer`` fields hold runtime objects (not
    serializable) so they use ``Any``.  Everything else is plain data.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    model: Any = Field(exclude=True)
    tokenizer: Any = Field(exclude=True)
    base_model_id: str
    attached_adapters: tuple[str, ...] = ()
    device: str = "cpu"


# ── Chat / Generation ─────────────────────────────────────────────────


class ChatRole(str, Enum):
    """Standard chat-message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a multi-turn conversation."""

    model_config = {"frozen": True}

    role: ChatRole
    content: str


class GenerationOptions(BaseModel):
    """Knobs exposed to callers for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


class ChatResponse(BaseModel):
    """Response from ``IAgentRuntime.chat``."""

    message: ChatMessage
    usage: dict[str, int] = Field(default_factory=dict)


# ── Data-Source Registry ──────────────────────────────────────────────


class DataSourceDescriptor(BaseModel):
    """Metadata entry describing an available data source type."""

    source_type: str
    description: str
    config_schema: dict[str, Any] = Field(default_factory=dict)
