"""Pydantic domain models for the llm_patch library."""

from __future__ import annotations

from datetime import UTC, datetime
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
