"""Pydantic request/response schemas for the wiki-agent FastAPI gateway."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    version: str


class AdapterEntry(BaseModel):
    """One row in :class:`AdaptersResponse`."""

    adapter_id: str
    context_id: str | None = None
    tags: tuple[str, ...] = ()
    summary: str | None = None
    storage_uri: str | None = None


class AdaptersResponse(BaseModel):
    adapters: list[AdapterEntry]


class ChatTurn(BaseModel):
    """One message in a multi-turn conversation."""

    model_config = {"frozen": True}

    role: str
    content: str


class ChatRequest(BaseModel):
    """Inbound chat request. The gateway will route based on ``metadata``."""

    messages: list[ChatTurn]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChatResponseModel(BaseModel):
    """Outbound chat response carrying the routing decision."""

    answer: str
    adapter_id: str
    reason: str


class RouteResponse(BaseModel):
    """Response for ``POST /v1/route``."""

    adapter_id: str
    reason: str
