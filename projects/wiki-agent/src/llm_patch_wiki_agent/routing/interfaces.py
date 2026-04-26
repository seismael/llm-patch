"""Routing abstractions — :class:`IAdapterRouter` (Strategy pattern)."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel


class RouteRequest(BaseModel):
    """Inbound request to the router.

    Attributes:
        query: The end-user prompt (may be unused by deterministic routers).
        metadata: Free-form context attached by the gateway (e.g.
            ``{"context_id": "api-v2-auth"}``).
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    query: str
    metadata: Mapping[str, Any] = {}


class RouteDecision(BaseModel):
    """Result of a successful routing decision.

    Attributes:
        adapter_id: The adapter to attach for this request.
        reason: Short explanation of why this adapter was chosen
            (used in logs and the response payload).
    """

    model_config = {"frozen": True}

    adapter_id: str
    reason: str


class IAdapterRouter(abc.ABC):
    """Strategy interface — pick an adapter for a given request.

    Implementations should be **side-effect free** and **fast**: routers run
    on the request hot-path. Heavy state (e.g. embedding indexes) belongs in
    the constructor and should be refreshable via :meth:`refresh`.
    """

    @abc.abstractmethod
    def route(self, request: RouteRequest) -> RouteDecision | None:
        """Return a :class:`RouteDecision` or ``None`` when nothing matches."""

    def refresh(self) -> None:
        """Re-read any external state (e.g. the metadata registry).

        Default: no-op. Override when the router caches data at construction.
        """
        return None
