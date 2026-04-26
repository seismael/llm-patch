"""Deterministic, metadata-driven router — exact match on ``context_id``."""

from __future__ import annotations

from llm_patch_wiki_agent.registry import AdapterMetadata, SidecarMetadataRegistry
from llm_patch_wiki_agent.routing.interfaces import (
    IAdapterRouter,
    RouteDecision,
    RouteRequest,
)


class MetadataExactMatchRouter(IAdapterRouter):
    """Routes by exact match on ``request.metadata['context_id']``.

    Falls back to matching by ``adapter_id`` directly. Returns ``None`` when
    neither key is provided or when no adapter has the requested context.
    Cross-contamination is impossible: only an exact key match wins.
    """

    def __init__(self, registry: SidecarMetadataRegistry) -> None:
        self._registry = registry
        self._by_context: dict[str, AdapterMetadata] = {}
        self._by_adapter_id: dict[str, AdapterMetadata] = {}
        self.refresh()

    def route(self, request: RouteRequest) -> RouteDecision | None:
        meta = request.metadata
        context_id = meta.get("context_id") if isinstance(meta.get("context_id"), str) else None
        adapter_id = meta.get("adapter_id") if isinstance(meta.get("adapter_id"), str) else None

        if context_id and context_id in self._by_context:
            entry = self._by_context[context_id]
            return RouteDecision(
                adapter_id=entry.adapter_id,
                reason=f"context_id exact match: {context_id!r}",
            )
        if adapter_id and adapter_id in self._by_adapter_id:
            return RouteDecision(
                adapter_id=adapter_id,
                reason=f"adapter_id exact match: {adapter_id!r}",
            )
        return None

    def refresh(self) -> None:
        records = self._registry.list_all()
        self._by_context = {r.context_id: r for r in records if r.context_id}
        self._by_adapter_id = {r.adapter_id: r for r in records}
