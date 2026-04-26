"""Adapter routing — Strategy pattern for selecting the right LoRA per request."""

from llm_patch_wiki_agent.routing.interfaces import (
    IAdapterRouter,
    RouteDecision,
    RouteRequest,
)
from llm_patch_wiki_agent.routing.metadata_router import MetadataExactMatchRouter

__all__ = [
    "IAdapterRouter",
    "MetadataExactMatchRouter",
    "RouteDecision",
    "RouteRequest",
]
