"""FastAPI inference gateway — Phase 2 of the Living Wiki Copilot use case."""

from llm_patch_wiki_agent.gateway.app import create_app
from llm_patch_wiki_agent.gateway.deps import GatewayContext
from llm_patch_wiki_agent.gateway.schemas import (
    AdapterEntry,
    AdaptersResponse,
    ChatRequest,
    ChatResponseModel,
    ChatTurn,
    HealthResponse,
    RouteResponse,
)

__all__ = [
    "AdapterEntry",
    "AdaptersResponse",
    "ChatRequest",
    "ChatResponseModel",
    "ChatTurn",
    "GatewayContext",
    "HealthResponse",
    "RouteResponse",
    "create_app",
]
