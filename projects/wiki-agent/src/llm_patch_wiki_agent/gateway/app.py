"""FastAPI app factory — endpoints for the wiki-agent inference gateway."""

from __future__ import annotations

from typing import TYPE_CHECKING

import llm_patch as engine
from llm_patch_utils import ConfigurationError, LlmPatchError, ResourceNotFoundError

from llm_patch_wiki_agent import __version__
from llm_patch_wiki_agent.gateway.schemas import (
    AdapterEntry,
    AdaptersResponse,
    ChatRequest,
    ChatResponseModel,
    HealthResponse,
    RouteResponse,
)
from llm_patch_wiki_agent.routing import RouteRequest

if TYPE_CHECKING:
    from llm_patch_wiki_agent.gateway.deps import GatewayContext


def create_app(context: "GatewayContext"):  # type: ignore[no-untyped-def]
    """Build a FastAPI app bound to ``context``.

    ``fastapi`` is imported lazily so the wiki-agent can be installed without
    the optional ``[server]`` extra. Importing this function without
    ``fastapi`` available raises a clear :class:`ImportError`.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:  # pragma: no cover - exercised by serve CLI
        raise ImportError(
            "fastapi is not installed. Install with: pip install 'llm-patch-wiki-agent[server]'"
        ) from exc

    app = FastAPI(title="llm-patch-wiki-agent gateway", version=__version__)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/v1/adapters", response_model=AdaptersResponse)
    def list_adapters() -> AdaptersResponse:
        rows = context.list_adapter_entries()
        return AdaptersResponse(
            adapters=[AdapterEntry.model_validate(row) for row in rows]
        )

    @app.post("/v1/route", response_model=RouteResponse)
    def route(request: ChatRequest) -> RouteResponse:
        query = request.messages[-1].content if request.messages else ""
        decision = context.router.route(
            RouteRequest(query=query, metadata=request.metadata)
        )
        if decision is None:
            raise HTTPException(
                status_code=404, detail="No adapter matched the request metadata."
            )
        return RouteResponse(adapter_id=decision.adapter_id, reason=decision.reason)

    @app.post("/v1/chat", response_model=ChatResponseModel)
    def chat(request: ChatRequest) -> ChatResponseModel:
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages must not be empty.")

        query = request.messages[-1].content
        decision = context.router.route(
            RouteRequest(query=query, metadata=request.metadata)
        )
        if decision is None:
            raise HTTPException(
                status_code=404, detail="No adapter matched the request metadata."
            )

        try:
            runtime = context.runtime_for(decision.adapter_id)
        except ResourceNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except LlmPatchError as exc:  # defence-in-depth
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        chat_messages = [
            engine.ChatMessage(role=engine.ChatRole(turn.role), content=turn.content)
            for turn in request.messages
        ]
        response = runtime.chat(chat_messages)
        return ChatResponseModel(
            answer=response.message.content,
            adapter_id=decision.adapter_id,
            reason=decision.reason,
        )

    return app
