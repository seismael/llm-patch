"""Tests for :mod:`llm_patch_wiki_agent.mcp_server`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import llm_patch as engine
import pytest
from llm_patch_utils import ConfigurationError

from llm_patch_wiki_agent import WikiAgentConfig
from llm_patch_wiki_agent.gateway import GatewayContext
from llm_patch_wiki_agent.mcp_server import build_tools
from llm_patch_wiki_agent.registry import AdapterMetadata, SidecarMetadataRegistry
from llm_patch_wiki_agent.routing import MetadataExactMatchRouter


class _FakeRepository(engine.IAdapterRepository):
    def __init__(self) -> None:
        self._store: dict[str, engine.AdapterManifest] = {}

    def save(self, adapter_id, weights, peft_config):  # type: ignore[no-untyped-def]
        manifest = engine.AdapterManifest(
            adapter_id=adapter_id,
            rank=8,
            target_modules=["q_proj"],
            storage_uri=f"adapters/{adapter_id}",
        )
        self._store[adapter_id] = manifest
        return manifest

    def load(self, adapter_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def exists(self, adapter_id: str) -> bool:
        return adapter_id in self._store

    def list_adapters(self) -> list[engine.AdapterManifest]:
        return list(self._store.values())

    def delete(self, adapter_id: str) -> None:
        self._store.pop(adapter_id, None)


class _FakeCompilePipeline:
    def __init__(self, repository: _FakeRepository) -> None:
        self._repository = repository
        self.calls: list[str] = []

    def process_document(self, context: engine.DocumentContext) -> engine.AdapterManifest:
        self.calls.append(context.document_id)
        return self._repository.save(context.document_id, {}, {})


class _FakeRuntime(engine.IAgentRuntime):
    def generate(self, prompt: str, **kwargs: Any) -> str:
        return prompt

    def chat(self, messages, **kwargs: Any) -> engine.ChatResponse:
        return engine.ChatResponse(
            message=engine.ChatMessage(
                role=engine.ChatRole.ASSISTANT,
                content=f"answered: {messages[-1].content}",
            )
        )

    def stream(self, prompt: str, **kwargs: Any):  # pragma: no cover
        yield prompt


class _FakeUsePipeline:
    def build_agent(
        self, model_id: str, adapter_ids: list[str] | None = None, **kwargs: Any
    ) -> engine.IAgentRuntime:
        return _FakeRuntime()


def _make_context(tmp_path: Path) -> tuple[GatewayContext, _FakeCompilePipeline]:
    registry = SidecarMetadataRegistry(tmp_path)
    repository = _FakeRepository()
    config = WikiAgentConfig(adapter_dir=tmp_path, model_id="demo/model")
    context = GatewayContext(
        config=config,
        repository=repository,
        metadata_registry=registry,
        router=MetadataExactMatchRouter(registry),
        use_pipeline=_FakeUsePipeline(),
    )
    pipeline = _FakeCompilePipeline(repository)
    context.attach_compile_pipeline(pipeline)
    return context, pipeline


def test_internalize_knowledge_compiles_and_registers_metadata(tmp_path: Path) -> None:
    context, pipeline = _make_context(tmp_path)
    doc_path = tmp_path / "api-v2-auth.md"
    doc_path.write_text("# OAuth flow", encoding="utf-8")

    tools = build_tools(context)
    result = tools["internalize_knowledge"](
        str(doc_path),
        context_id="api-v2-auth",
        tags=["api", "v2"],
        summary="OAuth flow",
    )

    assert result["adapter_id"] == "api-v2-auth"
    assert result["context_id"] == "api-v2-auth"
    assert pipeline.calls == ["api-v2-auth"]
    saved = context.metadata_registry.load("api-v2-auth")
    assert saved.tags == ("api", "v2")
    # And the router is now refreshed.
    from llm_patch_wiki_agent.routing import RouteRequest

    decision = context.router.route(
        RouteRequest(query="x", metadata={"context_id": "api-v2-auth"})
    )
    assert decision is not None
    assert decision.adapter_id == "api-v2-auth"


def test_internalize_knowledge_rejects_missing_path(tmp_path: Path) -> None:
    context, _ = _make_context(tmp_path)
    tools = build_tools(context)

    with pytest.raises(ConfigurationError):
        tools["internalize_knowledge"](str(tmp_path / "missing.md"))


def test_list_adapters_returns_metadata_join(tmp_path: Path) -> None:
    context, _ = _make_context(tmp_path)
    context.metadata_registry.save(
        AdapterMetadata(adapter_id="alpha", context_id="alpha", tags=("a",))
    )
    context.repository.save("alpha", {}, {})

    tools = build_tools(context)
    rows = tools["list_adapters"]()

    assert any(
        row["adapter_id"] == "alpha" and row["tags"] == ["a"] for row in rows
    )


def test_chat_with_adapter_uses_runtime(tmp_path: Path) -> None:
    context, _ = _make_context(tmp_path)
    context.repository.save("alpha", {}, {})

    tools = build_tools(context)
    result = tools["chat_with_adapter"]("alpha", "hello")

    assert result == {"adapter_id": "alpha", "answer": "answered: hello"}


def test_build_server_registers_three_tools(tmp_path: Path) -> None:
    pytest.importorskip("mcp")
    context, _ = _make_context(tmp_path)

    from llm_patch_wiki_agent.mcp_server import build_server

    server = build_server(context)
    # FastMCP keeps a tool manager with registered tools.
    tool_names = {t.name for t in server._tool_manager.list_tools()}
    assert {"internalize_knowledge", "list_adapters", "chat_with_adapter"} <= tool_names
