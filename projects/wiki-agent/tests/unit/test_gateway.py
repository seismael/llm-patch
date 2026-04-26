"""Tests for :mod:`llm_patch_wiki_agent.gateway`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import llm_patch as engine
import pytest

from llm_patch_wiki_agent import WikiAgentConfig
from llm_patch_wiki_agent.gateway import GatewayContext, create_app
from llm_patch_wiki_agent.registry import AdapterMetadata, SidecarMetadataRegistry
from llm_patch_wiki_agent.routing import MetadataExactMatchRouter

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


class _FakeRepository(engine.IAdapterRepository):
    def __init__(self, manifests: list[engine.AdapterManifest]) -> None:
        self._manifests = manifests

    def save(self, adapter_id, weights, peft_config):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def load(self, adapter_id):  # type: ignore[no-untyped-def]
        raise NotImplementedError

    def exists(self, adapter_id: str) -> bool:
        return any(m.adapter_id == adapter_id for m in self._manifests)

    def list_adapters(self) -> list[engine.AdapterManifest]:
        return list(self._manifests)

    def delete(self, adapter_id: str) -> None:
        self._manifests = [m for m in self._manifests if m.adapter_id != adapter_id]


class _FakeRuntime(engine.IAgentRuntime):
    def __init__(self, label: str) -> None:
        self._label = label

    def generate(self, prompt: str, **kwargs: Any) -> str:
        return f"{self._label}: {prompt}"

    def chat(self, messages: list[engine.ChatMessage], **kwargs: Any) -> engine.ChatResponse:
        last = messages[-1].content if messages else ""
        return engine.ChatResponse(
            message=engine.ChatMessage(
                role=engine.ChatRole.ASSISTANT,
                content=f"[{self._label}] {last}",
            )
        )

    def stream(self, prompt: str, **kwargs: Any):  # pragma: no cover
        yield f"{self._label}: {prompt}"


class _FakeUsePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def build_agent(
        self,
        model_id: str,
        adapter_ids: list[str] | None = None,
        **model_kwargs: Any,
    ) -> engine.IAgentRuntime:
        self.calls.append(
            {"model_id": model_id, "adapter_ids": adapter_ids, "kwargs": model_kwargs}
        )
        label = (adapter_ids or ["unknown"])[0]
        return _FakeRuntime(label)


def _build_context(tmp_path: Path) -> tuple[GatewayContext, _FakeUsePipeline]:
    registry = SidecarMetadataRegistry(tmp_path)
    registry.save(AdapterMetadata(adapter_id="api-v2-auth", context_id="api-v2-auth"))
    registry.save(AdapterMetadata(adapter_id="api-v3-auth", context_id="api-v3-auth"))

    repository = _FakeRepository(
        [
            engine.AdapterManifest(
                adapter_id="api-v2-auth",
                rank=8,
                target_modules=["q_proj"],
                storage_uri="adapters/api-v2-auth",
            ),
            engine.AdapterManifest(
                adapter_id="api-v3-auth",
                rank=8,
                target_modules=["q_proj"],
                storage_uri="adapters/api-v3-auth",
            ),
        ]
    )

    use_pipeline = _FakeUsePipeline()
    config = WikiAgentConfig(adapter_dir=tmp_path, model_id="demo/model")
    context = GatewayContext(
        config=config,
        repository=repository,
        metadata_registry=registry,
        router=MetadataExactMatchRouter(registry),
        use_pipeline=use_pipeline,
    )
    return context, use_pipeline


def test_health_endpoint_returns_ok(tmp_path: Path) -> None:
    context, _ = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_list_adapters_joins_metadata_with_manifests(tmp_path: Path) -> None:
    context, _ = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.get("/v1/adapters")

    assert response.status_code == 200
    body = response.json()
    ids = {entry["adapter_id"] for entry in body["adapters"]}
    assert ids == {"api-v2-auth", "api-v3-auth"}
    storage_uris = {entry["storage_uri"] for entry in body["adapters"]}
    assert all(uri is not None for uri in storage_uris)


def test_route_returns_decision_when_metadata_matches(tmp_path: Path) -> None:
    context, _ = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.post(
        "/v1/route",
        json={
            "messages": [{"role": "user", "content": "auth?"}],
            "metadata": {"context_id": "api-v2-auth"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["adapter_id"] == "api-v2-auth"
    assert "context_id" in body["reason"]


def test_route_returns_404_when_no_match(tmp_path: Path) -> None:
    context, _ = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.post(
        "/v1/route",
        json={
            "messages": [{"role": "user", "content": "auth?"}],
            "metadata": {"context_id": "nope"},
        },
    )

    assert response.status_code == 404


def test_chat_invokes_runtime_for_routed_adapter(tmp_path: Path) -> None:
    context, use_pipeline = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.post(
        "/v1/chat",
        json={
            "messages": [{"role": "user", "content": "How do I auth?"}],
            "metadata": {"context_id": "api-v2-auth"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["adapter_id"] == "api-v2-auth"
    assert "[api-v2-auth] How do I auth?" in body["answer"]
    # Runtime cache: a second request for the same adapter must not rebuild.
    client.post(
        "/v1/chat",
        json={
            "messages": [{"role": "user", "content": "again"}],
            "metadata": {"context_id": "api-v2-auth"},
        },
    )
    assert len(use_pipeline.calls) == 1


def test_chat_returns_400_for_empty_messages(tmp_path: Path) -> None:
    context, _ = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.post("/v1/chat", json={"messages": [], "metadata": {}})

    assert response.status_code == 400


def test_chat_returns_404_when_no_route(tmp_path: Path) -> None:
    context, _ = _build_context(tmp_path)
    client = TestClient(create_app(context))

    response = client.post(
        "/v1/chat",
        json={
            "messages": [{"role": "user", "content": "anything"}],
            "metadata": {"context_id": "missing"},
        },
    )

    assert response.status_code == 404


def test_runtime_for_requires_model_id(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)
    registry.save(AdapterMetadata(adapter_id="a1", context_id="a1"))
    repository = _FakeRepository(
        [
            engine.AdapterManifest(
                adapter_id="a1", rank=8, target_modules=["q"], storage_uri="x"
            )
        ]
    )
    config = WikiAgentConfig(adapter_dir=tmp_path)  # no model_id
    context = GatewayContext(
        config=config,
        repository=repository,
        metadata_registry=registry,
        router=MetadataExactMatchRouter(registry),
        use_pipeline=cast(Any, _FakeUsePipeline()),
    )

    from llm_patch_shared import ConfigurationError

    with pytest.raises(ConfigurationError):
        context.runtime_for("a1")
