"""Tests for :mod:`llm_patch_wiki_agent.routing`."""

from __future__ import annotations

from pathlib import Path

from llm_patch_wiki_agent.registry import AdapterMetadata, SidecarMetadataRegistry
from llm_patch_wiki_agent.routing import (
    IAdapterRouter,
    MetadataExactMatchRouter,
    RouteRequest,
)


def _make_registry(tmp_path: Path, *records: AdapterMetadata) -> SidecarMetadataRegistry:
    registry = SidecarMetadataRegistry(tmp_path)
    for record in records:
        registry.save(record)
    return registry


def test_router_implements_iadapter_router(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    router = MetadataExactMatchRouter(registry)
    assert isinstance(router, IAdapterRouter)


def test_routes_by_context_id(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path,
        AdapterMetadata(adapter_id="a1", context_id="api-v2-auth"),
        AdapterMetadata(adapter_id="a2", context_id="api-v3-auth"),
    )
    router = MetadataExactMatchRouter(registry)

    decision = router.route(
        RouteRequest(query="how do i auth?", metadata={"context_id": "api-v2-auth"})
    )

    assert decision is not None
    assert decision.adapter_id == "a1"
    assert "context_id" in decision.reason


def test_routes_by_adapter_id_when_no_context(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path, AdapterMetadata(adapter_id="a1"))
    router = MetadataExactMatchRouter(registry)

    decision = router.route(RouteRequest(query="anything", metadata={"adapter_id": "a1"}))

    assert decision is not None
    assert decision.adapter_id == "a1"
    assert "adapter_id" in decision.reason


def test_returns_none_when_no_match(tmp_path: Path) -> None:
    registry = _make_registry(
        tmp_path, AdapterMetadata(adapter_id="a1", context_id="api-v2-auth")
    )
    router = MetadataExactMatchRouter(registry)

    assert router.route(RouteRequest(query="x", metadata={"context_id": "missing"})) is None
    assert router.route(RouteRequest(query="x", metadata={})) is None


def test_refresh_picks_up_newly_added_metadata(tmp_path: Path) -> None:
    registry = _make_registry(tmp_path)
    router = MetadataExactMatchRouter(registry)
    assert router.route(RouteRequest(query="x", metadata={"context_id": "fresh"})) is None

    registry.save(AdapterMetadata(adapter_id="fresh", context_id="fresh"))
    router.refresh()

    decision = router.route(RouteRequest(query="x", metadata={"context_id": "fresh"}))
    assert decision is not None
    assert decision.adapter_id == "fresh"
