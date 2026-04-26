"""Tests for :mod:`llm_patch_wiki_agent.registry.metadata`."""

from __future__ import annotations

from pathlib import Path

import pytest
from llm_patch_utils import ConfigurationError, ResourceNotFoundError

from llm_patch_wiki_agent.registry import AdapterMetadata, SidecarMetadataRegistry


def test_save_then_load_roundtrips_metadata(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)
    metadata = AdapterMetadata(
        adapter_id="api-v2-auth",
        context_id="api-v2-auth",
        tags=("api", "v2", "auth"),
        summary="OAuth flow for the v2 API.",
        source_path="docs/api-v2-auth.md",
    )

    sidecar = registry.save(metadata)

    assert sidecar.name == "api-v2-auth.meta.json"
    loaded = registry.load("api-v2-auth")
    assert loaded.adapter_id == "api-v2-auth"
    assert loaded.context_id == "api-v2-auth"
    assert loaded.tags == ("api", "v2", "auth")
    assert loaded.summary == "OAuth flow for the v2 API."


def test_save_rejects_empty_adapter_id(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)

    with pytest.raises(ConfigurationError):
        registry.save(AdapterMetadata(adapter_id=""))


def test_load_missing_adapter_raises_resource_not_found(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)

    with pytest.raises(ResourceNotFoundError):
        registry.load("does-not-exist")


def test_list_all_skips_malformed_sidecars(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)
    registry.save(AdapterMetadata(adapter_id="alpha", context_id="alpha"))
    registry.save(AdapterMetadata(adapter_id="beta", context_id="beta"))
    (tmp_path / "broken.meta.json").write_text("{not json", encoding="utf-8")

    records = registry.list_all()

    ids = tuple(record.adapter_id for record in records)
    assert ids == ("alpha", "beta")


def test_list_all_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path / "absent")

    assert registry.list_all() == ()


def test_delete_removes_sidecar(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)
    registry.save(AdapterMetadata(adapter_id="alpha"))

    registry.delete("alpha")

    assert not registry.exists("alpha")
    with pytest.raises(ResourceNotFoundError):
        registry.delete("alpha")


def test_upsert_from_payload_normalizes_tags(tmp_path: Path) -> None:
    registry = SidecarMetadataRegistry(tmp_path)

    metadata = registry.upsert_from_payload(
        "api-v2-auth",
        {"context_id": "api-v2-auth", "tags": ["api", "v2"], "summary": "auth"},
    )

    assert metadata.tags == ("api", "v2")
    assert registry.load("api-v2-auth").summary == "auth"
