"""Integration: manifest v2 survives a full publish -> search -> pull cycle.

Exercises the contract :class:`IAdapterRegistryClient` plugins must
honor by running an in-memory fake client through the round-trip a real
publish/pull would take. All v2 manifest fields (namespace, version,
checksum_sha256, base_model_compatibility, tags, description) must be
preserved bit-for-bit when serialized to JSON and back.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from llm_patch import AdapterManifest, AdapterRef
from llm_patch.core.interfaces import IAdapterRegistryClient


@dataclass
class _InMemoryRegistry(IAdapterRegistryClient):
    """Round-trip fake: stores manifests as JSON, just like a real hub."""

    storage: dict[str, str] = field(default_factory=dict)
    blobs: dict[str, bytes] = field(default_factory=dict)

    def _key(self, ref: AdapterRef) -> str:
        return f"{ref.namespace}/{ref.name}:{ref.version}"

    def search(self, query: str, *, limit: int = 10) -> list[AdapterManifest]:
        results: list[AdapterManifest] = []
        for raw in self.storage.values():
            manifest = AdapterManifest.model_validate_json(raw)
            haystack = " ".join(
                [
                    manifest.namespace or "",
                    manifest.description or "",
                    *(manifest.tags or []),
                ]
            ).lower()
            if query.lower() in haystack:
                results.append(manifest)
            if len(results) >= limit:
                break
        return results

    def resolve(self, ref: AdapterRef) -> AdapterManifest:
        raw = self.storage.get(self._key(ref))
        if raw is None:
            raise KeyError(f"unknown ref: {ref.to_uri()}")
        return AdapterManifest.model_validate_json(raw)

    def pull(self, ref: AdapterRef) -> AdapterManifest:
        manifest = self.resolve(ref)
        # Verify checksum if declared.
        if manifest.checksum_sha256 is not None:
            blob = self.blobs[self._key(ref)]
            actual = hashlib.sha256(blob).hexdigest()
            if actual != manifest.checksum_sha256:
                from llm_patch_utils import ChecksumMismatchError

                raise ChecksumMismatchError(
                    f"checksum mismatch: expected {manifest.checksum_sha256}, got {actual}"
                )
        return manifest

    def push(self, adapter_id: str, ref: AdapterRef) -> AdapterManifest:
        # The fake just stores a synthetic manifest derived from the id.
        manifest = AdapterManifest(
            adapter_id=adapter_id,
            rank=8,
            target_modules=["q_proj"],
            storage_uri=f"hub://{ref.namespace}/{ref.name}",
            manifest_version=2,
            namespace=f"{ref.namespace}/{ref.name}",
            version=ref.version if ref.version != "latest" else "1.0.0",
            checksum_sha256="b" * 64,
            base_model_compatibility=["google/gemma-2-2b-it"],
            tags=["test"],
            description="Pushed by test",
        )
        self.storage[self._key(ref)] = manifest.model_dump_json()
        self.blobs[self._key(ref)] = b"test-weights"
        return manifest


@pytest.fixture
def manifest() -> AdapterManifest:
    blob = b"adapter-weights-blob"
    return AdapterManifest(
        adapter_id="acme__react-19__1.0.0",
        rank=8,
        target_modules=["q_proj", "v_proj"],
        storage_uri="/local/adapters/acme__react-19__1.0.0",
        manifest_version=2,
        namespace="acme/react-19",
        version="1.0.0",
        checksum_sha256=hashlib.sha256(blob).hexdigest(),
        base_model_compatibility=["google/gemma-2-2b-it", "meta-llama/Llama-3-8B"],
        tags=["framework", "react", "frontend"],
        description="React 19 conventions and patterns.",
    )


class TestManifestV2Roundtrip:
    def test_v2_fields_survive_publish_resolve_pull(
        self, manifest: AdapterManifest
    ) -> None:
        registry = _InMemoryRegistry()
        ref = AdapterRef.parse("hub://acme/react-19:1.0.0")
        # Publish (using the manifest verbatim — bypass the synthetic
        # push to verify *all* declared fields survive the wire).
        registry.storage[registry._key(ref)] = manifest.model_dump_json()
        registry.blobs[registry._key(ref)] = b"adapter-weights-blob"

        resolved = registry.resolve(ref)
        pulled = registry.pull(ref)

        for got in (resolved, pulled):
            assert got.adapter_id == manifest.adapter_id
            assert got.manifest_version == 2
            assert got.namespace == "acme/react-19"
            assert got.version == "1.0.0"
            assert got.checksum_sha256 == manifest.checksum_sha256
            assert got.base_model_compatibility == manifest.base_model_compatibility
            assert got.tags == manifest.tags
            assert got.description == manifest.description
            assert got.rank == 8
            assert got.target_modules == ["q_proj", "v_proj"]

    def test_search_finds_by_tag_and_namespace(
        self, manifest: AdapterManifest
    ) -> None:
        registry = _InMemoryRegistry()
        ref = AdapterRef.parse("hub://acme/react-19:1.0.0")
        registry.storage[registry._key(ref)] = manifest.model_dump_json()
        registry.blobs[registry._key(ref)] = b"adapter-weights-blob"

        by_tag = registry.search("framework")
        by_ns = registry.search("acme")
        miss = registry.search("nonexistent-term")

        assert len(by_tag) == 1
        assert by_tag[0].namespace == "acme/react-19"
        assert len(by_ns) == 1
        assert miss == []

    def test_pull_raises_on_checksum_mismatch(
        self, manifest: AdapterManifest
    ) -> None:
        from llm_patch_utils import ChecksumMismatchError

        registry = _InMemoryRegistry()
        ref = AdapterRef.parse("hub://acme/react-19:1.0.0")
        registry.storage[registry._key(ref)] = manifest.model_dump_json()
        # Tamper with the blob — checksum will no longer match.
        registry.blobs[registry._key(ref)] = b"tampered-bytes"

        with pytest.raises(ChecksumMismatchError):
            registry.pull(ref)

    def test_round_trip_via_disk_preserves_all_fields(
        self, manifest: AdapterManifest, tmp_path: Path
    ) -> None:
        # Simulate a registry that stores manifests on disk between
        # processes — JSON written and re-read should be byte-identical
        # in every public field.
        path = tmp_path / "manifest.json"
        path.write_text(manifest.model_dump_json(), encoding="utf-8")

        on_disk = json.loads(path.read_text(encoding="utf-8"))
        restored = AdapterManifest.model_validate(on_disk)

        assert restored == manifest
