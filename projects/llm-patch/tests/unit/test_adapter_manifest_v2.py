"""Unit tests for ``AdapterManifest`` v2 fields and ``AdapterRef`` parsing."""

from __future__ import annotations

import pytest

from llm_patch import AdapterManifest, AdapterRef


# ── AdapterRef ──────────────────────────────────────────────────────


class TestAdapterRefParse:
    def test_parses_full_uri(self) -> None:
        ref = AdapterRef.parse("hub://acme/react-19:v1.2.0")
        assert ref.namespace == "acme"
        assert ref.name == "react-19"
        assert ref.version == "v1.2.0"

    def test_defaults_version_to_latest(self) -> None:
        ref = AdapterRef.parse("hub://acme/react-19")
        assert ref.version == "latest"

    def test_round_trip_to_uri(self) -> None:
        original = "hub://acme/react-19:v1.2.0"
        assert AdapterRef.parse(original).to_uri() == original

    def test_adapter_id_is_filesystem_safe(self) -> None:
        ref = AdapterRef.parse("hub://acme/react-19:v1.2.0")
        assert "/" not in ref.adapter_id
        assert ref.adapter_id == "acme__react-19__v1.2.0"

    def test_rejects_non_hub_scheme(self) -> None:
        with pytest.raises(ValueError, match="hub://"):
            AdapterRef.parse("https://acme/react-19")

    def test_rejects_missing_name(self) -> None:
        with pytest.raises(ValueError):
            AdapterRef.parse("hub://acme")

    def test_is_frozen(self) -> None:
        ref = AdapterRef.parse("hub://acme/x")
        with pytest.raises((TypeError, ValueError)):
            ref.namespace = "other"  # type: ignore[misc]


# ── Manifest v2 ─────────────────────────────────────────────────────


class TestAdapterManifestV2:
    def test_v1_round_trip(self) -> None:
        m = AdapterManifest(
            adapter_id="x", rank=8, target_modules=["q"], storage_uri="/tmp"
        )
        assert m.manifest_version == 1
        assert m.namespace is None
        assert m.checksum_sha256 is None
        # Round-trip via JSON keeps v1 defaults.
        restored = AdapterManifest.model_validate_json(m.model_dump_json())
        assert restored == m

    def test_v2_extended_fields(self) -> None:
        m = AdapterManifest(
            adapter_id="x",
            rank=8,
            target_modules=["q"],
            storage_uri="/tmp",
            manifest_version=2,
            namespace="acme/x",
            version="1.2.0",
            checksum_sha256="a" * 64,
            base_model_compatibility=["google/gemma-2-2b-it"],
            tags=["framework", "react"],
            description="Example",
        )
        assert m.namespace == "acme/x"
        assert m.tags == ["framework", "react"]

    def test_invalid_checksum_rejected(self) -> None:
        with pytest.raises(ValueError, match="checksum"):
            AdapterManifest(
                adapter_id="x",
                rank=8,
                target_modules=["q"],
                storage_uri="/tmp",
                checksum_sha256="not-hex",
            )

    def test_invalid_version_rejected(self) -> None:
        with pytest.raises(ValueError, match="SemVer"):
            AdapterManifest(
                adapter_id="x",
                rank=8,
                target_modules=["q"],
                storage_uri="/tmp",
                version="not.a.version!",
            )

    def test_invalid_namespace_rejected(self) -> None:
        with pytest.raises(ValueError, match="namespace"):
            AdapterManifest(
                adapter_id="x",
                rank=8,
                target_modules=["q"],
                storage_uri="/tmp",
                namespace="UPPER/CASE",
            )
