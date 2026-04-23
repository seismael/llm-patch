"""Unit tests for Pydantic domain models and configuration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_patch.core.config import GeneratorConfig, StorageConfig, WatcherConfig
from llm_patch.core.models import AdapterManifest, DocumentContext


class TestDocumentContext:
    def test_creation_with_required_fields(self) -> None:
        doc = DocumentContext(document_id="test", content="Hello world")
        assert doc.document_id == "test"
        assert doc.content == "Hello world"
        assert doc.metadata == {}

    def test_creation_with_metadata(self) -> None:
        doc = DocumentContext(
            document_id="test",
            content="Hello",
            metadata={"source": "file.md"},
        )
        assert doc.metadata["source"] == "file.md"

    def test_frozen_model_prevents_mutation(self) -> None:
        doc = DocumentContext(document_id="test", content="Hello")
        with pytest.raises(ValidationError):
            doc.document_id = "changed"  # type: ignore[misc]

    def test_missing_required_field_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DocumentContext(document_id="test")  # type: ignore[call-arg]

    def test_missing_document_id_raises_validation_error(self) -> None:
        with pytest.raises(ValidationError):
            DocumentContext(content="hello")  # type: ignore[call-arg]


class TestAdapterManifest:
    def test_creation(self) -> None:
        manifest = AdapterManifest(
            adapter_id="api_v2",
            rank=8,
            target_modules=["q_proj", "v_proj"],
            storage_uri="/adapters/api_v2",
        )
        assert manifest.adapter_id == "api_v2"
        assert manifest.rank == 8
        assert len(manifest.target_modules) == 2
        assert isinstance(manifest.created_at, datetime)

    def test_created_at_defaults_to_utc_now(self) -> None:
        before = datetime.now(UTC)
        manifest = AdapterManifest(
            adapter_id="test",
            rank=8,
            target_modules=[],
            storage_uri="/tmp",
        )
        after = datetime.now(UTC)
        assert before <= manifest.created_at <= after

    def test_serialization_roundtrip(self) -> None:
        manifest = AdapterManifest(
            adapter_id="api_v2",
            rank=16,
            target_modules=["q_proj", "k_proj", "v_proj"],
            storage_uri="/adapters/api_v2",
        )
        json_str = manifest.model_dump_json()
        restored = AdapterManifest.model_validate_json(json_str)
        assert restored.adapter_id == manifest.adapter_id
        assert restored.rank == manifest.rank
        assert restored.target_modules == manifest.target_modules
        assert restored.storage_uri == manifest.storage_uri


class TestGeneratorConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        config = GeneratorConfig(checkpoint_dir=tmp_path, device="cpu")
        assert config.checkpoint_dir == tmp_path
        assert config.device == "cpu"

    def test_default_device_is_cuda(self, tmp_path: Path) -> None:
        config = GeneratorConfig(checkpoint_dir=tmp_path)
        assert config.device == "cuda"

    def test_missing_checkpoint_dir_raises(self) -> None:
        with pytest.raises(ValidationError):
            GeneratorConfig()  # type: ignore[call-arg]


class TestWatcherConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        config = WatcherConfig(directory=tmp_path)
        assert config.directory == tmp_path
        assert config.patterns == ["*.md"]
        assert config.recursive is True

    def test_custom_patterns(self, tmp_path: Path) -> None:
        config = WatcherConfig(directory=tmp_path, patterns=["*.txt", "*.md"])
        assert len(config.patterns) == 2

    def test_debounce_default(self, tmp_path: Path) -> None:
        config = WatcherConfig(directory=tmp_path)
        assert config.debounce_seconds == 0.5


class TestStorageConfig:
    def test_valid_config(self, tmp_path: Path) -> None:
        config = StorageConfig(output_dir=tmp_path)
        assert config.output_dir == tmp_path

    def test_missing_output_dir_raises(self) -> None:
        with pytest.raises(ValidationError):
            StorageConfig()  # type: ignore[call-arg]
