"""Unit tests for LocalSafetensorsRepository (Repository Pattern)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch not available", allow_module_level=True)

from llm_patch.core.config import StorageConfig
from llm_patch.core.models import AdapterManifest
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository


@pytest.fixture()
def repo(tmp_path: Path) -> LocalSafetensorsRepository:
    config = StorageConfig(output_dir=tmp_path / "adapters")
    return LocalSafetensorsRepository(config)


@pytest.fixture()
def peft_config_mock() -> MagicMock:
    config = MagicMock()
    config.to_dict.return_value = {
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "lora_alpha": 16,
        "peft_type": "LORA",
    }
    return config


@pytest.fixture()
def weights() -> dict[str, torch.Tensor]:
    return {
        "lora_A": torch.randn(8, 256),
        "lora_B": torch.randn(256, 8),
    }


class TestSave:
    def test_creates_safetensors_file(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        repo.save("test_adapter", weights, peft_config_mock)
        assert (Path(repo._output_dir) / "test_adapter" / "adapter_model.safetensors").exists()

    def test_creates_adapter_config_json(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        repo.save("test_adapter", weights, peft_config_mock)
        assert (Path(repo._output_dir) / "test_adapter" / "adapter_config.json").exists()

    def test_creates_manifest_json(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        repo.save("test_adapter", weights, peft_config_mock)
        assert (Path(repo._output_dir) / "test_adapter" / "manifest.json").exists()

    def test_returns_adapter_manifest(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        manifest = repo.save("test_adapter", weights, peft_config_mock)
        assert isinstance(manifest, AdapterManifest)
        assert manifest.adapter_id == "test_adapter"
        assert manifest.rank == 8
        assert manifest.target_modules == ["q_proj", "v_proj"]

    def test_save_with_dict_config(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
    ) -> None:
        config_dict: dict[str, Any] = {
            "r": 16,
            "target_modules": ["q_proj"],
            "peft_type": "LORA",
        }
        manifest = repo.save("dict_adapter", weights, config_dict)
        assert manifest.rank == 16


class TestLoad:
    def test_load_returns_correct_tensors(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        repo.save("test_adapter", weights, peft_config_mock)
        loaded = repo.load("test_adapter")

        assert set(loaded.keys()) == set(weights.keys())
        for key in weights:
            assert torch.allclose(loaded[key], weights[key])

    def test_load_nonexistent_raises_file_not_found(self, repo: LocalSafetensorsRepository) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            repo.load("nonexistent")


class TestExists:
    def test_exists_true_after_save(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        repo.save("test_adapter", weights, peft_config_mock)
        assert repo.exists("test_adapter") is True

    def test_exists_false_before_save(self, repo: LocalSafetensorsRepository) -> None:
        assert repo.exists("nonexistent") is False


class TestListAdapters:
    def test_list_returns_all_saved(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        for name in ["adapter_a", "adapter_b", "adapter_c"]:
            repo.save(name, weights, peft_config_mock)

        adapters = repo.list_adapters()
        assert len(adapters) == 3
        ids = {a.adapter_id for a in adapters}
        assert ids == {"adapter_a", "adapter_b", "adapter_c"}

    def test_list_empty_returns_empty_list(self, repo: LocalSafetensorsRepository) -> None:
        assert repo.list_adapters() == []


class TestDelete:
    def test_delete_removes_adapter(
        self,
        repo: LocalSafetensorsRepository,
        weights: dict[str, torch.Tensor],
        peft_config_mock: MagicMock,
    ) -> None:
        repo.save("test_adapter", weights, peft_config_mock)
        assert repo.exists("test_adapter") is True

        repo.delete("test_adapter")
        assert repo.exists("test_adapter") is False

    def test_delete_nonexistent_does_not_raise(self, repo: LocalSafetensorsRepository) -> None:
        repo.delete("nonexistent")  # Should not raise
