"""Local filesystem storage using safetensors format (Repository Pattern)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from llm_patch.core.config import StorageConfig
from llm_patch.core.interfaces import IAdapterRepository
from llm_patch.core.models import AdapterManifest

logger = logging.getLogger(__name__)


class LocalSafetensorsRepository(IAdapterRepository):
    """Stores LoRA adapters as safetensors files on the local filesystem.

    Directory layout per adapter::

        {output_dir}/{adapter_id}/
        ├── adapter_model.safetensors
        ├── adapter_config.json
        └── manifest.json
    """

    def __init__(self, config: StorageConfig) -> None:
        self._output_dir = Path(config.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def _adapter_dir(self, adapter_id: str) -> Path:
        return self._output_dir / adapter_id

    def save(
        self,
        adapter_id: str,
        weights: dict[str, torch.Tensor],
        peft_config: Any,
    ) -> AdapterManifest:
        """Save adapter weights and config to a local directory."""
        adapter_dir = self._adapter_dir(adapter_id)
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save weights as safetensors
        weights_path = adapter_dir / "adapter_model.safetensors"
        save_file(weights, str(weights_path))
        logger.info("Saved weights to %s", weights_path)

        # 2. Save PEFT adapter config
        config_path = adapter_dir / "adapter_config.json"
        if hasattr(peft_config, "to_dict"):
            config_dict = peft_config.to_dict()
        elif isinstance(peft_config, dict):
            config_dict = peft_config
        else:
            config_dict = {"raw": str(peft_config)}
        config_path.write_text(json.dumps(config_dict, indent=2))
        logger.info("Saved adapter config to %s", config_path)

        # 3. Build and save manifest
        rank = config_dict.get("r", 0)
        target_modules = config_dict.get("target_modules", [])
        if isinstance(target_modules, set):
            target_modules = sorted(target_modules)

        manifest = AdapterManifest(
            adapter_id=adapter_id,
            rank=rank,
            target_modules=target_modules,
            storage_uri=str(adapter_dir),
        )

        manifest_path = adapter_dir / "manifest.json"
        manifest_path.write_text(manifest.model_dump_json(indent=2))

        logger.info("Adapter %s saved to %s", adapter_id, adapter_dir)
        return manifest

    def load(self, adapter_id: str) -> dict[str, torch.Tensor]:
        """Load adapter weights from a local safetensors file."""
        adapter_dir = self._adapter_dir(adapter_id)
        weights_path = adapter_dir / "adapter_model.safetensors"

        if not weights_path.exists():
            msg = f"Adapter '{adapter_id}' not found at {weights_path}"
            raise FileNotFoundError(msg)

        return dict(load_file(str(weights_path)))

    def exists(self, adapter_id: str) -> bool:
        """Check if an adapter's safetensors file exists."""
        weights_path = self._adapter_dir(adapter_id) / "adapter_model.safetensors"
        return weights_path.exists()

    def list_adapters(self) -> list[AdapterManifest]:
        """Scan output directory and load all adapter manifests."""
        manifests: list[AdapterManifest] = []
        if not self._output_dir.exists():
            return manifests

        for subdir in sorted(self._output_dir.iterdir()):
            manifest_path = subdir / "manifest.json"
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text())
                manifests.append(AdapterManifest.model_validate(data))

        return manifests

    def delete(self, adapter_id: str) -> None:
        """Remove an adapter directory entirely."""
        adapter_dir = self._adapter_dir(adapter_id)
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
            logger.info("Deleted adapter %s from %s", adapter_id, adapter_dir)
