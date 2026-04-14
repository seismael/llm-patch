"""Shared test fixtures for llm_patch test suite."""

from __future__ import annotations

import platform
import sys

# Workaround: platform.uname() hangs on some Windows systems.
# Pre-cache to prevent hang when torch or other C extensions call it.
if sys.platform == "win32" and not hasattr(platform, "_uname_cache"):
    platform._uname_cache = platform.uname_result(
        system="Windows", node="", release="10", version="", machine="AMD64"
    )

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_patch.core.config import StorageConfig, WatcherConfig
from llm_patch.core.models import AdapterManifest, DocumentContext

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


@pytest.fixture()
def sample_document() -> DocumentContext:
    """A minimal DocumentContext for testing."""
    return DocumentContext(
        document_id="test_doc",
        content="This is sample document content for testing.",
        metadata={"source_path": "/tmp/test_doc.md"},
    )


@pytest.fixture()
def sample_weights() -> dict[str, Any]:
    """A minimal LoRA weight dict for testing."""
    pytest.importorskip("torch")
    return {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 1024),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(1024, 8),
    }


@pytest.fixture()
def sample_peft_config() -> dict[str, Any]:
    """A mock PEFT config dict with to_dict() method."""
    config = MagicMock()
    config.to_dict.return_value = {
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
    }
    return config


@pytest.fixture()
def sample_peft_config_dict() -> dict[str, Any]:
    """A plain dict representing PEFT config."""
    return {
        "r": 8,
        "target_modules": ["q_proj", "v_proj"],
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "bias": "none",
        "peft_type": "LORA",
    }


@pytest.fixture()
def markdown_dir(tmp_path: Path) -> Path:
    """A temporary directory seeded with sample markdown files."""
    docs = tmp_path / "docs"
    docs.mkdir()

    (docs / "api_v1.md").write_text("# API v1\nAuthentication via API key.", encoding="utf-8")
    (docs / "api_v2.md").write_text("# API v2\nOAuth2 bearer tokens.", encoding="utf-8")
    (docs / "guide.md").write_text("# User Guide\nGetting started.", encoding="utf-8")
    (docs / "notes.txt").write_text("This is not markdown.", encoding="utf-8")

    return docs


@pytest.fixture()
def storage_config(tmp_path: Path) -> StorageConfig:
    """A StorageConfig pointing to a temporary output directory."""
    output = tmp_path / "adapters"
    return StorageConfig(output_dir=output)


@pytest.fixture()
def watcher_config(markdown_dir: Path) -> WatcherConfig:
    """A WatcherConfig pointing to the sample markdown directory."""
    return WatcherConfig(directory=markdown_dir)


@pytest.fixture()
def mock_generator() -> MagicMock:
    """A mock IWeightGenerator."""
    gen = MagicMock()
    if HAS_TORCH:
        gen.generate.return_value = {
            "lora_A": torch.zeros(8, 256),
            "lora_B": torch.zeros(256, 8),
        }
    else:
        gen.generate.return_value = {
            "lora_A": MagicMock(),
            "lora_B": MagicMock(),
        }
    gen.get_peft_config.return_value = MagicMock(
        to_dict=MagicMock(
            return_value={
                "r": 8,
                "target_modules": ["q_proj", "v_proj"],
                "peft_type": "LORA",
            }
        )
    )
    return gen


@pytest.fixture()
def mock_repository() -> MagicMock:
    """A mock IAdapterRepository."""
    repo = MagicMock()
    repo.save.return_value = AdapterManifest(
        adapter_id="test_doc",
        rank=8,
        target_modules=["q_proj", "v_proj"],
        storage_uri="/tmp/adapters/test_doc",
    )
    return repo


@pytest.fixture()
def mock_source() -> MagicMock:
    """A mock IKnowledgeSource."""
    source = MagicMock()
    source.scan_existing.return_value = [
        DocumentContext(document_id="doc1", content="Content 1"),
        DocumentContext(document_id="doc2", content="Content 2"),
        DocumentContext(document_id="doc3", content="Content 3"),
    ]
    return source
