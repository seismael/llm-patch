"""Integration tests for the full document-to-adapter pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch not available", allow_module_level=True)

from llm_patch.core.config import StorageConfig, WatcherConfig
from llm_patch.core.models import AdapterManifest
from llm_patch.orchestrator import KnowledgeFusionOrchestrator
from llm_patch.sources.markdown_watcher import MarkdownDirectoryWatcher
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository


@pytest.fixture()
def pipeline_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create source docs dir with markdown files and output adapters dir."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "api_v1.md").write_text("# API v1\nKey-based auth.", encoding="utf-8")
    (docs_dir / "api_v2.md").write_text("# API v2\nOAuth2 tokens.", encoding="utf-8")
    (docs_dir / "guide.md").write_text("# Guide\nGetting started.", encoding="utf-8")

    adapters_dir = tmp_path / "adapters"
    return docs_dir, adapters_dir


@pytest.fixture()
def mock_weight_generator() -> MagicMock:
    """A mock generator that returns deterministic fake weights."""
    gen = MagicMock()
    gen.generate.return_value = {
        "lora_A": torch.randn(8, 256),
        "lora_B": torch.randn(256, 8),
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


@pytest.mark.integration
class TestCompileAllPipeline:
    """Tests compile_all with real watcher + real storage + mock generator."""

    def test_compile_all_creates_adapter_directories(
        self, pipeline_dirs: tuple[Path, Path], mock_weight_generator: MagicMock
    ) -> None:
        docs_dir, adapters_dir = pipeline_dirs

        watcher = MarkdownDirectoryWatcher(WatcherConfig(directory=docs_dir))
        repository = LocalSafetensorsRepository(StorageConfig(output_dir=adapters_dir))

        orchestrator = KnowledgeFusionOrchestrator(
            source=watcher,
            generator=mock_weight_generator,
            repository=repository,
        )

        manifests = orchestrator.compile_all()

        assert len(manifests) == 3
        assert all(isinstance(m, AdapterManifest) for m in manifests)

        # Verify adapter directories and files were created
        for manifest in manifests:
            adapter_dir = Path(manifest.storage_uri)
            assert adapter_dir.exists()
            assert (adapter_dir / "adapter_model.safetensors").exists()
            assert (adapter_dir / "adapter_config.json").exists()
            assert (adapter_dir / "manifest.json").exists()

    def test_compile_all_adapter_ids_match_filenames(
        self, pipeline_dirs: tuple[Path, Path], mock_weight_generator: MagicMock
    ) -> None:
        docs_dir, adapters_dir = pipeline_dirs

        watcher = MarkdownDirectoryWatcher(WatcherConfig(directory=docs_dir))
        repository = LocalSafetensorsRepository(StorageConfig(output_dir=adapters_dir))

        orchestrator = KnowledgeFusionOrchestrator(
            source=watcher,
            generator=mock_weight_generator,
            repository=repository,
        )

        manifests = orchestrator.compile_all()
        ids = {m.adapter_id for m in manifests}
        assert ids == {"api_v1", "api_v2", "guide"}

    def test_saved_adapters_are_loadable(
        self, pipeline_dirs: tuple[Path, Path], mock_weight_generator: MagicMock
    ) -> None:
        docs_dir, adapters_dir = pipeline_dirs

        watcher = MarkdownDirectoryWatcher(WatcherConfig(directory=docs_dir))
        repository = LocalSafetensorsRepository(StorageConfig(output_dir=adapters_dir))

        orchestrator = KnowledgeFusionOrchestrator(
            source=watcher,
            generator=mock_weight_generator,
            repository=repository,
        )

        orchestrator.compile_all()

        # Verify we can load each adapter
        for adapter_id in ["api_v1", "api_v2", "guide"]:
            weights = repository.load(adapter_id)
            assert isinstance(weights, dict)
            assert len(weights) > 0

    def test_repository_list_shows_all_adapters(
        self, pipeline_dirs: tuple[Path, Path], mock_weight_generator: MagicMock
    ) -> None:
        docs_dir, adapters_dir = pipeline_dirs

        watcher = MarkdownDirectoryWatcher(WatcherConfig(directory=docs_dir))
        repository = LocalSafetensorsRepository(StorageConfig(output_dir=adapters_dir))

        orchestrator = KnowledgeFusionOrchestrator(
            source=watcher,
            generator=mock_weight_generator,
            repository=repository,
        )

        orchestrator.compile_all()

        listed = repository.list_adapters()
        assert len(listed) == 3
