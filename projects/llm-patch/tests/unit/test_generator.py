"""Unit tests for SakanaT2LGenerator (Strategy Pattern).

These tests mock the hyper_llm_modulator internals via sys.modules patching
to verify the generator's integration logic without requiring GPU or model
checkpoints or the actual Sakana library.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from llm_patch_shared import ConfigurationError

try:
    import torch
except (ImportError, OSError):
    pytest.skip("torch not available", allow_module_level=True)

from llm_patch.core.config import GeneratorConfig
from llm_patch.core.models import DocumentContext


class TestSakanaT2LGeneratorInit:
    def test_init_validates_missing_checkpoint_dir(self, tmp_path: Path) -> None:
        from llm_patch.generators.sakana_t2l import SakanaT2LGenerator

        config = GeneratorConfig(checkpoint_dir=tmp_path / "nonexistent", device="cpu")
        with pytest.raises(ConfigurationError, match=r"hypermod\.pt"):
            SakanaT2LGenerator(config)

    def test_init_validates_missing_args_yaml(self, tmp_path: Path) -> None:
        from llm_patch.generators.sakana_t2l import SakanaT2LGenerator

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "hypermod.pt").write_bytes(b"fake")

        config = GeneratorConfig(checkpoint_dir=checkpoint_dir, device="cpu")
        with pytest.raises(ConfigurationError, match=r"args\.yaml"):
            SakanaT2LGenerator(config)

    def test_init_validates_missing_adapter_config(self, tmp_path: Path) -> None:
        from llm_patch.generators.sakana_t2l import SakanaT2LGenerator

        checkpoint_dir = tmp_path / "checkpoint"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "hypermod.pt").write_bytes(b"fake")
        (checkpoint_dir / "args.yaml").write_text("model_dir: test")

        config = GeneratorConfig(checkpoint_dir=checkpoint_dir, device="cpu")
        with pytest.raises(ConfigurationError, match=r"adapter_config\.json"):
            SakanaT2LGenerator(config)


def _make_sakana_mocks() -> dict[str, MagicMock]:
    """Create mock objects mimicking Sakana's API surface and register them in sys.modules."""
    mock_hypermod_obj = MagicMock()
    mock_hypermod_obj.peft_config = MagicMock(r=8)
    mock_hypermod_obj.task_encoder.return_value = {
        "encoded_task_emb": torch.randn(1, 128),
    }
    mock_hypermod_obj.gen_lora.return_value = {
        "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 1024),
        "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.randn(1024, 8),
    }

    mock_layers = [MagicMock() for _ in range(4)]

    # Build module-level mocks
    mock_hyper_modulator_mod = MagicMock()
    mock_hyper_modulator_mod.load_hypermod_checkpoint.return_value = (
        MagicMock(),  # args
        mock_hypermod_obj,
        MagicMock(),  # model
        MagicMock(),  # tokenizer
        MagicMock(),  # emb_model
        MagicMock(),  # emb_tokenizer
        MagicMock(),  # task_desc_format_fn
        MagicMock(),  # pooling_fn
    )

    mock_utils_mod = MagicMock()
    mock_utils_mod.get_layers.return_value = mock_layers
    mock_utils_mod.embed_texts.return_value = torch.randn(1, 1024)

    return {
        "hypermod": mock_hypermod_obj,
        "layers": mock_layers,
        "hyper_modulator_mod": mock_hyper_modulator_mod,
        "utils_mod": mock_utils_mod,
    }


class TestSakanaT2LGeneratorWithMocks:
    """Tests that mock the Sakana library via sys.modules to verify integration logic."""

    @pytest.fixture()
    def checkpoint_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "checkpoint"
        d.mkdir()
        (d / "hypermod.pt").write_bytes(b"fake")
        (d / "args.yaml").write_text("model_dir: test")
        (d / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
        return d

    @pytest.fixture()
    def sakana_mocks(self) -> dict[str, MagicMock]:
        return _make_sakana_mocks()

    def _build_generator(
        self,
        checkpoint_dir: Path,
        sakana_mocks: dict[str, MagicMock],
    ) -> SakanaT2LGenerator:  # noqa: F821
        """Construct a SakanaT2LGenerator with mocked Sakana sys.modules."""
        fake_modules = {
            "hyper_llm_modulator": MagicMock(),
            "hyper_llm_modulator.hyper_modulator": sakana_mocks["hyper_modulator_mod"],
            "hyper_llm_modulator.utils": sakana_mocks["utils_mod"],
        }
        with patch.dict(sys.modules, fake_modules):
            # Force fresh import so deferred imports inside __init__ hit our mocks
            mod_key = "llm_patch.generators.sakana_t2l"
            sys.modules.pop(mod_key, None)
            from llm_patch.generators.sakana_t2l import SakanaT2LGenerator

            config = GeneratorConfig(checkpoint_dir=checkpoint_dir, device="cpu")
            generator = SakanaT2LGenerator(config)

        return generator

    def test_generate_returns_dict_of_tensors(
        self, checkpoint_dir: Path, sakana_mocks: dict[str, MagicMock]
    ) -> None:
        generator = self._build_generator(checkpoint_dir, sakana_mocks)

        fake_modules = {
            "hyper_llm_modulator": MagicMock(),
            "hyper_llm_modulator.hyper_modulator": sakana_mocks["hyper_modulator_mod"],
            "hyper_llm_modulator.utils": sakana_mocks["utils_mod"],
        }
        with patch.dict(sys.modules, fake_modules):
            doc = DocumentContext(document_id="test", content="Hello world")
            result = generator.generate(doc)

        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, torch.Tensor)

    def test_get_peft_config_returns_config(
        self, checkpoint_dir: Path, sakana_mocks: dict[str, MagicMock]
    ) -> None:
        generator = self._build_generator(checkpoint_dir, sakana_mocks)
        config = generator.get_peft_config()
        assert config is sakana_mocks["hypermod"].peft_config

    def test_generate_calls_task_encoder(
        self, checkpoint_dir: Path, sakana_mocks: dict[str, MagicMock]
    ) -> None:
        generator = self._build_generator(checkpoint_dir, sakana_mocks)

        fake_modules = {
            "hyper_llm_modulator": MagicMock(),
            "hyper_llm_modulator.hyper_modulator": sakana_mocks["hyper_modulator_mod"],
            "hyper_llm_modulator.utils": sakana_mocks["utils_mod"],
        }
        with patch.dict(sys.modules, fake_modules):
            doc = DocumentContext(document_id="test", content="Test content")
            generator.generate(doc)

        sakana_mocks["hypermod"].task_encoder.assert_called_once()

    def test_generate_calls_gen_lora(
        self, checkpoint_dir: Path, sakana_mocks: dict[str, MagicMock]
    ) -> None:
        generator = self._build_generator(checkpoint_dir, sakana_mocks)

        fake_modules = {
            "hyper_llm_modulator": MagicMock(),
            "hyper_llm_modulator.hyper_modulator": sakana_mocks["hyper_modulator_mod"],
            "hyper_llm_modulator.utils": sakana_mocks["utils_mod"],
        }
        with patch.dict(sys.modules, fake_modules):
            doc = DocumentContext(document_id="test", content="Test content")
            generator.generate(doc)

        sakana_mocks["hypermod"].gen_lora.assert_called_once()
