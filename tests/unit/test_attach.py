"""Tests for llm_patch.attach — model_provider, peft_loader, merger."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_patch.core.models import AdapterManifest, ModelHandle


# ── HFModelProvider ────────────────────────────────────────────────────


class TestHFModelProvider:
    """Tests for HFModelProvider (torch + transformers mocked)."""

    def test_load_returns_model_handle(self):
        from llm_patch.attach.model_provider import HFModelProvider

        mock_torch = MagicMock()
        mock_torch.float16 = "fp16"

        mock_model = MagicMock()
        mock_model.device = "cuda:0"

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            provider = HFModelProvider()
            handle = provider.load("test-model")

        assert isinstance(handle, ModelHandle)
        assert handle.base_model_id == "test-model"
        assert handle.device == "cuda:0"
        assert handle.attached_adapters == ()

    def test_load_custom_kwargs(self):
        from llm_patch.attach.model_provider import HFModelProvider

        mock_torch = MagicMock()
        mock_torch.float32 = "fp32"

        mock_model = MagicMock()
        mock_model.device = "cpu"

        mock_transformers = MagicMock()
        mock_transformers.AutoModelForCausalLM.from_pretrained.return_value = mock_model
        mock_transformers.AutoTokenizer.from_pretrained.return_value = MagicMock()

        with patch.dict(sys.modules, {"torch": mock_torch, "transformers": mock_transformers}):
            provider = HFModelProvider()
            handle = provider.load(
                "test-model",
                dtype="float32",
                device_map="cpu",
                trust_remote_code=True,
            )

        assert handle.device == "cpu"
        call_kwargs = mock_transformers.AutoModelForCausalLM.from_pretrained.call_args
        assert call_kwargs.kwargs["device_map"] == "cpu"
        assert call_kwargs.kwargs["trust_remote_code"] is True


# ── PeftAdapterLoader ──────────────────────────────────────────────────


class TestPeftAdapterLoader:
    """Tests for PeftAdapterLoader (peft mocked)."""

    def _make_handle(self, attached=()):
        return ModelHandle(
            model=MagicMock(),
            tokenizer=MagicMock(),
            base_model_id="test-model",
            attached_adapters=attached,
            device="cpu",
        )

    def _make_manifest(self, adapter_id="adapt1"):
        return AdapterManifest(
            adapter_id=adapter_id,
            rank=8,
            target_modules=["q_proj"],
            storage_uri="/tmp/adapt1",
        )

    def test_attach_first_adapter(self):
        from llm_patch.attach.peft_loader import PeftAdapterLoader

        wrapped_model = MagicMock()
        mock_peft = MagicMock()
        mock_peft.PeftModel.from_pretrained.return_value = wrapped_model

        with patch.dict(sys.modules, {"peft": mock_peft}):
            loader = PeftAdapterLoader()
            handle = self._make_handle()
            manifest = self._make_manifest()
            new_handle = loader.attach(handle, manifest)

        assert new_handle.attached_adapters == ("adapt1",)
        mock_peft.PeftModel.from_pretrained.assert_called_once()

    def test_stack_second_adapter(self):
        from llm_patch.attach.peft_loader import PeftAdapterLoader

        mock_peft = MagicMock()

        with patch.dict(sys.modules, {"peft": mock_peft}):
            loader = PeftAdapterLoader()
            handle = self._make_handle(attached=("first",))
            manifest = self._make_manifest(adapter_id="second")
            new_handle = loader.attach(handle, manifest)

        assert new_handle.attached_adapters == ("first", "second")
        handle.model.load_adapter.assert_called_once()
        handle.model.set_adapter.assert_called_once_with("second")
        mock_peft.PeftModel.from_pretrained.assert_not_called()


# ── Merger ──────────────────────────────────────────────────────────────


class TestMergeIntoBase:
    """Tests for merge_into_base."""

    def test_merge_creates_dir_and_saves(self, tmp_path):
        from llm_patch.attach.merger import merge_into_base

        model = MagicMock()
        merged = MagicMock()
        model.merge_and_unload.return_value = merged
        tokenizer = MagicMock()

        handle = ModelHandle(
            model=model,
            tokenizer=tokenizer,
            base_model_id="test",
            attached_adapters=("a1",),
        )

        out_dir = tmp_path / "merged"
        result = merge_into_base(handle, out_dir)

        assert result == out_dir
        assert out_dir.exists()
        model.merge_and_unload.assert_called_once()
        merged.save_pretrained.assert_called_once_with(str(out_dir))
        tokenizer.save_pretrained.assert_called_once_with(str(out_dir))


class TestWeightedBlend:
    """Tests for weighted_blend."""

    def test_blend_with_weights(self):
        from llm_patch.attach.merger import weighted_blend

        model = MagicMock()
        tokenizer = MagicMock()
        handle = ModelHandle(
            model=model,
            tokenizer=tokenizer,
            base_model_id="test",
            attached_adapters=("a", "b"),
        )

        result = weighted_blend(handle, {"a": 1.0, "b": 0.5}, combined_name="merged")

        model.add_weighted_adapter.assert_called_once_with(
            ["a", "b"], [1.0, 0.5],
            combination_type="linear",
            adapter_name="merged",
        )
        model.set_adapter.assert_called_once_with("merged")
        assert "merged" in result.attached_adapters
