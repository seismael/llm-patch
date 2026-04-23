"""Tests for public engine exports and legacy compatibility shims."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from llm_patch import (
    AdapterManifest,
    ChatSession,
    HFModelProvider,
    IAdapterLoader,
    IAdapterRepository,
    IAgentRuntime,
    IModelProvider,
    IWeightGenerator,
    KnowledgeFusionOrchestrator,
    PeftAdapterLoader,
    PeftAgentRuntime,
)

if TYPE_CHECKING:
    import torch


class _LegacySource:
    def register_callback(self, callback: object) -> None:
        self.callback = callback

    def scan_existing(self) -> list[object]:
        return []

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


class _FakeGenerator(IWeightGenerator):
    def generate(self, _context: object) -> dict[str, torch.Tensor]:
        return cast("dict[str, torch.Tensor]", {})

    def get_peft_config(self) -> object:
        return {}


class _FakeRepository(IAdapterRepository):
    def save(
        self,
        adapter_id: str,
        _weights: dict[str, torch.Tensor],
        _peft_config: object,
    ) -> AdapterManifest:
        return AdapterManifest(
            adapter_id=adapter_id,
            rank=8,
            target_modules=["q_proj"],
            storage_uri=f"adapters/{adapter_id}",
        )

    def load(self, _adapter_id: str) -> dict[str, torch.Tensor]:
        return cast("dict[str, torch.Tensor]", {})

    def exists(self, _adapter_id: str) -> bool:
        return False

    def list_adapters(self) -> list[AdapterManifest]:
        return []

    def delete(self, _adapter_id: str) -> None:
        return None


def test_runtime_and_attach_exports_are_available_from_top_level() -> None:
    assert issubclass(HFModelProvider, IModelProvider)
    assert issubclass(PeftAdapterLoader, IAdapterLoader)
    assert issubclass(PeftAgentRuntime, IAgentRuntime)
    assert ChatSession.__name__ == "ChatSession"


def test_knowledge_fusion_orchestrator_warns_on_instantiation() -> None:
    source = _LegacySource()

    with pytest.warns(DeprecationWarning):
        KnowledgeFusionOrchestrator(
            source=source,
            generator=_FakeGenerator(),
            repository=_FakeRepository(),
        )
