"""Tests for :class:`PeftRuntimeController`.

These tests exercise the controller against fakes — no torch / PEFT.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_patch import (
    AdapterManifest,
    AdapterRef,
    PeftRuntimeController,
)
from llm_patch.core.interfaces import (
    IAdapterLoader,
    IAdapterRegistryClient,
    IAdapterRepository,
)
from llm_patch_utils import (
    AdapterNotFoundError,
    IncompatibleBaseModelError,
    RegistryUnavailableError,
)


# ── Fakes ────────────────────────────────────────────────────────────


class FakeHandle:
    """Minimal stand-in for :class:`ModelHandle`."""

    def __init__(self, base_model_id: str = "google/gemma-2-2b-it") -> None:
        self.base_model_id = base_model_id
        self.attached_adapters: tuple[str, ...] = ()


class FakeLoader(IAdapterLoader):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def attach(self, handle: Any, manifest: AdapterManifest) -> Any:
        self.calls.append(manifest.adapter_id)
        return handle


class FakeRepo(IAdapterRepository):
    def __init__(self, manifests: dict[str, AdapterManifest] | None = None) -> None:
        self._items = manifests or {}

    def save(self, adapter_id: str, weights: Any, peft_config: Any) -> AdapterManifest:
        m = AdapterManifest(
            adapter_id=adapter_id, rank=8, target_modules=["q"], storage_uri="/tmp"
        )
        self._items[adapter_id] = m
        return m

    def load(self, adapter_id: str) -> dict[str, Any]:
        return {}

    def exists(self, adapter_id: str) -> bool:
        return adapter_id in self._items

    def list_adapters(self) -> list[AdapterManifest]:
        return list(self._items.values())

    def delete(self, adapter_id: str) -> None:
        self._items.pop(adapter_id, None)


class FakeRegistry(IAdapterRegistryClient):
    def __init__(self, manifests: dict[str, AdapterManifest] | None = None) -> None:
        self._items = manifests or {}
        self.pulled: list[str] = []

    def search(self, query: str, *, limit: int = 10) -> list[AdapterManifest]:
        return list(self._items.values())[:limit]

    def resolve(self, ref: AdapterRef) -> AdapterManifest:
        if ref.adapter_id not in self._items:
            raise AdapterNotFoundError(ref.to_uri())
        return self._items[ref.adapter_id]

    def pull(self, ref: AdapterRef) -> AdapterManifest:
        if ref.adapter_id not in self._items:
            raise AdapterNotFoundError(ref.to_uri())
        self.pulled.append(ref.to_uri())
        return self._items[ref.adapter_id]

    def push(self, adapter_id: str, ref: AdapterRef) -> AdapterManifest:
        m = AdapterManifest(
            adapter_id=ref.adapter_id, rank=8, target_modules=["q"], storage_uri="/tmp"
        )
        self._items[ref.adapter_id] = m
        return m


# ── Tests ────────────────────────────────────────────────────────────


class TestPeftRuntimeController:
    def test_attach_uses_local_when_present(self) -> None:
        ref = AdapterRef.parse("hub://acme/x:v1")
        local = AdapterManifest(
            adapter_id=ref.adapter_id, rank=8, target_modules=["q"], storage_uri="/tmp"
        )
        repo = FakeRepo({local.adapter_id: local})
        loader = FakeLoader()
        ctrl = PeftRuntimeController(FakeHandle(), loader, repo)  # type: ignore[arg-type]
        ctrl.attach(ref)
        assert loader.calls == [local.adapter_id]
        assert ctrl.active() == [local.adapter_id]

    def test_attach_pulls_from_registry_when_missing(self) -> None:
        ref = AdapterRef.parse("hub://acme/x:v1")
        remote = AdapterManifest(
            adapter_id=ref.adapter_id,
            rank=8,
            target_modules=["q"],
            storage_uri="/tmp",
        )
        repo = FakeRepo()
        registry = FakeRegistry({ref.adapter_id: remote})
        loader = FakeLoader()
        ctrl = PeftRuntimeController(FakeHandle(), loader, repo, registry=registry)  # type: ignore[arg-type]
        ctrl.attach(ref)
        assert registry.pulled == [ref.to_uri()]

    def test_attach_without_registry_raises(self) -> None:
        ref = AdapterRef.parse("hub://acme/x:v1")
        ctrl = PeftRuntimeController(FakeHandle(), FakeLoader(), FakeRepo())  # type: ignore[arg-type]
        with pytest.raises(RegistryUnavailableError):
            ctrl.attach(ref)

    def test_incompatible_base_model_raises(self) -> None:
        ref = AdapterRef.parse("hub://acme/x:v1")
        local = AdapterManifest(
            adapter_id=ref.adapter_id,
            rank=8,
            target_modules=["q"],
            storage_uri="/tmp",
            base_model_compatibility=["other/model"],
        )
        repo = FakeRepo({local.adapter_id: local})
        ctrl = PeftRuntimeController(FakeHandle(), FakeLoader(), repo)  # type: ignore[arg-type]
        with pytest.raises(IncompatibleBaseModelError):
            ctrl.attach(ref)

    def test_detach_is_idempotent(self) -> None:
        ctrl = PeftRuntimeController(FakeHandle(), FakeLoader(), FakeRepo())  # type: ignore[arg-type]
        ctrl.detach("missing")
        assert ctrl.active() == []
