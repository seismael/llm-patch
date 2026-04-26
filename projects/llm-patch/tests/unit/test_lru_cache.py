"""Tests for :class:`LRUAdapterCache`."""

from __future__ import annotations

import pytest

from llm_patch import AdapterManifest, LRUAdapterCache
from llm_patch_utils import CapacityExceededError


def _manifest(adapter_id: str) -> AdapterManifest:
    return AdapterManifest(
        adapter_id=adapter_id, rank=8, target_modules=["q"], storage_uri=f"/tmp/{adapter_id}"
    )


class TestLRUAdapterCache:
    def test_get_returns_none_when_absent(self) -> None:
        cache = LRUAdapterCache(capacity=3)
        assert cache.get("missing") is None

    def test_put_then_get(self) -> None:
        cache = LRUAdapterCache(capacity=3)
        m = _manifest("a")
        cache.put(m)
        assert cache.get("a") is m
        assert len(cache) == 1

    def test_put_evicts_oldest_at_capacity(self) -> None:
        cache = LRUAdapterCache(capacity=2)
        cache.put(_manifest("a"))
        cache.put(_manifest("b"))
        cache.put(_manifest("c"))
        assert cache.get("a") is None
        assert cache.get("b") is not None
        assert cache.get("c") is not None
        assert len(cache) == 2

    def test_get_promotes_recency(self) -> None:
        cache = LRUAdapterCache(capacity=2)
        cache.put(_manifest("a"))
        cache.put(_manifest("b"))
        # Touch 'a' so 'b' becomes LRU.
        assert cache.get("a") is not None
        cache.put(_manifest("c"))
        assert cache.get("b") is None
        assert cache.get("a") is not None

    def test_evict_is_idempotent(self) -> None:
        cache = LRUAdapterCache(capacity=2)
        cache.put(_manifest("a"))
        cache.evict("a")
        cache.evict("a")
        assert "a" not in cache
        assert len(cache) == 0

    def test_zero_capacity_raises(self) -> None:
        with pytest.raises(CapacityExceededError):
            LRUAdapterCache(capacity=0)

    def test_capacity_property(self) -> None:
        assert LRUAdapterCache(capacity=5).capacity == 5
