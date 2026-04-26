"""LRU implementation of :class:`IAdapterCache` (manifests only).

Pure stdlib — does not import torch. The cache holds
:class:`AdapterManifest` records keyed by ``adapter_id``; weight tensors
remain materialized on disk through whatever :class:`IAdapterRepository`
the surrounding pipeline uses.

This module is safe to import without any heavy ML dependency, which is
why it lives next to :mod:`llm_patch.storage` rather than in
:mod:`llm_patch.runtime`.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from llm_patch.core.interfaces import IAdapterCache
from llm_patch.core.models import AdapterManifest
from llm_patch_shared import CapacityExceededError

__all__ = ["LRUAdapterCache"]


class LRUAdapterCache(IAdapterCache):
    """Thread-safe LRU cache for :class:`AdapterManifest` entries."""

    def __init__(self, capacity: int = 16) -> None:
        if capacity <= 0:
            raise CapacityExceededError(
                f"capacity must be positive, got {capacity}"
            )
        self._capacity = capacity
        self._items: OrderedDict[str, AdapterManifest] = OrderedDict()
        self._lock = threading.RLock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def get(self, adapter_id: str) -> AdapterManifest | None:
        with self._lock:
            manifest = self._items.get(adapter_id)
            if manifest is not None:
                self._items.move_to_end(adapter_id)
            return manifest

    def put(self, manifest: AdapterManifest) -> None:
        with self._lock:
            self._items[manifest.adapter_id] = manifest
            self._items.move_to_end(manifest.adapter_id)
            while len(self._items) > self._capacity:
                self._items.popitem(last=False)

    def evict(self, adapter_id: str) -> None:
        with self._lock:
            self._items.pop(adapter_id, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __contains__(self, adapter_id: object) -> bool:
        if not isinstance(adapter_id, str):
            return False
        with self._lock:
            return adapter_id in self._items
