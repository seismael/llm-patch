"""Adapter storage backends (Repository Pattern)."""

import contextlib

from llm_patch.storage.lru_cache import LRUAdapterCache

with contextlib.suppress(ImportError, OSError):
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

__all__ = ["LRUAdapterCache", "LocalSafetensorsRepository"]
