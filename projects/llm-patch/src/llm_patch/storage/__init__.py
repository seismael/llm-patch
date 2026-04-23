"""Adapter storage backends (Repository Pattern)."""

import contextlib

with contextlib.suppress(ImportError, OSError):
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

__all__ = ["LocalSafetensorsRepository"]
