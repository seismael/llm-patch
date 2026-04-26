"""Default :class:`IRuntimeAdapterController` implementation.

Wraps an :class:`IAdapterLoader` + :class:`IAdapterRepository` and
optionally an :class:`IAdapterRegistryClient` so that runtime callers
(server endpoints, MCP tools) can hot-swap adapters by
:class:`AdapterRef`.

Concurrency: a single :class:`threading.RLock` serializes ``attach`` /
``detach`` calls, which is the minimum guarantee documented by the
:class:`IRuntimeAdapterController` contract. Higher throughput
(LoRAX-style batched multi-adapter inference) is **deferred** — see
``docs/SERVER_ARCHITECTURE.md``.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

from llm_patch_utils import (
    AdapterNotFoundError,
    IncompatibleBaseModelError,
    RegistryUnavailableError,
)

from llm_patch.core.interfaces import (
    IAdapterLoader,
    IAdapterRegistryClient,
    IAdapterRepository,
    IRuntimeAdapterController,
)

if TYPE_CHECKING:
    from llm_patch.core.models import AdapterManifest, AdapterRef, ModelHandle

logger = logging.getLogger(__name__)

__all__ = ["PeftRuntimeController"]


class PeftRuntimeController(IRuntimeAdapterController):
    """Hot-swap controller backed by an :class:`IAdapterLoader`.

    Args:
        handle: The active :class:`ModelHandle`. Mutated in place via
            successive ``loader.attach`` calls (PEFT semantics).
        loader: Strategy that knows how to attach LoRA weights.
        repository: Source-of-truth for locally-materialized adapters.
        registry: Optional remote registry client; required when
            attaching a ``hub://`` ref that is not already local.
    """

    def __init__(
        self,
        handle: ModelHandle,
        loader: IAdapterLoader,
        repository: IAdapterRepository,
        registry: IAdapterRegistryClient | None = None,
    ) -> None:
        self._handle = handle
        self._loader = loader
        self._repository = repository
        self._registry = registry
        self._active: list[str] = list(handle.attached_adapters)
        self._lock = threading.RLock()

    @property
    def handle(self) -> ModelHandle:
        """Current handle (mutates as adapters attach/detach)."""
        return self._handle

    def attach(self, ref: AdapterRef) -> AdapterManifest:
        with self._lock:
            manifest = self._resolve(ref)
            self._check_compatibility(manifest)
            self._handle = self._loader.attach(self._handle, manifest)
            if manifest.adapter_id not in self._active:
                self._active.append(manifest.adapter_id)
            logger.info("Attached adapter %s (active=%d)", manifest.adapter_id, len(self._active))
            return manifest

    def detach(self, adapter_id: str) -> None:
        with self._lock:
            if adapter_id not in self._active:
                return
            # PEFT does not always expose a clean unload-single-adapter API,
            # so we drop the id from our bookkeeping and let callers swap
            # the underlying handle when they need a hard reset.
            self._active.remove(adapter_id)
            logger.info("Detached adapter %s (active=%d)", adapter_id, len(self._active))

    def active(self) -> list[str]:
        with self._lock:
            return list(self._active)

    # ── Internals ─────────────────────────────────────────────────────

    def _resolve(self, ref: AdapterRef) -> AdapterManifest:
        local_id = ref.adapter_id
        if self._repository.exists(local_id):
            for m in self._repository.list_adapters():
                if m.adapter_id == local_id:
                    return m
        if self._registry is None:
            raise RegistryUnavailableError(
                f"Adapter {ref.to_uri()} not local and no registry client configured."
            )
        try:
            return self._registry.pull(ref)
        except AdapterNotFoundError:
            raise

    def _check_compatibility(self, manifest: AdapterManifest) -> None:
        compat = manifest.base_model_compatibility
        if not compat:
            return
        if self._handle.base_model_id not in compat:
            raise IncompatibleBaseModelError(
                f"Adapter {manifest.adapter_id} declares compatibility with "
                f"{compat!r} but active base model is {self._handle.base_model_id!r}."
            )
