"""Abstract interfaces defining the core contracts (Dependency Inversion Principle).

Layer 0 — every concrete implementation depends on these; nothing here
depends on concrete code.

Interfaces
----------
- ``IDataSource``          — pull-based document ingestion (batch).
- ``IKnowledgeStream``     — push-based real-time document change notifications.
- ``IWeightGenerator``     — text → LoRA weight tensor generation.
- ``IAdapterRepository``   — adapter storage (save / load / list / delete).
- ``IModelProvider``       — load a base model into memory.
- ``IAdapterLoader``       — attach adapter weights onto a loaded model.
- ``IAgentRuntime``        — chat / generate over a patched model.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable

    import torch

    from llm_patch.core.models import (
        AdapterManifest,
        AdapterRef,
        ChatMessage,
        ChatResponse,
        DocumentContext,
        ModelHandle,
    )


# ── Data Ingestion ────────────────────────────────────────────────────


class IDataSource(abc.ABC):
    """Pull-based data source that yields normalized documents.

    All concrete sources (Markdown directory, Wiki, PDF, JSONL, HTTP API,
    etc.) implement this interface.  The pipeline calls ``fetch_all`` to
    retrieve documents in batch.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Human-readable source name (e.g. ``'markdown'``, ``'wiki'``)."""

    @abc.abstractmethod
    def fetch_all(self) -> Iterable[DocumentContext]:
        """Yield every document currently available from the source."""

    def fetch_one(self, document_id: str) -> DocumentContext | None:
        """Fetch a single document by ID, or ``None`` if not found.

        Default implementation iterates ``fetch_all``; subclasses may
        override with an efficient lookup.
        """
        for doc in self.fetch_all():
            if doc.document_id == document_id:
                return doc
        return None


class IKnowledgeStream(abc.ABC):
    """Push-based real-time document change stream (optional mixin).

    Sources that support live monitoring (e.g. filesystem watchers)
    implement this alongside ``IDataSource``.
    """

    @abc.abstractmethod
    def subscribe(self, callback: Callable[[DocumentContext], None]) -> None:
        """Register a callback invoked on each document change."""

    @abc.abstractmethod
    def start(self) -> None:
        """Begin monitoring for changes."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop monitoring."""


# ── Weight Generation ─────────────────────────────────────────────────


class IWeightGenerator(abc.ABC):
    """Strategy interface for converting document text into LoRA weight matrices."""

    @abc.abstractmethod
    def generate(self, context: DocumentContext) -> dict[str, torch.Tensor]:
        """Convert a document context into LoRA weight tensors.

        Args:
            context: The document to internalize.

        Returns:
            A state dict mapping PEFT-format keys to weight tensors.
        """

    @abc.abstractmethod
    def get_peft_config(self) -> Any:
        """Return the PEFT LoraConfig associated with this generator.

        Returns:
            A peft.LoraConfig instance describing the adapter structure.
        """


# ── Adapter Storage ───────────────────────────────────────────────────


class IAdapterRepository(abc.ABC):
    """Repository interface for persisting and retrieving LoRA adapter weights."""

    @abc.abstractmethod
    def save(
        self,
        adapter_id: str,
        weights: dict[str, torch.Tensor],
        peft_config: Any,
    ) -> AdapterManifest:
        """Serialize and store adapter weights.

        Args:
            adapter_id: Unique identifier for the adapter.
            weights: State dict of LoRA weight tensors.
            peft_config: PEFT configuration to persist alongside weights.

        Returns:
            A manifest describing the stored adapter.
        """

    @abc.abstractmethod
    def load(self, adapter_id: str) -> dict[str, torch.Tensor]:
        """Load adapter weights from storage.

        Args:
            adapter_id: Unique identifier of the adapter to load.

        Returns:
            State dict of LoRA weight tensors.
        """

    @abc.abstractmethod
    def exists(self, adapter_id: str) -> bool:
        """Check whether an adapter exists in storage."""

    @abc.abstractmethod
    def list_adapters(self) -> list[AdapterManifest]:
        """List all stored adapters.

        Returns:
            A list of adapter manifests.
        """

    @abc.abstractmethod
    def delete(self, adapter_id: str) -> None:
        """Remove an adapter from storage.

        Args:
            adapter_id: Unique identifier of the adapter to delete.
        """


# ── Model Loading ─────────────────────────────────────────────────────


class IModelProvider(abc.ABC):
    """Loads a base model + tokenizer into memory."""

    @abc.abstractmethod
    def load(self, model_id: str, **kwargs: Any) -> ModelHandle:
        """Load a base model and return a handle.

        Args:
            model_id: HuggingFace model ID or local path.
            **kwargs: Forwarded to the loader (dtype, device_map, …).

        Returns:
            A ``ModelHandle`` wrapping (model, tokenizer, metadata).
        """


# ── Adapter Attachment ────────────────────────────────────────────────


class IAdapterLoader(abc.ABC):
    """Attaches LoRA adapter weights onto a loaded base model."""

    @abc.abstractmethod
    def attach(
        self,
        handle: ModelHandle,
        manifest: AdapterManifest,
    ) -> ModelHandle:
        """Load adapter weights onto *handle* and return an updated handle.

        Args:
            handle: The base (or already-patched) model handle.
            manifest: Manifest of the adapter to attach.

        Returns:
            A new ``ModelHandle`` with the adapter active.
        """


# ── Agent Runtime ─────────────────────────────────────────────────────


class IAgentRuntime(abc.ABC):
    """Chat / generate interface over a patched model."""

    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a text completion.

        Args:
            prompt: The input prompt.

        Returns:
            The generated text.
        """

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Multi-turn chat completion.

        Args:
            messages: Conversation history.

        Returns:
            A ``ChatResponse`` with the assistant reply and metadata.
        """

    def stream(self, prompt: str, **kwargs: Any) -> Generator[str, None, None]:
        """Streaming token generator (optional).

        Default falls back to non-streaming ``generate``.
        """
        yield self.generate(prompt, **kwargs)


# ── Distributed Adapter Registry (Adapter Market) ─────────────────────


class IAdapterRegistryClient(abc.ABC):
    """Client for a remote adapter registry / "Adapter Market" hub.

    Conceptually a Repository over the network: it resolves
    :class:`~llm_patch.core.models.AdapterRef` URIs to manifests,
    downloads weights into a local :class:`IAdapterRepository`, and
    publishes locally-stored adapters back to a remote hub.

    Implementations are NOT shipped by the engine. Concrete HTTP / S3 /
    OCI clients live behind optional extras (see ``docs/REGISTRY_PROTOCOL.md``).
    """

    @abc.abstractmethod
    def search(self, query: str, *, limit: int = 10) -> list[AdapterManifest]:
        """Return manifests matching *query*, ordered by relevance.

        Args:
            query: Free-form text — framework names, tags, descriptions.
            limit: Maximum number of results.
        """

    @abc.abstractmethod
    def resolve(self, ref: AdapterRef) -> AdapterManifest:
        """Resolve a reference to its manifest without downloading weights."""

    @abc.abstractmethod
    def pull(self, ref: AdapterRef) -> AdapterManifest:
        """Download adapter weights into the local repository.

        Implementations MUST verify ``manifest.checksum_sha256`` and
        raise :class:`llm_patch_utils.ChecksumMismatchError` on
        mismatch. Returns the verified, locally-stored manifest.
        """

    @abc.abstractmethod
    def push(self, adapter_id: str, ref: AdapterRef) -> AdapterManifest:
        """Upload a locally-stored adapter to the registry under *ref*."""


# ── Adapter Cache (LRU Decorator) ─────────────────────────────────────


class IAdapterCache(abc.ABC):
    """Bounded cache of :class:`AdapterManifest` entries.

    Decorator over :class:`IAdapterRepository`: holds at most
    :attr:`capacity` recently-used manifests in memory. Eviction is
    LRU. The cache stores manifests only — adapter weights remain
    materialized through the underlying repository.
    """

    @property
    @abc.abstractmethod
    def capacity(self) -> int:
        """Maximum number of manifests retained."""

    @abc.abstractmethod
    def get(self, adapter_id: str) -> AdapterManifest | None:
        """Return the manifest for *adapter_id*, or ``None`` if absent."""

    @abc.abstractmethod
    def put(self, manifest: AdapterManifest) -> None:
        """Insert *manifest*, evicting the LRU entry if at capacity."""

    @abc.abstractmethod
    def evict(self, adapter_id: str) -> None:
        """Remove *adapter_id* from the cache (no-op if absent)."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Current number of cached manifests."""


# ── Runtime Adapter Controller (hot-swap) ─────────────────────────────


class IRuntimeAdapterController(abc.ABC):
    """Mutates a live :class:`ModelHandle` by attaching/detaching adapters.

    Sits between :class:`IAgentRuntime` and :class:`IAdapterLoader`.
    Implementations MUST serialize concurrent attach/detach calls
    (typically via an ``RLock`` or ``asyncio.Lock``) because PEFT
    state is not thread-safe.
    """

    @abc.abstractmethod
    def attach(self, ref: AdapterRef) -> AdapterManifest:
        """Attach the adapter named by *ref* onto the active handle."""

    @abc.abstractmethod
    def detach(self, adapter_id: str) -> None:
        """Detach the adapter with *adapter_id* (no-op if not active)."""

    @abc.abstractmethod
    def active(self) -> list[str]:
        """List ``adapter_id`` values currently attached, in attach-order."""
