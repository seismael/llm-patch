"""Abstract interfaces defining the core contracts (Dependency Inversion Principle)."""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    import torch

    from llm_patch.core.models import AdapterManifest, DocumentContext


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


class IKnowledgeSource(abc.ABC):
    """Observer interface for monitoring and ingesting document changes."""

    @abc.abstractmethod
    def register_callback(self, callback: Callable[[DocumentContext], None]) -> None:
        """Register a callback to be invoked when a document changes.

        Args:
            callback: Function accepting a DocumentContext on document change.
        """

    @abc.abstractmethod
    def start(self) -> None:
        """Begin monitoring for document changes."""

    @abc.abstractmethod
    def stop(self) -> None:
        """Stop monitoring for document changes."""

    @abc.abstractmethod
    def scan_existing(self) -> list[DocumentContext]:
        """Perform a one-time scan of all current documents.

        Returns:
            A list of DocumentContext objects for all existing documents.
        """
