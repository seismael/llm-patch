"""Knowledge Fusion Orchestrator — the central Facade coordinating the pipeline."""

from __future__ import annotations

import logging

from llm_patch.core.interfaces import IAdapterRepository, IKnowledgeSource, IWeightGenerator
from llm_patch.core.models import AdapterManifest, DocumentContext

logger = logging.getLogger(__name__)


class KnowledgeFusionOrchestrator:
    """Central manager coordinating the document-to-LoRA pipeline.

    Binds a knowledge source (Observer), weight generator (Strategy),
    and adapter repository (Repository) into a unified workflow.

    On construction, registers itself as an observer of the source so that
    document changes automatically trigger weight generation and storage.

    Args:
        source: The document source to observe for changes.
        generator: The weight generation strategy to use.
        repository: The storage backend for persisting adapters.
    """

    def __init__(
        self,
        source: IKnowledgeSource,
        generator: IWeightGenerator,
        repository: IAdapterRepository,
    ) -> None:
        self._source = source
        self._generator = generator
        self._repository = repository

        # Register as observer (callback pattern)
        self._source.register_callback(self._on_document_changed)

    def _on_document_changed(self, context: DocumentContext) -> None:
        """Internal callback invoked when the source detects a document change."""
        logger.info("Processing document: %s", context.document_id)
        self.process_document(context)

    def process_document(self, context: DocumentContext) -> AdapterManifest:
        """Generate weights from a document and store the resulting adapter.

        Args:
            context: The document to process.

        Returns:
            The manifest of the stored adapter.
        """
        weights = self._generator.generate(context)
        peft_config = self._generator.get_peft_config()
        manifest = self._repository.save(context.document_id, weights, peft_config)

        logger.info(
            "Adapter %s saved to %s",
            manifest.adapter_id,
            manifest.storage_uri,
        )
        return manifest

    def compile_all(self) -> list[AdapterManifest]:
        """Scan all existing documents and compile adapters for each.

        Returns:
            A list of manifests for all compiled adapters.
        """
        documents = self._source.scan_existing()
        logger.info("Compiling %d existing documents", len(documents))

        manifests: list[AdapterManifest] = []
        for doc in documents:
            manifest = self.process_document(doc)
            manifests.append(manifest)

        return manifests

    def start(self) -> None:
        """Start the knowledge source watcher."""
        self._source.start()

    def stop(self) -> None:
        """Stop the knowledge source watcher."""
        self._source.stop()

    def __enter__(self) -> KnowledgeFusionOrchestrator:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
