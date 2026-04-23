"""Compilation pipeline — orchestrates data ingestion → LoRA adapter generation.

Binds an ``IDataSource`` (or legacy ``IKnowledgeStream``), an
``IWeightGenerator``, and an ``IAdapterRepository`` into a single
compile workflow.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_patch.core.interfaces import (
    IAdapterRepository,
    IDataSource,
    IKnowledgeStream,
    IWeightGenerator,
)
from llm_patch.core.models import AdapterManifest, DocumentContext

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CompilePipeline:
    """Compiles documents from a data source into LoRA adapter weights.

    Supports both pull-based (``IDataSource``) and push-based
    (``IKnowledgeStream``) ingestion.

    Args:
        source: Pull-based document source for batch compilation.
        generator: Weight generation strategy.
        repository: Adapter storage backend.
        stream: Optional push-based stream for live compilation.
    """

    def __init__(
        self,
        source: IDataSource,
        generator: IWeightGenerator,
        repository: IAdapterRepository,
        *,
        stream: IKnowledgeStream | None = None,
    ) -> None:
        self._source = source
        self._generator = generator
        self._repository = repository
        self._stream = stream

        if self._stream is not None:
            self._stream.subscribe(self._on_document_changed)

    def _on_document_changed(self, context: DocumentContext) -> None:
        """Callback for live-stream document changes."""
        logger.info("Live compile: %s", context.document_id)
        self.process_document(context)

    def process_document(self, context: DocumentContext) -> AdapterManifest:
        """Generate weights from a single document and store the adapter."""
        weights = self._generator.generate(context)
        peft_config = self._generator.get_peft_config()
        manifest = self._repository.save(context.document_id, weights, peft_config)

        logger.info("Adapter %s saved to %s", manifest.adapter_id, manifest.storage_uri)
        return manifest

    def compile_all(self) -> list[AdapterManifest]:
        """Batch compile all documents from the data source."""
        documents = list(self._source.fetch_all())
        logger.info("Compiling %d documents", len(documents))

        manifests: list[AdapterManifest] = []
        for doc in documents:
            manifest = self.process_document(doc)
            manifests.append(manifest)

        return manifests

    def start(self) -> None:
        """Start the live stream (if configured)."""
        if self._stream is not None:
            self._stream.start()

    def stop(self) -> None:
        """Stop the live stream (if configured)."""
        if self._stream is not None:
            self._stream.stop()

    def __enter__(self) -> CompilePipeline:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
