"""Knowledge Fusion Orchestrator — backward-compat shim.

New code should use ``llm_patch.pipelines.CompilePipeline`` instead.
This module preserves the legacy API that works with objects exposing
``register_callback`` / ``scan_existing`` / ``start`` / ``stop``.
"""

from __future__ import annotations

import logging
import warnings

from llm_patch.core.interfaces import IAdapterRepository, IWeightGenerator
from llm_patch.core.models import AdapterManifest, DocumentContext

logger = logging.getLogger(__name__)

_DEPRECATION_MESSAGE = (
    "KnowledgeFusionOrchestrator is deprecated and will be removed in a future "
    "major release. Use llm_patch.CompilePipeline instead."
)


class KnowledgeFusionOrchestrator:
    """Legacy facade — delegates to register_callback / scan_existing.

    Accepts any source object that exposes the old ``IKnowledgeSource``
    duck-typed API (``register_callback``, ``scan_existing``, ``start``,
    ``stop``).  The backward-compat shim classes in
    ``llm_patch.sources.markdown`` and ``llm_patch.sources.wiki``
    satisfy this contract.
    """

    def __init__(
        self,
        source: object,
        generator: IWeightGenerator,
        repository: IAdapterRepository,
    ) -> None:
        warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=2)
        self._source = source
        self._generator = generator
        self._repository = repository

        # Legacy callback registration
        self._source.register_callback(self._on_document_changed)  # type: ignore[attr-defined]

    def _on_document_changed(self, context: DocumentContext) -> None:
        logger.info("Processing document: %s", context.document_id)
        self.process_document(context)

    def process_document(self, context: DocumentContext) -> AdapterManifest:
        weights = self._generator.generate(context)
        peft_config = self._generator.get_peft_config()
        manifest = self._repository.save(context.document_id, weights, peft_config)

        logger.info("Adapter %s saved to %s", manifest.adapter_id, manifest.storage_uri)
        return manifest

    def compile_all(self) -> list[AdapterManifest]:
        documents = self._source.scan_existing()  # type: ignore[attr-defined]
        logger.info("Compiling %d existing documents", len(documents))

        manifests: list[AdapterManifest] = []
        for doc in documents:
            manifest = self.process_document(doc)
            manifests.append(manifest)

        return manifests

    def start(self) -> None:
        self._source.start()  # type: ignore[attr-defined]

    def stop(self) -> None:
        self._source.stop()  # type: ignore[attr-defined]

    def __enter__(self) -> KnowledgeFusionOrchestrator:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
