"""Composite data source — merges documents from multiple ``IDataSource`` instances.

Document IDs are namespaced as ``<source_name>:<original_id>`` to avoid
collisions between sources.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from llm_patch.core.interfaces import IDataSource
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)


class CompositeDataSource(IDataSource):
    """Merges documents from multiple data sources.

    Args:
        sources: One or more ``IDataSource`` implementations.
        namespace_ids: If ``True`` (default), prefix each document ID
            with its source name to avoid collisions.
    """

    def __init__(self, *sources: IDataSource, namespace_ids: bool = True) -> None:
        if not sources:
            raise ValueError("CompositeDataSource requires at least one source")
        self._sources = sources
        self._namespace_ids = namespace_ids

    @property
    def name(self) -> str:
        return "composite"

    def fetch_all(self) -> Iterable[DocumentContext]:
        for source in self._sources:
            for doc in source.fetch_all():
                if self._namespace_ids:
                    namespaced_id = f"{source.name}:{doc.document_id}"
                    merged_meta = {**doc.metadata, "original_source": source.name}
                    yield DocumentContext(
                        document_id=namespaced_id,
                        content=doc.content,
                        metadata=merged_meta,
                    )
                else:
                    yield doc

    def fetch_one(self, document_id: str) -> DocumentContext | None:
        if self._namespace_ids and ":" in document_id:
            source_name, _, inner_id = document_id.partition(":")
            for source in self._sources:
                if source.name == source_name:
                    doc = source.fetch_one(inner_id)
                    if doc is not None:
                        return DocumentContext(
                            document_id=document_id,
                            content=doc.content,
                            metadata={**doc.metadata, "original_source": source.name},
                        )
            return None
        return super().fetch_one(document_id)
