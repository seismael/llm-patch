"""Backward-compat re-export — use ``llm_patch.sources.markdown`` instead."""

from llm_patch.sources.markdown import (
    MarkdownDataSource,
    MarkdownDirectoryWatcher,
    MarkdownWatcher,
    _derive_document_id,
    _matches_patterns,
    _read_document,
)

__all__ = [
    "MarkdownDataSource",
    "MarkdownDirectoryWatcher",
    "MarkdownWatcher",
    "_derive_document_id",
    "_matches_patterns",
    "_read_document",
]
