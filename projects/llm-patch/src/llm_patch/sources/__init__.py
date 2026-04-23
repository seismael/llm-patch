"""Data-source implementations (IDataSource / IKnowledgeStream)."""

from llm_patch.sources.composite import CompositeDataSource
from llm_patch.sources.jsonl import JsonlDataSource
from llm_patch.sources.markdown import (
    MarkdownDataSource,
    MarkdownDirectoryWatcher,
    MarkdownWatcher,
)
from llm_patch.sources.wiki import WikiDataSource, WikiDocumentAggregator, WikiKnowledgeSource

__all__ = [
    "CompositeDataSource",
    "JsonlDataSource",
    "MarkdownDataSource",
    "MarkdownDirectoryWatcher",
    "MarkdownWatcher",
    "WikiDataSource",
    "WikiDocumentAggregator",
    "WikiKnowledgeSource",
]
