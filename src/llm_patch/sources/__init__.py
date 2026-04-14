"""Document ingestion sources (Observer Pattern)."""

from llm_patch.sources.markdown_watcher import MarkdownDirectoryWatcher
from llm_patch.sources.wiki_source import WikiDocumentAggregator, WikiKnowledgeSource

__all__ = ["MarkdownDirectoryWatcher", "WikiDocumentAggregator", "WikiKnowledgeSource"]
