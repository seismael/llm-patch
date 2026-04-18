"""Backward-compat re-export — use ``llm_patch.sources.wiki`` instead."""

from llm_patch.sources.wiki import (  # noqa: F401
    WikiDataSource,
    WikiDocumentAggregator,
    WikiKnowledgeSource,
    WikiWatcher,
    _extract_wikilinks,
    _parse_frontmatter,
)

__all__ = [
    "WikiDataSource",
    "WikiDocumentAggregator",
    "WikiKnowledgeSource",
    "WikiWatcher",
    "_extract_wikilinks",
    "_parse_frontmatter",
]
