"""Tests for llm_patch.wiki.index — WikiIndex."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_patch.wiki.index import IndexEntry, WikiIndex


@pytest.fixture()
def wiki_index(tmp_path: Path) -> WikiIndex:
    return WikiIndex(tmp_path / "wiki" / "index.md")


class TestWikiIndex:
    def test_add_and_get_entry(self, wiki_index: WikiIndex) -> None:
        entry = IndexEntry(
            path="concepts/attention.md",
            title="Attention",
            summary="A mechanism for weighting.",
            tags=["nlp"],
            category="concepts",
        )
        wiki_index.add_entry(entry)
        assert wiki_index.get_entry("concepts/attention.md") is entry

    def test_remove_entry(self, wiki_index: WikiIndex) -> None:
        wiki_index.add_entry(IndexEntry(path="x.md", title="X"))
        assert wiki_index.remove_entry("x.md") is True
        assert wiki_index.get_entry("x.md") is None

    def test_remove_missing(self, wiki_index: WikiIndex) -> None:
        assert wiki_index.remove_entry("nope.md") is False

    def test_search(self, wiki_index: WikiIndex) -> None:
        wiki_index.add_entry(IndexEntry(path="a.md", title="Attention", summary="Weighting"))
        wiki_index.add_entry(IndexEntry(path="b.md", title="Transformer", summary="Architecture"))
        results = wiki_index.search("attention")
        assert len(results) == 1
        assert results[0].path == "a.md"

    def test_find_by_tag(self, wiki_index: WikiIndex) -> None:
        wiki_index.add_entry(IndexEntry(path="a.md", title="A", tags=["nlp"]))
        wiki_index.add_entry(IndexEntry(path="b.md", title="B", tags=["cv"]))
        results = wiki_index.find_by_tag("nlp")
        assert len(results) == 1

    def test_find_by_type(self, wiki_index: WikiIndex) -> None:
        wiki_index.add_entry(IndexEntry(path="a.md", title="A", category="concepts"))
        wiki_index.add_entry(IndexEntry(path="b.md", title="B", category="entities"))
        results = wiki_index.find_by_type("concepts")
        assert len(results) == 1

    def test_save_and_reload(self, wiki_index: WikiIndex) -> None:
        wiki_index.add_entry(
            IndexEntry(
                path="concepts/attention.md",
                title="Attention",
                summary="A mechanism",
                tags=["nlp"],
                category="concepts",
            )
        )
        wiki_index.save()

        # Reload from disk
        reloaded = WikiIndex(wiki_index._path)
        entry = reloaded.get_entry("concepts/attention.md")
        assert entry is not None
        assert entry.title == "Attention"
        assert "nlp" in entry.tags

    def test_len(self, wiki_index: WikiIndex) -> None:
        assert len(wiki_index) == 0
        wiki_index.add_entry(IndexEntry(path="a.md", title="A"))
        assert len(wiki_index) == 1

    def test_update_entry(self, wiki_index: WikiIndex) -> None:
        wiki_index.add_entry(IndexEntry(path="a.md", title="Old"))
        assert wiki_index.update_entry("a.md", title="New") is True
        assert wiki_index.get_entry("a.md").title == "New"  # type: ignore[union-attr]

    def test_update_missing(self, wiki_index: WikiIndex) -> None:
        assert wiki_index.update_entry("nope.md", title="X") is False
