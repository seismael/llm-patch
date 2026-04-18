"""Tests for llm_patch.wiki.log — WikiLog."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_patch.wiki.log import WikiLog


@pytest.fixture()
def wiki_log(tmp_path: Path) -> WikiLog:
    return WikiLog(tmp_path / "wiki" / "log.md")


class TestWikiLog:
    def test_append_creates_file(self, wiki_log: WikiLog) -> None:
        entry = wiki_log.append(
            operation="ingest",
            description="Ingested paper.md",
            pages_touched=["summaries/paper.md"],
        )
        assert entry.operation == "ingest"
        assert wiki_log._path.exists()

    def test_recent(self, wiki_log: WikiLog) -> None:
        wiki_log.append("ingest", "First")
        wiki_log.append("query", "Second")
        wiki_log.append("lint", "Third")
        recent = wiki_log.recent(2)
        assert len(recent) == 2
        assert recent[-1].operation == "lint"

    def test_filter_by_operation(self, wiki_log: WikiLog) -> None:
        wiki_log.append("ingest", "One")
        wiki_log.append("query", "Two")
        wiki_log.append("ingest", "Three")
        ingests = wiki_log.filter_by_operation("ingest")
        assert len(ingests) == 2

    def test_len(self, wiki_log: WikiLog) -> None:
        assert len(wiki_log) == 0
        wiki_log.append("ingest", "test")
        assert len(wiki_log) == 1

    def test_pages_touched_recorded(self, wiki_log: WikiLog) -> None:
        wiki_log.append("ingest", "test", ["a.md", "b.md"])
        entries = wiki_log.recent(1)
        assert entries[0].pages_touched == ["a.md", "b.md"]
