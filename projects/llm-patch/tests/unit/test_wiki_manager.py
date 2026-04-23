"""Tests for llm_patch.wiki.manager — WikiManager facade."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.manager import WikiManager


@pytest.fixture()
def wiki_base(tmp_path: Path) -> Path:
    """Set up a base directory with a raw source."""
    base = tmp_path / "project"
    raw = base / "raw" / "papers"
    raw.mkdir(parents=True)
    (raw / "attention-is-all-you-need.md").write_text(
        "# Attention Is All You Need\n\n"
        "The dominant sequence transduction models are based on complex recurrent "
        "or convolutional neural networks that include an encoder and a decoder. "
        "The best performing models also connect the encoder and decoder through "
        "an attention mechanism. We propose a new simple network architecture, "
        "the [[Transformer]], based solely on [[Self-Attention]] mechanisms.\n\n"
        "## Architecture\n\nThe Transformer follows an encoder-decoder structure.\n\n"
        "## Results\n\nOur model achieves 28.4 BLEU on WMT 2014 English-to-German.\n",
        encoding="utf-8",
    )
    return base


@pytest.fixture()
def manager(wiki_base: Path) -> WikiManager:
    agent = MockWikiAgent()
    mgr = WikiManager(agent=agent, base_dir=wiki_base)
    mgr.init()
    return mgr


class TestWikiManager:
    def test_init_creates_directories(self, manager: WikiManager) -> None:
        assert manager.wiki_dir.is_dir()
        assert (manager.wiki_dir / "summaries").is_dir()
        assert (manager.wiki_dir / "concepts").is_dir()
        assert (manager.wiki_dir / "entities").is_dir()

    def test_ingest_creates_summary(self, manager: WikiManager, wiki_base: Path) -> None:
        source = wiki_base / "raw" / "papers" / "attention-is-all-you-need.md"
        result = manager.ingest(source)
        assert result.summary_page != ""
        assert len(result.pages_created) >= 1
        # Summary file should exist on disk
        summary_path = manager.wiki_dir / result.summary_page
        assert summary_path.exists()

    def test_ingest_extracts_entities(self, manager: WikiManager, wiki_base: Path) -> None:
        source = wiki_base / "raw" / "papers" / "attention-is-all-you-need.md"
        result = manager.ingest(source)
        assert len(result.entities_extracted) >= 1
        entity_names = [e.name for e in result.entities_extracted]
        assert any("Transformer" in n for n in entity_names)

    def test_ingest_updates_index(self, manager: WikiManager, wiki_base: Path) -> None:
        source = wiki_base / "raw" / "papers" / "attention-is-all-you-need.md"
        manager.ingest(source)
        assert len(manager.index) >= 1

    def test_query(self, manager: WikiManager, wiki_base: Path) -> None:
        source = wiki_base / "raw" / "papers" / "attention-is-all-you-need.md"
        manager.ingest(source)
        result = manager.query("What is Attention?")
        assert result.answer != ""

    def test_query_save_as_synthesis(self, manager: WikiManager, wiki_base: Path) -> None:
        source = wiki_base / "raw" / "papers" / "attention-is-all-you-need.md"
        manager.ingest(source)
        result = manager.query("What is Attention?", save_as_synthesis=True)
        if result.filed_as:
            assert (manager.wiki_dir / result.filed_as).exists()

    def test_lint(self, manager: WikiManager, wiki_base: Path) -> None:
        source = wiki_base / "raw" / "papers" / "attention-is-all-you-need.md"
        manager.ingest(source)
        report = manager.lint()
        # Lint should run without errors (may or may not find issues)
        assert report.issue_count >= 0

    def test_compile_all(self, manager: WikiManager) -> None:
        results = manager.compile_all()
        assert len(results) == 1  # one raw source

    def test_compile_all_skips_ingested(self, manager: WikiManager, wiki_base: Path) -> None:
        _ = wiki_base
        manager.compile_all()
        results = manager.compile_all()
        assert len(results) == 0  # already ingested

    def test_status(self, manager: WikiManager, wiki_base: Path) -> None:
        _ = wiki_base
        status = manager.status()
        assert status["raw_sources"] >= 1
        assert "wiki_pages" in status
        assert "index_entries" in status

    def test_read_page_not_found(self, manager: WikiManager) -> None:
        assert manager.read_page("nonexistent.md") is None
