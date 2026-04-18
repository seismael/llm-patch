"""Tests for llm_patch.wiki.linker — WikiLinker."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_patch.wiki.linker import WikiLinker


@pytest.fixture()
def wiki_dir(tmp_path: Path) -> Path:
    """Set up a wiki directory with sample pages."""
    wiki = tmp_path / "wiki"
    (wiki / "summaries").mkdir(parents=True)
    (wiki / "concepts").mkdir()
    (wiki / "entities").mkdir()

    (wiki / "summaries" / "attention-paper.md").write_text(
        '---\ntitle: "Attention Paper"\ntype: summary\n---\n'
        "See [[Self-Attention]] and [[Transformer]].\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "self-attention.md").write_text(
        '---\ntitle: "Self-Attention"\ntype: concept\n---\n'
        "Related to [[Transformer]].\n",
        encoding="utf-8",
    )
    (wiki / "entities" / "transformer.md").write_text(
        '---\ntitle: "Transformer"\ntype: entity\n---\n'
        "Architecture for NLP.\n",
        encoding="utf-8",
    )
    return wiki


class TestWikiLinker:
    def test_resolve_link_by_slug(self, wiki_dir: Path) -> None:
        linker = WikiLinker(wiki_dir)
        result = linker.resolve_link("Self-Attention")
        assert result is not None
        assert result.name == "self-attention.md"

    def test_resolve_link_not_found(self, wiki_dir: Path) -> None:
        linker = WikiLinker(wiki_dir)
        assert linker.resolve_link("Nonexistent") is None

    def test_resolve_link_with_path(self, wiki_dir: Path) -> None:
        linker = WikiLinker(wiki_dir)
        result = linker.resolve_link("concepts/self-attention")
        assert result is not None

    def test_find_all_pages(self, wiki_dir: Path) -> None:
        linker = WikiLinker(wiki_dir)
        pages = linker.find_all_pages()
        assert len(pages) == 3

    def test_check_integrity_broken_link(self, tmp_path: Path) -> None:
        wiki = tmp_path / "wiki"
        (wiki / "concepts").mkdir(parents=True)
        # This page links to [[Missing Page]] which doesn't exist
        (wiki / "concepts" / "test.md").write_text(
            '---\ntitle: "Test"\ntype: concept\n---\nSee [[Missing Page]].\n',
            encoding="utf-8",
        )
        linker = WikiLinker(wiki)
        report = linker.check_integrity()
        assert len(report.broken_links) > 0

    def test_build_backlinks_index(self, wiki_dir: Path) -> None:
        linker = WikiLinker(wiki_dir)
        report = linker.check_integrity()
        # transformer.md should be linked from both attention-paper and self-attention
        transformer_key = "entities/transformer.md"
        assert transformer_key in report.backlinks
        assert len(report.backlinks[transformer_key]) >= 1

    def test_build_backlinks_index_markdown(self, wiki_dir: Path) -> None:
        linker = WikiLinker(wiki_dir)
        markdown = linker.build_backlinks_index()
        assert "# Backlinks Index" in markdown
        assert "Transformer" in markdown
