"""Integration tests for WikiKnowledgeSource and the wiki pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from llm_patch.core.config import WatcherConfig
from llm_patch.core.models import DocumentContext
from llm_patch.orchestrator import KnowledgeFusionOrchestrator
from llm_patch.sources.wiki_source import (
    WikiDocumentAggregator,
    WikiKnowledgeSource,
    _extract_wikilinks,
    _parse_frontmatter,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_dir(tmp_path: Path) -> Path:
    """A temporary wiki directory with sources, entities, and concepts."""
    sources = tmp_path / "sources"
    entities = tmp_path / "entities"
    concepts = tmp_path / "concepts"
    sources.mkdir()
    entities.mkdir()
    concepts.mkdir()

    (sources / "attention-paper.md").write_text(
        "---\ntitle: Attention Is All You Need\nauthors: Vaswani et al.\nyear: 2017\n---\n\n"
        "# Attention Is All You Need\n\n"
        "The [[Transformer]] architecture uses [[Self-Attention]] mechanisms.\n",
        encoding="utf-8",
    )

    (sources / "lora-paper.md").write_text(
        "---\ntitle: LoRA\nauthors: Hu et al.\nyear: 2021\n---\n\n"
        "# LoRA\n\n"
        "Low-rank adaptation of large language models using [[LoRA]] matrices.\n",
        encoding="utf-8",
    )

    (entities / "transformer.md").write_text(
        "---\ntitle: Transformer\ntype: entity\n---\n\n"
        "# Transformer\n\n"
        "An architecture based on self-attention for sequence transduction.\n",
        encoding="utf-8",
    )

    (entities / "self-attention.md").write_text(
        "---\ntitle: Self-Attention\ntype: entity\n---\n\n"
        "# Self-Attention\n\n"
        "Scaled dot-product attention mechanism.\n",
        encoding="utf-8",
    )

    (concepts / "lora.md").write_text(
        "---\ntitle: LoRA\ntype: concept\n---\n\n"
        "# LoRA\n\n"
        "Low-rank decomposition of weight update matrices.\n",
        encoding="utf-8",
    )

    # A non-matching file (should be ignored)
    (sources / "notes.txt").write_text("Not a markdown file.", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def wiki_config(wiki_dir: Path) -> WatcherConfig:
    """WatcherConfig pointing at the wiki directory."""
    return WatcherConfig(directory=wiki_dir)


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------


class TestFrontmatterParsing:
    def test_basic_frontmatter(self) -> None:
        text = "---\ntitle: Hello\nauthor: World\n---\n\nBody text."
        meta, body = _parse_frontmatter(text)
        assert meta == {"title": "Hello", "author": "World"}
        assert body.strip() == "Body text."

    def test_no_frontmatter(self) -> None:
        text = "# Just a heading\n\nSome content."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\n\nContent."
        meta, body = _parse_frontmatter(text)
        assert meta == {}
        assert body.strip() == "Content."


# ---------------------------------------------------------------------------
# Wikilink extraction
# ---------------------------------------------------------------------------


class TestWikilinkExtraction:
    def test_basic_links(self) -> None:
        text = "Uses [[Transformer]] and [[Self-Attention]] mechanisms."
        links = _extract_wikilinks(text)
        assert links == ["Transformer", "Self-Attention"]

    def test_no_links(self) -> None:
        assert _extract_wikilinks("Plain text without links.") == []

    def test_display_text_links(self) -> None:
        text = "See [[Transformer|the architecture]]."
        links = _extract_wikilinks(text)
        assert links == ["Transformer"]


# ---------------------------------------------------------------------------
# WikiDocumentAggregator
# ---------------------------------------------------------------------------


class TestWikiDocumentAggregator:
    def test_aggregates_linked_pages(self, wiki_dir: Path) -> None:
        agg = WikiDocumentAggregator(wiki_dir)
        doc = DocumentContext(
            document_id="sources/attention-paper",
            content="# Attention\n\nUses [[Transformer]] and [[Self-Attention]].",
            metadata={"wikilinks": ["Transformer", "Self-Attention"]},
        )
        result = agg.aggregate(doc)

        assert result.document_id == "sources/attention-paper"
        assert "Transformer" in result.content
        assert "Self-Attention" in result.content
        assert result.metadata.get("aggregated") is True
        assert set(result.metadata["resolved_links"]) == {"Transformer", "Self-Attention"}

    def test_no_links_returns_original(self, wiki_dir: Path) -> None:
        agg = WikiDocumentAggregator(wiki_dir)
        doc = DocumentContext(
            document_id="test",
            content="No links here.",
            metadata={},
        )
        result = agg.aggregate(doc)
        assert result is doc

    def test_unresolvable_links(self, wiki_dir: Path) -> None:
        agg = WikiDocumentAggregator(wiki_dir)
        doc = DocumentContext(
            document_id="test",
            content="See [[Nonexistent]].",
            metadata={"wikilinks": ["Nonexistent"]},
        )
        result = agg.aggregate(doc)
        # Link didn't resolve, so doc is returned unchanged
        assert result is doc


# ---------------------------------------------------------------------------
# WikiKnowledgeSource — scan_existing
# ---------------------------------------------------------------------------


class TestWikiKnowledgeSourceScan:
    def test_scan_finds_all_markdown(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config)
        docs = source.scan_existing()

        # 2 sources + 2 entities + 1 concept = 5 .md files
        assert len(docs) == 5

        ids = {d.document_id for d in docs}
        assert "sources/attention-paper" in ids
        assert "sources/lora-paper" in ids
        assert "entities/transformer" in ids

    def test_scan_parses_frontmatter(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config)
        docs = source.scan_existing()
        attention = next(d for d in docs if d.document_id == "sources/attention-paper")

        assert attention.metadata["title"] == "Attention Is All You Need"
        assert attention.metadata["year"] == "2017"

    def test_scan_extracts_wikilinks(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config)
        docs = source.scan_existing()
        attention = next(d for d in docs if d.document_id == "sources/attention-paper")

        assert "wikilinks" in attention.metadata
        assert "Transformer" in attention.metadata["wikilinks"]

    def test_scan_with_aggregation(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config, aggregate=True)
        docs = source.scan_existing()
        attention = next(d for d in docs if d.document_id == "sources/attention-paper")

        # Aggregated content should include linked entity text
        assert "self-attention for sequence transduction" in attention.content.lower()
        assert attention.metadata.get("aggregated") is True

    def test_scan_empty_dir(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty_wiki"
        empty.mkdir()
        config = WatcherConfig(directory=empty)
        source = WikiKnowledgeSource(config)
        assert source.scan_existing() == []

    def test_scan_nonexistent_dir(self, tmp_path: Path) -> None:
        config = WatcherConfig(directory=tmp_path / "does_not_exist")
        source = WikiKnowledgeSource(config)
        assert source.scan_existing() == []


# ---------------------------------------------------------------------------
# Full pipeline with mock generator
# ---------------------------------------------------------------------------


class TestWikiPipeline:
    def test_compile_all_produces_manifests(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config)
        generator = MagicMock()
        generator.generate.return_value = {"lora_A": MagicMock(), "lora_B": MagicMock()}
        generator.get_peft_config.return_value = {
            "r": 8,
            "target_modules": ["q_proj"],
            "peft_type": "LORA",
        }

        repository = MagicMock()
        repository.save.side_effect = lambda aid, _weights, _config: MagicMock(
            adapter_id=aid, rank=8, target_modules=["q_proj"], storage_uri=f"/out/{aid}"
        )

        orchestrator = KnowledgeFusionOrchestrator(
            source=source,
            generator=generator,
            repository=repository,
        )
        manifests = orchestrator.compile_all()

        assert len(manifests) == 5
        assert generator.generate.call_count == 5
        assert repository.save.call_count == 5

    def test_compile_all_with_aggregation(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config, aggregate=True)
        generator = MagicMock()
        generator.generate.return_value = {"lora_A": MagicMock()}
        generator.get_peft_config.return_value = {"r": 4, "target_modules": [], "peft_type": "LORA"}

        repository = MagicMock()
        repository.save.side_effect = lambda aid, _weights, _config: MagicMock(
            adapter_id=aid, rank=4, target_modules=[], storage_uri=f"/out/{aid}"
        )

        orchestrator = KnowledgeFusionOrchestrator(
            source=source, generator=generator, repository=repository
        )
        orchestrator.compile_all()

        # The attention paper doc should have been aggregated
        attention_call = next(
            c
            for c in generator.generate.call_args_list
            if c.args[0].document_id == "sources/attention-paper"
        )
        doc: DocumentContext = attention_call.args[0]
        assert doc.metadata.get("aggregated") is True

    def test_callback_on_register(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config)
        cb = MagicMock()
        source.register_callback(cb)

        # Simulate processing a document through the source
        docs = source.scan_existing()
        assert len(docs) > 0
        # Callbacks are invoked by the watcher, not scan_existing
        assert cb.call_count == 0

    def test_context_manager(self, wiki_config: WatcherConfig) -> None:
        source = WikiKnowledgeSource(wiki_config)
        with source:
            # Observer should be running (non-None)
            assert source._observer is not None
        # After exit, observer should be stopped
        assert source._observer is None
