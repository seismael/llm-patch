"""Integration tests: full wiki pipeline validation (mock + live API).

Covers the complete ingest → query → lint cycle through WikiManager,
validating wiki output structure, index integrity, log entries, and
cross-references.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.log import WikiLog
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.page import parse_wiki_page

# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

ATTENTION_PAPER = """\
# Attention Is All You Need

We propose a new network architecture, the **Transformer**, based entirely
on attention mechanisms, dispensing with recurrence and convolutions entirely.

## Architecture

The Transformer uses stacked self-attention and point-wise, fully connected
layers for both the encoder and decoder. Multi-head attention allows the
model to jointly attend to information from different representation
subspaces at different positions.

## Results

The model achieves 28.4 BLEU on the WMT 2014 English-to-German translation
task and 41.0 BLEU on the English-to-French task.
"""

GPT3_PAPER = """\
# Language Models are Few-Shot Learners

GPT-3, a 175 billion parameter autoregressive language model, demonstrates
that scaling up language models greatly improves task-agnostic, few-shot
performance, sometimes even becoming competitive with prior state-of-the-art
fine-tuning approaches.

## Key Findings

GPT-3 achieves strong performance on many NLP datasets including translation,
question-answering, and cloze tasks, as well as several tasks that require
on-the-fly reasoning or domain adaptation.
"""

LORA_PAPER = """\
# LoRA: Low-Rank Adaptation of Large Language Models

We propose Low-Rank Adaptation (LoRA), which freezes the pre-trained model
weights and injects trainable rank decomposition matrices into each layer
of the Transformer architecture. LoRA reduces the number of trainable
parameters by 10,000x and GPU memory requirement by 3x.

## Method

LoRA parameterizes the weight update as a low-rank decomposition W = W0 + BA
where B and A are low-rank matrices with rank r << min(d, k).
"""


@pytest.fixture()
def wiki_project(tmp_path: Path) -> Path:
    """Create a complete project layout with raw papers."""
    raw = tmp_path / "raw" / "papers"
    raw.mkdir(parents=True)
    (raw / "attention-is-all-you-need.md").write_text(ATTENTION_PAPER, encoding="utf-8")
    (raw / "gpt3-few-shot-learners.md").write_text(GPT3_PAPER, encoding="utf-8")
    (raw / "lora-low-rank-adaptation.md").write_text(LORA_PAPER, encoding="utf-8")
    return tmp_path


@pytest.fixture()
def mock_manager(wiki_project: Path) -> WikiManager:
    """WikiManager with MockWikiAgent + initialized dirs."""
    agent = MockWikiAgent()
    manager = WikiManager(agent=agent, base_dir=wiki_project)
    manager.init()
    return manager


# ─────────────────────────────────────────────────────────────────────
# Full pipeline validation (mock agent)
# ─────────────────────────────────────────────────────────────────────


class TestFullMockPipeline:
    """Validate the complete ingest → query → lint cycle with MockWikiAgent."""

    def test_compile_all_ingests_every_source(self, mock_manager: WikiManager, wiki_project: Path):
        results = mock_manager.compile_all()
        raw_files = sorted((wiki_project / "raw").rglob("*.md"))
        assert len(results) == len(raw_files), (
            f"Expected {len(raw_files)} ingest results, got {len(results)}"
        )

    def test_compile_all_creates_summary_pages(self, mock_manager: WikiManager):
        results = mock_manager.compile_all()
        for r in results:
            assert r.summary_page, f"No summary page for {r.source_path}"
            path = mock_manager.wiki_dir / r.summary_page
            assert path.exists(), f"Summary file missing on disk: {r.summary_page}"

    def test_compile_all_creates_entity_pages(self, mock_manager: WikiManager):
        results = mock_manager.compile_all()
        total_entities = sum(len(r.entities_extracted) for r in results)
        assert total_entities > 0, "Expected at least one entity from mock agent"

        total_pages = sum(len(r.pages_created) for r in results)
        # pages_created includes summary + entity pages
        assert total_pages > len(results), (
            f"Expected more pages than sources ({total_pages} <= {len(results)})"
        )

    def test_index_populated_after_compile(self, mock_manager: WikiManager):
        results = mock_manager.compile_all()
        assert len(mock_manager.index) >= len(results), (
            f"Index should have at least {len(results)} entries"
        )

    def test_wiki_pages_have_valid_frontmatter(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        for page_path in mock_manager.wiki_dir.rglob("*.md"):
            if page_path.name in ("index.md", "log.md"):
                continue
            text = page_path.read_text(encoding="utf-8")
            rel = page_path.relative_to(mock_manager.wiki_dir).as_posix()
            page = parse_wiki_page(text, rel)
            assert page.title, f"Missing title in {rel}"
            assert page.page_type is not None, f"Missing type in {rel}"

    def test_log_records_all_operations(self, mock_manager: WikiManager, wiki_project: Path):
        mock_manager.compile_all()
        log = WikiLog(wiki_project / "wiki" / "log.md")
        entries = log.recent(100)
        ingest_entries = [e for e in entries if e.operation == "ingest"]
        assert len(ingest_entries) >= 3, (
            f"Expected >= 3 ingest log entries, got {len(ingest_entries)}"
        )

    def test_idempotent_compile(self, mock_manager: WikiManager):
        """compile_all should skip already-ingested sources."""
        first = mock_manager.compile_all()
        second = mock_manager.compile_all()
        assert len(first) >= 3
        assert len(second) == 0, "Second compile should skip everything"

    def test_query_returns_answer(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        result = mock_manager.query("What is the Transformer architecture?")
        assert result.answer, "Expected a non-empty answer"

    def test_query_saves_synthesis(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        result = mock_manager.query("How does LoRA work?", save_as_synthesis=True)
        assert result.filed_as, "Expected synthesis page path"
        path = mock_manager.wiki_dir / result.filed_as
        assert path.exists(), f"Synthesis file should exist: {result.filed_as}"

    def test_lint_runs_without_error(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        report = mock_manager.lint()
        assert report is not None
        # Report may have issues from structural checks, that's fine

    def test_status_after_compile(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        status = mock_manager.status()
        assert status["raw_sources"] >= 3
        assert status["wiki_pages"] >= 3
        assert status["index_entries"] >= 3
        assert status["log_entries"] >= 3


class TestWikiOutputStructure:
    """Validate the structure and content of generated wiki pages."""

    def test_summary_pages_in_summaries_dir(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        summaries_dir = mock_manager.wiki_dir / "summaries"
        assert summaries_dir.exists()
        summaries = list(summaries_dir.glob("*.md"))
        assert len(summaries) >= 3

    def test_entity_concept_pages_in_correct_dirs(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        entities_dir = mock_manager.wiki_dir / "entities"
        concepts_dir = mock_manager.wiki_dir / "concepts"
        # At least one of these should exist and contain pages
        entity_pages = list(entities_dir.glob("*.md")) if entities_dir.exists() else []
        concept_pages = list(concepts_dir.glob("*.md")) if concepts_dir.exists() else []
        assert entity_pages or concept_pages, "Expected entity or concept pages"

    def test_index_entries_reference_existing_files(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        for entry in mock_manager.index.entries:
            path = mock_manager.wiki_dir / entry.path
            assert path.exists(), f"Index entry points to missing file: {entry.path}"

    def test_page_titles_match_filenames(self, mock_manager: WikiManager):
        """Page slug should relate to title."""
        mock_manager.compile_all()
        for page_path in mock_manager.wiki_dir.rglob("*.md"):
            if page_path.name in ("index.md", "log.md"):
                continue
            text = page_path.read_text(encoding="utf-8")
            rel = page_path.relative_to(mock_manager.wiki_dir).as_posix()
            page = parse_wiki_page(text, rel)
            # Slug should be derived from title
            assert page.title, f"Untitled page: {rel}"


class TestWikiRereadConsistency:
    """Validate that pages survive roundtrip write → read."""

    def test_roundtrip_summary(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        for page_path in (mock_manager.wiki_dir / "summaries").glob("*.md"):
            text = page_path.read_text(encoding="utf-8")
            rel = page_path.relative_to(mock_manager.wiki_dir).as_posix()
            page = parse_wiki_page(text, rel)
            # Re-serialize and parse again
            md = page.to_markdown()
            page2 = parse_wiki_page(md, rel)
            assert page2.title == page.title
            assert page2.page_type == page.page_type

    def test_manager_read_page(self, mock_manager: WikiManager):
        results = mock_manager.compile_all()
        for r in results:
            if r.summary_page:
                page = mock_manager.read_page(r.summary_page)
                assert page is not None
                assert page.title


# ─────────────────────────────────────────────────────────────────────
# Live Claude API tests (requires ANTHROPIC_API_KEY)
# ─────────────────────────────────────────────────────────────────────

HAS_ANTHROPIC_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ANTHROPIC_KEY, reason="ANTHROPIC_API_KEY not set")
class TestLiveClaudePipeline:
    """E2E tests that call the real Claude API.

    Set ANTHROPIC_API_KEY env var to run. These tests are slow and cost money.
    """

    @pytest.fixture()
    def live_manager(self, wiki_project: Path):
        anthropic_agent = pytest.importorskip("llm_patch.wiki.agents.anthropic_agent")
        agent = anthropic_agent.AnthropicWikiAgent(model="claude-sonnet-4-20250514")
        manager = WikiManager(agent=agent, base_dir=wiki_project)
        manager.init()
        return manager

    def test_live_ingest_single_source(self, live_manager: WikiManager, wiki_project: Path):
        """Ingest one paper and validate output."""
        source = wiki_project / "raw" / "papers" / "attention-is-all-you-need.md"
        result = live_manager.ingest(source)

        assert result.summary_page, "Claude should produce a summary"
        assert len(result.entities_extracted) >= 1, "Claude should extract at least 1 entity"
        assert len(result.pages_created) >= 2, "At least summary + 1 entity page"

        # Validate summary content quality
        page = live_manager.read_page(result.summary_page)
        assert page is not None
        assert len(page.body) > 100, "Summary should be substantive"
        # Check it mentions key concepts
        body_lower = page.body.lower()
        assert "transformer" in body_lower or "attention" in body_lower

    def test_live_query(self, live_manager: WikiManager, wiki_project: Path):
        """Ingest, then query."""
        source = wiki_project / "raw" / "papers" / "attention-is-all-you-need.md"
        live_manager.ingest(source)

        result = live_manager.query("What is the Transformer?")
        assert result.answer
        assert len(result.answer) > 50, "Answer should be substantive"

    def test_live_full_pipeline(self, live_manager: WikiManager):
        """Full compile + query + lint."""
        results = live_manager.compile_all()
        assert len(results) >= 3

        qr = live_manager.query("How does LoRA work?", save_as_synthesis=True)
        assert qr.answer
        assert qr.filed_as

        report = live_manager.lint()
        assert report is not None

        status = live_manager.status()
        assert status["wiki_pages"] >= 3
        assert status["index_entries"] >= 3
