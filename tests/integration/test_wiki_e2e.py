"""Integration test — full wiki cycle: init → ingest → query → lint."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.manager import WikiManager


@pytest.fixture()
def wiki_project(tmp_path: Path) -> Path:
    """Create a project directory with multiple raw sources."""
    base = tmp_path / "project"
    raw = base / "raw" / "papers"
    raw.mkdir(parents=True)

    (raw / "attention-is-all-you-need.md").write_text(
        "# Attention Is All You Need\n\n"
        "We propose the [[Transformer]] architecture based on [[Self-Attention]].\n\n"
        "## Architecture\n\nEncoder-decoder with multi-head attention.\n\n"
        "## Results\n\n28.4 BLEU on WMT 2014 English-to-German.\n",
        encoding="utf-8",
    )
    (raw / "gpt3-few-shot-learners.md").write_text(
        "# GPT-3: Language Models are Few-Shot Learners\n\n"
        "We train [[GPT-3]], an autoregressive language model using the "
        "[[Transformer]] architecture with 175 billion parameters.\n\n"
        "## Scaling Laws\n\nPerformance scales smoothly with model size.\n\n"
        "## Few-Shot Learning\n\nGPT-3 can perform tasks with few examples.\n",
        encoding="utf-8",
    )
    (raw / "lora-low-rank-adaptation.md").write_text(
        "# LoRA: Low-Rank Adaptation\n\n"
        "We propose [[LoRA]], a method for efficient fine-tuning of "
        "[[Transformer]] models using low-rank decomposition.\n\n"
        "## Method\n\nFreeze original weights, inject trainable rank decomposition.\n\n"
        "## Results\n\nMatches full fine-tuning with 10000x fewer parameters.\n",
        encoding="utf-8",
    )
    return base


@pytest.mark.integration
class TestWikiE2E:
    def test_full_cycle(self, wiki_project: Path) -> None:
        """Test the full raw → wiki → query → lint cycle."""
        agent = MockWikiAgent()
        manager = WikiManager(agent=agent, base_dir=wiki_project)

        # Step 1: Initialize
        manager.init()
        assert (wiki_project / "wiki" / "summaries").is_dir()
        assert (wiki_project / "wiki" / "concepts").is_dir()

        # Step 2: Compile all raw sources
        results = manager.compile_all()
        assert len(results) == 3

        # Verify pages were created
        total_created = sum(len(r.pages_created) for r in results)
        assert total_created >= 3  # at least 3 summary pages

        # Step 3: Check index
        status = manager.status()
        assert status["wiki_pages"] >= 3
        assert status["index_entries"] >= 3

        # Step 4: Verify Transformer entity is shared across sources
        entries = manager.index.search("transformer")
        assert len(entries) >= 1, "Transformer should appear in the index"

        # Step 5: Query
        result = manager.query("What is the Transformer?")
        assert result.answer != ""

        # Step 6: Query and save as synthesis
        result = manager.query(
            "How does LoRA relate to the Transformer?",
            save_as_synthesis=True,
        )
        if result.filed_as:
            assert (wiki_project / "wiki" / result.filed_as).exists()

        # Step 7: Lint
        report = manager.lint()
        assert report.issue_count >= 0  # should run without errors

        # Step 8: Second compile should be a no-op
        second_results = manager.compile_all()
        assert len(second_results) == 0

    def test_incremental_ingest(self, wiki_project: Path) -> None:
        """Test adding a new source after initial compilation."""
        agent = MockWikiAgent()
        manager = WikiManager(agent=agent, base_dir=wiki_project)
        manager.init()

        # Compile initial sources
        manager.compile_all()
        initial_count = len(manager.index)

        # Add a new raw source
        new_source = wiki_project / "raw" / "papers" / "new-paper.md"
        new_source.write_text(
            "# A New Paper\n\nIntroducing [[NewConcept]] using [[Transformer]].\n",
            encoding="utf-8",
        )

        # Compile again — should only process the new source
        results = manager.compile_all()
        assert len(results) == 1
        assert len(manager.index) > initial_count
