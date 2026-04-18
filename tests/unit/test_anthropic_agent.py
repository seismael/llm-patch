"""Tests for AnthropicWikiAgent — using mocked Anthropic SDK.

These tests verify all 6 IWikiAgent methods are correctly implemented
without making real API calls.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_patch.wiki.operations import EntityExtraction, LintReport, QueryResult
from llm_patch.wiki.page import (
    ConfidenceLevel,
    PageType,
    WikiPage,
    WikiPageFrontmatter,
)
from llm_patch.wiki.schema import WikiSchema

# Guard: skip all if anthropic not installed
anthropic = pytest.importorskip("anthropic")


@pytest.fixture()
def mock_client():
    """A mock Anthropic client that returns controlled responses."""
    client = MagicMock()
    return client


@pytest.fixture()
def agent(mock_client):
    """AnthropicWikiAgent with a mocked client."""
    from llm_patch.wiki.agents.anthropic_agent import AnthropicWikiAgent

    with patch("llm_patch.wiki.agents.anthropic_agent.resolve_api_key", return_value="test-key"):
        a = AnthropicWikiAgent(api_key="test-key-not-real")
    a._client = mock_client  # Replace with mock
    return a


def _make_response(text: str) -> MagicMock:
    """Build a mock Anthropic Message response."""
    msg = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    msg.content = [content_block]
    msg.stop_reason = "end_turn"
    return msg


class TestAnthropicSummarize:
    def test_returns_markdown_body(self, agent, mock_client):
        expected = (
            "## Key Points\n\n- Transformers use self-attention\n\n"
            "## Relevant Concepts\n\nSee [[Self-Attention]].\n\n"
            "## Source Metadata\n\n- Year: 2017\n"
        )
        mock_client.messages.create.return_value = _make_response(expected)

        result = agent.summarize("Some paper text...", "Attention Paper")

        assert "Key Points" in result
        assert "Self-Attention" in result
        mock_client.messages.create.assert_called_once()
        call_kwargs = mock_client.messages.create.call_args.kwargs
        assert "user" in call_kwargs["messages"][0]["role"]

    def test_truncates_long_input(self, agent, mock_client):
        mock_client.messages.create.return_value = _make_response("Summary.")
        long_text = "word " * 10000
        agent.summarize(long_text, "Long Doc")
        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_text = call_kwargs["messages"][0]["content"]
        assert len(user_text) < len(long_text)


class TestAnthropicExtractEntities:
    def test_parses_json_array(self, agent, mock_client):
        entities_json = json.dumps([
            {"name": "Transformer", "entity_type": "entity", "confidence": 0.95,
             "description": "Neural network architecture"},
            {"name": "Self-Attention", "entity_type": "concept", "confidence": 0.9,
             "description": "Attention mechanism"},
        ])
        mock_client.messages.create.return_value = _make_response(entities_json)

        result = agent.extract_entities("Some text about transformers...")

        assert len(result) == 2
        assert result[0].name == "Transformer"
        assert result[0].entity_type == "entity"
        assert result[1].name == "Self-Attention"

    def test_handles_code_fenced_json(self, agent, mock_client):
        fenced = '```json\n[{"name": "LoRA", "entity_type": "concept"}]\n```'
        mock_client.messages.create.return_value = _make_response(fenced)

        result = agent.extract_entities("text")
        assert len(result) == 1
        assert result[0].name == "LoRA"

    def test_handles_invalid_json(self, agent, mock_client):
        mock_client.messages.create.return_value = _make_response("not json at all")
        result = agent.extract_entities("text")
        assert result == []

    def test_handles_partial_objects(self, agent, mock_client):
        """Objects missing 'name' should be skipped."""
        entities_json = json.dumps([
            {"name": "Valid", "entity_type": "entity"},
            {"entity_type": "concept"},  # missing name
        ])
        mock_client.messages.create.return_value = _make_response(entities_json)
        result = agent.extract_entities("text")
        assert len(result) == 1


class TestAnthropicGeneratePage:
    def test_generates_wiki_page(self, agent, mock_client):
        body = "# LoRA\n\n## Definition\n\nLow-Rank Adaptation...\n"
        mock_client.messages.create.return_value = _make_response(body)

        result = agent.generate_page(
            page_type="concept",
            title="LoRA",
            content="Low-Rank Adaptation method for fine-tuning.",
        )

        assert isinstance(result, WikiPage)
        assert result.title == "LoRA"
        assert result.page_type == PageType.CONCEPT
        assert "LoRA" in result.body
        assert result.frontmatter.confidence == ConfidenceLevel.MEDIUM

    def test_includes_context_in_prompt(self, agent, mock_client):
        mock_client.messages.create.return_value = _make_response("body")
        context_page = WikiPage(
            frontmatter=WikiPageFrontmatter(title="Transformer"),
            body="Related stuff.",
            path="entities/transformer.md",
        )
        agent.generate_page("entity", "LoRA", "content", context=[context_page])
        call_kwargs = mock_client.messages.create.call_args.kwargs
        user_msg = call_kwargs["messages"][0]["content"]
        assert "Transformer" in user_msg


class TestAnthropicUpdatePage:
    def test_updates_existing_page(self, agent, mock_client):
        existing = WikiPage(
            frontmatter=WikiPageFrontmatter(title="LoRA", type=PageType.CONCEPT),
            body="# LoRA\n\nOriginal content.",
            path="concepts/lora.md",
        )
        updated_body = "# LoRA\n\nOriginal content.\n\n## Update\n\nNew finding about rank."
        mock_client.messages.create.return_value = _make_response(updated_body)

        result = agent.update_page(existing, "New finding about rank selection.")

        assert isinstance(result, WikiPage)
        assert result.path == "concepts/lora.md"
        assert "New finding" in result.body or "rank" in result.body


class TestAnthropicAnswerQuery:
    def test_answers_with_citations(self, agent, mock_client):
        answer = "The Transformer uses [[Self-Attention]] mechanisms."
        mock_client.messages.create.return_value = _make_response(answer)

        context = [
            WikiPage(
                frontmatter=WikiPageFrontmatter(title="Attention Paper"),
                body="Attention is all you need.",
                path="summaries/attention.md",
            ),
        ]
        result = agent.answer_query("What is the Transformer?", context)

        assert isinstance(result, QueryResult)
        assert result.answer != ""
        assert len(result.cited_pages) >= 1

    def test_no_context_returns_empty(self, agent, mock_client):
        result = agent.answer_query("Random question?", [])
        assert "No relevant" in result.answer
        mock_client.messages.create.assert_not_called()


class TestAnthropicLintPages:
    def test_parses_lint_json(self, agent, mock_client):
        lint_json = json.dumps({
            "issues": [
                {"category": "contradiction", "page": "concepts/lora.md",
                 "description": "Contradicts transformer.md on rank selection."},
            ],
            "suggestions": ["Add cross-reference between LoRA and Transformer."],
        })
        mock_client.messages.create.return_value = _make_response(lint_json)

        pages = [
            WikiPage(
                frontmatter=WikiPageFrontmatter(title="LoRA"),
                body="Content about LoRA.",
                path="concepts/lora.md",
            ),
        ]
        result = agent.lint_pages(pages)

        assert isinstance(result, LintReport)
        assert len(result.issues) == 1
        assert result.issues[0].category == "contradiction"
        assert len(result.suggestions) == 1

    def test_empty_pages(self, agent, mock_client):
        result = agent.lint_pages([])
        assert result.issue_count == 0
        mock_client.messages.create.assert_not_called()

    def test_handles_malformed_lint_json(self, agent, mock_client):
        mock_client.messages.create.return_value = _make_response("garbled output")
        pages = [WikiPage(frontmatter=WikiPageFrontmatter(title="X"), body="Y")]
        result = agent.lint_pages(pages)
        assert result.issue_count == 0  # graceful fallback


class TestAnthropicAgentIntegration:
    """Integration test: agent + WikiManager (all via mock client)."""

    def test_full_ingest_cycle(self, agent, mock_client, tmp_path):
        """Wire agent into WikiManager and run a complete ingest."""
        # Set up responses for the full ingest flow
        summary_response = (
            "## Key Points\n\n- Main idea\n\n"
            "## Relevant Concepts\n\n- [[Attention]]\n\n"
            "## Source Metadata\n\n- Type: paper\n"
        )
        entities_response = json.dumps([
            {"name": "Transformer", "entity_type": "entity", "confidence": 0.95,
             "description": "Neural network architecture"},
        ])
        page_response = (
            "# Transformer\n\n"
            "## Overview\n\nA neural network architecture.\n\n"
            "## Key Contributions\n\nSelf-attention mechanism.\n"
        )
        mock_client.messages.create.side_effect = [
            _make_response(summary_response),
            _make_response(entities_response),
            _make_response(page_response),
        ]

        # Set up directory
        base = tmp_path / "project"
        raw = base / "raw"
        raw.mkdir(parents=True)
        (raw / "paper.md").write_text(
            "# Attention Is All You Need\n\nWe propose the Transformer.\n",
            encoding="utf-8",
        )

        from llm_patch.wiki.manager import WikiManager
        manager = WikiManager(agent=agent, base_dir=base)
        manager.init()

        result = manager.ingest(raw / "paper.md")

        assert result.summary_page != ""
        assert len(result.pages_created) >= 1
        assert len(result.entities_extracted) >= 1
        assert mock_client.messages.create.call_count == 3
