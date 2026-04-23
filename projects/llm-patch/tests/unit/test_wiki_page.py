"""Tests for llm_patch.wiki.page — WikiPage model and parsing."""

from __future__ import annotations

from llm_patch.wiki.page import (
    ConfidenceLevel,
    PageType,
    WikiPage,
    WikiPageFrontmatter,
    extract_wikilinks,
    parse_frontmatter,
    parse_wiki_page,
    slugify,
)


class TestParseFrontmatter:
    def test_frontmatter_basic(self) -> None:
        text = '---\ntitle: "Test"\ntype: concept\n---\nBody content.'
        meta, body = parse_frontmatter(text)
        assert meta["title"] == '"Test"'
        assert meta["type"] == "concept"
        assert body == "Body content."

    def test_no_frontmatter(self) -> None:
        text = "Just body content."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == text

    def test_empty_frontmatter(self) -> None:
        text = "---\n---\nBody."
        meta, body = parse_frontmatter(text)
        assert meta == {}
        assert body == "Body."


class TestParseWikiPage:
    def test_round_trip(self) -> None:
        page = WikiPage(
            frontmatter=WikiPageFrontmatter(
                title="Attention",
                type=PageType.CONCEPT,
                tags=["nlp", "transformers"],
                created="2024-01-01",
                updated="2024-01-02",
                sources=["raw/papers/attention.md"],
                confidence=ConfidenceLevel.HIGH,
            ),
            body="# Attention\n\nA mechanism for...",
            path="concepts/attention.md",
        )
        markdown = page.to_markdown()
        reparsed = parse_wiki_page(markdown, "concepts/attention.md")
        assert reparsed.title == "Attention"
        assert reparsed.page_type == PageType.CONCEPT
        assert reparsed.frontmatter.confidence == ConfidenceLevel.HIGH
        assert "nlp" in reparsed.frontmatter.tags
        assert reparsed.path == "concepts/attention.md"

    def test_parse_minimal(self) -> None:
        text = "# Just a heading\n\nSome text."
        page = parse_wiki_page(text)
        assert page.body == text
        assert page.frontmatter.title == ""

    def test_lists_parsed(self) -> None:
        text = '---\ntitle: "Test"\ntags: [a, b, c]\nsources: ["raw/x.md"]\n---\nBody.'
        page = parse_wiki_page(text)
        assert page.frontmatter.tags == ["a", "b", "c"]
        assert page.frontmatter.sources == ["raw/x.md"]


class TestExtractWikilinks:
    def test_basic_wikilinks(self) -> None:
        text = "See [[Attention]] and [[Transformer]]."
        links = extract_wikilinks(text)
        assert links == ["Attention", "Transformer"]

    def test_piped_wikilinks(self) -> None:
        text = "See [[concepts/attention|Attention Mechanism]]."
        links = extract_wikilinks(text)
        assert links == ["concepts/attention"]

    def test_no_wikilinks(self) -> None:
        assert extract_wikilinks("No links here.") == []


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify("Attention Mechanism") == "attention-mechanism"

    def test_special_chars(self) -> None:
        assert slugify("GPT-3 (Few Shot)") == "gpt-3-few-shot"

    def test_already_slug(self) -> None:
        assert slugify("hello-world") == "hello-world"


class TestWikiPageModel:
    def test_wikilinks_property(self) -> None:
        page = WikiPage(body="See [[Attention]] and [[LoRA]].")
        assert set(page.wikilinks) == {"Attention", "LoRA"}

    def test_title_property(self) -> None:
        page = WikiPage(frontmatter=WikiPageFrontmatter(title="Test"))
        assert page.title == "Test"

    def test_page_type_property(self) -> None:
        page = WikiPage(frontmatter=WikiPageFrontmatter(type=PageType.ENTITY))
        assert page.page_type == PageType.ENTITY
