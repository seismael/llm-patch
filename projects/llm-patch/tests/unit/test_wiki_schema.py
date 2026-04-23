"""Tests for llm_patch.wiki.schema — WikiSchema configuration."""

from __future__ import annotations

from pathlib import Path

from llm_patch.wiki.schema import WikiSchema


class TestWikiSchemaDefault:
    def test_default_has_page_types(self) -> None:
        schema = WikiSchema.default()
        assert "summary" in schema.page_types
        assert "concept" in schema.page_types
        assert "entity" in schema.page_types
        assert "synthesis" in schema.page_types
        assert "journal" in schema.page_types

    def test_default_has_rules(self) -> None:
        schema = WikiSchema.default()
        assert len(schema.rules) >= 5

    def test_get_directory_for_type(self) -> None:
        schema = WikiSchema.default()
        assert schema.get_directory_for_type("concept") == "wiki/concepts"
        assert schema.get_directory_for_type("summary") == "wiki/summaries"

    def test_get_directory_for_unknown_type(self) -> None:
        schema = WikiSchema.default()
        assert schema.get_directory_for_type("unknown") == "wiki"

    def test_get_required_sections(self) -> None:
        schema = WikiSchema.default()
        sections = schema.get_required_sections("summary")
        assert "Key Points" in sections

    def test_get_required_sections_unknown(self) -> None:
        schema = WikiSchema.default()
        assert schema.get_required_sections("unknown") == []


class TestWikiSchemaFromFile:
    def test_load_from_markdown(self, tmp_path: Path) -> None:
        schema_file = tmp_path / "schema.md"
        schema_file.write_text(
            "# My Wiki\n\n## Rules\n\n- Rule one\n- Rule two\n",
            encoding="utf-8",
        )
        schema = WikiSchema.from_file(schema_file)
        assert schema.domain == "My Wiki"
        assert "Rule one" in schema.rules
        assert "Rule two" in schema.rules


class TestEnsureDirectories:
    def test_creates_directories(self, tmp_path: Path) -> None:
        schema = WikiSchema.default()
        schema.ensure_directories(tmp_path)

        assert (tmp_path / "raw").is_dir()
        assert (tmp_path / "wiki" / "summaries").is_dir()
        assert (tmp_path / "wiki" / "concepts").is_dir()
        assert (tmp_path / "wiki" / "entities").is_dir()
        assert (tmp_path / "wiki" / "syntheses").is_dir()
        assert (tmp_path / "wiki" / "journal").is_dir()
