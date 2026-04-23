"""Tests for llm_patch.sources.jsonl — JsonlDataSource."""

from __future__ import annotations

import json

from llm_patch.sources.jsonl import JsonlDataSource


class TestJsonlDataSource:
    """Tests for JsonlDataSource."""

    def test_name_property(self, tmp_path):
        src = JsonlDataSource(tmp_path / "data.jsonl")
        assert src.name == "jsonl"

    def test_fetch_all_valid_records(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"id": "doc1", "text": "Hello world"})
            + "\n"
            + json.dumps({"id": "doc2", "text": "Second doc"})
            + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path)
        docs = list(src.fetch_all())

        assert len(docs) == 2
        assert docs[0].document_id == "doc1"
        assert docs[0].content == "Hello world"
        assert docs[1].document_id == "doc2"
        assert docs[1].content == "Second doc"

    def test_fetch_all_skips_blank_lines(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"id": "a", "text": "first"})
            + "\n"
            + "\n"  # blank
            + "   \n"  # whitespace-only
            + json.dumps({"id": "b", "text": "second"})
            + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path)
        docs = list(src.fetch_all())
        assert len(docs) == 2

    def test_fetch_all_skips_invalid_json(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"id": "valid", "text": "ok"})
            + "\n"
            + "NOT VALID JSON\n"
            + json.dumps({"id": "valid2", "text": "ok2"})
            + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path)
        docs = list(src.fetch_all())
        assert len(docs) == 2
        assert docs[0].document_id == "valid"
        assert docs[1].document_id == "valid2"

    def test_fetch_all_skips_missing_text_field(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"id": "a", "text": "has text"})
            + "\n"
            + json.dumps({"id": "b", "other": "no text field"})
            + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path)
        docs = list(src.fetch_all())
        assert len(docs) == 1
        assert docs[0].document_id == "a"

    def test_falls_back_to_line_index_for_id(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"text": "no id field"}) + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path)
        docs = list(src.fetch_all())
        assert len(docs) == 1
        assert docs[0].document_id == "0"  # line index

    def test_custom_field_names(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"body": "custom text", "uid": "x1", "extra": "val"}) + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path, text_field="body", id_field="uid")
        docs = list(src.fetch_all())
        assert len(docs) == 1
        assert docs[0].document_id == "x1"
        assert docs[0].content == "custom text"
        assert docs[0].metadata["extra"] == "val"

    def test_extra_keys_stored_in_metadata(self, tmp_path):
        path = tmp_path / "data.jsonl"
        path.write_text(
            json.dumps({"id": "d", "text": "t", "author": "me", "year": 2024}) + "\n",
            encoding="utf-8",
        )
        src = JsonlDataSource(path)
        docs = list(src.fetch_all())
        assert docs[0].metadata["author"] == "me"
        assert docs[0].metadata["year"] == 2024
        assert docs[0].metadata["source_path"] == str(path)
        assert docs[0].metadata["line_number"] == 0

    def test_nonexistent_file_returns_empty(self, tmp_path):
        src = JsonlDataSource(tmp_path / "missing.jsonl")
        docs = list(src.fetch_all())
        assert docs == []
