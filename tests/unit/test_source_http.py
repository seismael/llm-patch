"""Tests for llm_patch.sources.http_api — HttpApiDataSource."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from llm_patch.sources.http_api import HttpApiDataSource, _extract


class TestExtract:
    """Tests for the _extract helper."""

    def test_simple_key(self):
        assert _extract({"text": "hello"}, "text") == "hello"

    def test_nested_key(self):
        assert _extract({"data": {"body": "nested"}}, "data.body") == "nested"

    def test_missing_key(self):
        assert _extract({"a": 1}, "b") is None

    def test_nested_missing_key(self):
        assert _extract({"a": {"b": 1}}, "a.c") is None

    def test_non_dict_intermediate(self):
        assert _extract({"a": "string"}, "a.b") is None


def _make_httpx_mock(json_data=None, *, error=None):
    """Create a mock httpx module with a get() that returns json_data."""
    mock_httpx = MagicMock()
    if error:
        mock_httpx.get.side_effect = error
    else:
        resp = MagicMock()
        resp.json.return_value = json_data
        resp.raise_for_status = MagicMock()
        mock_httpx.get.return_value = resp
    return mock_httpx


class TestHttpApiDataSource:
    """Tests for HttpApiDataSource (httpx mocked via sys.modules)."""

    def test_name_property(self):
        src = HttpApiDataSource("http://example.com/api")
        assert src.name == "http"

    def test_fetch_all_json_array(self):
        mock_httpx = _make_httpx_mock([
            {"id": "d1", "text": "Hello"},
            {"id": "d2", "text": "World"},
        ])
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource("http://example.com/api")
            docs = list(src.fetch_all())

        assert len(docs) == 2
        assert docs[0].document_id == "d1"
        assert docs[0].content == "Hello"
        assert docs[1].document_id == "d2"
        mock_httpx.get.assert_called_once_with(
            "http://example.com/api", headers={}, timeout=30
        )

    def test_fetch_all_single_object(self):
        mock_httpx = _make_httpx_mock({"id": "single", "text": "only one"})
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource("http://example.com/api")
            docs = list(src.fetch_all())
        assert len(docs) == 1
        assert docs[0].document_id == "single"

    def test_fetch_all_nested_paths(self):
        mock_httpx = _make_httpx_mock([
            {"data": {"body": "nested text"}, "meta": {"uid": "n1"}},
        ])
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource(
                "http://example.com/api",
                text_path="data.body",
                id_path="meta.uid",
            )
            docs = list(src.fetch_all())
        assert len(docs) == 1
        assert docs[0].document_id == "n1"
        assert docs[0].content == "nested text"

    def test_fetch_all_http_error(self):
        mock_httpx = _make_httpx_mock(error=Exception("Connection refused"))
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource("http://example.com/api")
            docs = list(src.fetch_all())
        assert docs == []

    def test_fetch_all_skips_non_dict_records(self):
        mock_httpx = _make_httpx_mock([
            {"id": "ok", "text": "valid"},
            "not a dict",
            42,
        ])
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource("http://example.com/api")
            docs = list(src.fetch_all())
        assert len(docs) == 1

    def test_fetch_all_missing_text_path(self):
        mock_httpx = _make_httpx_mock([{"id": "x", "other": "no text"}])
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource("http://example.com/api")
            docs = list(src.fetch_all())
        assert len(docs) == 0

    def test_fetch_all_custom_headers(self):
        mock_httpx = _make_httpx_mock([])
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource(
                "http://example.com/api",
                headers={"Authorization": "Bearer tok"},
            )
            list(src.fetch_all())
        mock_httpx.get.assert_called_once_with(
            "http://example.com/api",
            headers={"Authorization": "Bearer tok"},
            timeout=30,
        )

    def test_source_url_in_metadata(self):
        mock_httpx = _make_httpx_mock([{"id": "a", "text": "hello"}])
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            src = HttpApiDataSource("http://example.com/api")
            docs = list(src.fetch_all())
        assert docs[0].metadata["source_url"] == "http://example.com/api"
