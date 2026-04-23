"""Tests for llm_patch.sources.composite — CompositeDataSource."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_patch.core.models import DocumentContext
from llm_patch.sources.composite import CompositeDataSource


def _mock_source(name: str, docs: list[DocumentContext]) -> MagicMock:
    """Create a mock IDataSource with given name and documents."""
    src = MagicMock()
    src.name = name
    src.fetch_all.return_value = iter(docs)
    src.fetch_one.side_effect = lambda doc_id: next(
        (d for d in docs if d.document_id == doc_id), None
    )
    return src


class TestCompositeDataSource:
    def test_name_property(self):
        src = _mock_source("a", [])
        composite = CompositeDataSource(src)
        assert composite.name == "composite"

    def test_requires_at_least_one_source(self):
        with pytest.raises(ValueError, match="at least one source"):
            CompositeDataSource()

    def test_fetch_all_merges_sources(self):
        s1 = _mock_source(
            "md",
            [
                DocumentContext(document_id="d1", content="c1"),
            ],
        )
        s2 = _mock_source(
            "json",
            [
                DocumentContext(document_id="d2", content="c2"),
            ],
        )
        composite = CompositeDataSource(s1, s2)
        docs = list(composite.fetch_all())
        assert len(docs) == 2

    def test_fetch_all_namespaces_ids(self):
        s1 = _mock_source(
            "md",
            [
                DocumentContext(document_id="d1", content="c1"),
            ],
        )
        composite = CompositeDataSource(s1, namespace_ids=True)
        docs = list(composite.fetch_all())
        assert docs[0].document_id == "md:d1"
        assert docs[0].metadata["original_source"] == "md"

    def test_fetch_all_no_namespace(self):
        s1 = _mock_source(
            "md",
            [
                DocumentContext(document_id="d1", content="c1"),
            ],
        )
        composite = CompositeDataSource(s1, namespace_ids=False)
        docs = list(composite.fetch_all())
        assert docs[0].document_id == "d1"

    def test_fetch_one_with_namespace(self):
        doc = DocumentContext(document_id="d1", content="hello")
        s1 = _mock_source("md", [doc])
        composite = CompositeDataSource(s1, namespace_ids=True)

        result = composite.fetch_one("md:d1")
        assert result is not None
        assert result.document_id == "md:d1"
        assert result.content == "hello"
        assert result.metadata["original_source"] == "md"

    def test_fetch_one_wrong_source_returns_none(self):
        s1 = _mock_source(
            "md",
            [
                DocumentContext(document_id="d1", content="c1"),
            ],
        )
        composite = CompositeDataSource(s1, namespace_ids=True)
        assert composite.fetch_one("json:d1") is None

    def test_fetch_one_no_colon_falls_back_to_super(self):
        """Without namespace prefix, falls back to base IDataSource.fetch_one."""
        s1 = _mock_source(
            "md",
            [
                DocumentContext(document_id="d1", content="c1"),
            ],
        )
        composite = CompositeDataSource(s1, namespace_ids=True)
        # "d1" doesn't have ":", so it falls back to iterating fetch_all
        # which yields "md:d1" — no match for "d1"
        result = composite.fetch_one("d1")
        assert result is None

    def test_multiple_sources_preserves_order(self):
        s1 = _mock_source(
            "a",
            [
                DocumentContext(document_id="x", content="first"),
            ],
        )
        s2 = _mock_source(
            "b",
            [
                DocumentContext(document_id="y", content="second"),
            ],
        )
        composite = CompositeDataSource(s1, s2)
        docs = list(composite.fetch_all())
        assert docs[0].document_id == "a:x"
        assert docs[1].document_id == "b:y"
