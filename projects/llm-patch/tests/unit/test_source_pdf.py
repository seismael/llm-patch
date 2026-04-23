"""Tests for llm_patch.sources.pdf — PdfDataSource."""

from __future__ import annotations

from unittest.mock import patch

from llm_patch.core.models import DocumentContext


class TestPdfDataSource:
    """Tests for PdfDataSource (pypdf mocked)."""

    def _make_source(self, directory, **kwargs):
        from llm_patch.sources.pdf import PdfDataSource

        return PdfDataSource(directory, **kwargs)

    def test_name_property(self, tmp_path):
        src = self._make_source(tmp_path)
        assert src.name == "pdf"

    def test_fetch_all_empty_dir(self, tmp_path):
        src = self._make_source(tmp_path)
        assert list(src.fetch_all()) == []

    def test_fetch_all_nonexistent_dir(self, tmp_path):
        src = self._make_source(tmp_path / "nope")
        assert list(src.fetch_all()) == []

    @patch("llm_patch.sources.pdf._read_pdf")
    def test_fetch_all_reads_pdfs(self, mock_read, tmp_path):
        (tmp_path / "a.pdf").write_bytes(b"fake")
        (tmp_path / "b.pdf").write_bytes(b"fake")
        (tmp_path / "c.txt").write_text("not pdf")

        mock_read.side_effect = [
            DocumentContext(document_id="a", content="A text"),
            DocumentContext(document_id="b", content="B text"),
        ]

        src = self._make_source(tmp_path)
        docs = list(src.fetch_all())
        assert len(docs) == 2
        assert mock_read.call_count == 2

    @patch("llm_patch.sources.pdf._read_pdf")
    def test_fetch_all_non_recursive(self, mock_read, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.pdf").write_bytes(b"fake")
        (sub / "deep.pdf").write_bytes(b"fake")

        mock_read.return_value = DocumentContext(document_id="top", content="Top")

        src = self._make_source(tmp_path, recursive=False)
        docs = list(src.fetch_all())
        assert len(docs) == 1
        assert mock_read.call_count == 1

    @patch("llm_patch.sources.pdf._read_pdf")
    def test_fetch_all_skips_failed_pdfs(self, mock_read, tmp_path):
        (tmp_path / "good.pdf").write_bytes(b"fake")
        (tmp_path / "bad.pdf").write_bytes(b"fake")

        mock_read.side_effect = [
            DocumentContext(document_id="good", content="ok"),
            Exception("corrupt PDF"),
        ]

        src = self._make_source(tmp_path)
        docs = list(src.fetch_all())
        assert len(docs) == 1
        assert docs[0].document_id == "good"


class TestReadPdf:
    """Tests for the _read_pdf helper."""

    def test_read_pdf_extracts_text(self, tmp_path):
        import sys
        from unittest.mock import MagicMock

        page1 = MagicMock()
        page1.extract_text.return_value = "Page one text"
        page2 = MagicMock()
        page2.extract_text.return_value = "Page two text"

        reader_inst = MagicMock()
        reader_inst.pages = [page1, page2]

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = reader_inst

        file_path = tmp_path / "doc.pdf"
        file_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"pypdf": mock_pypdf}):
            from llm_patch.sources.pdf import _read_pdf

            doc = _read_pdf(file_path, tmp_path)

        assert doc.document_id == "doc"
        assert "Page one text" in doc.content
        assert "Page two text" in doc.content
        assert doc.metadata["page_count"] == 2
        assert doc.metadata["source_path"] == str(file_path)

    def test_read_pdf_handles_none_text(self, tmp_path):
        import sys
        from unittest.mock import MagicMock

        page = MagicMock()
        page.extract_text.return_value = None

        reader_inst = MagicMock()
        reader_inst.pages = [page]

        mock_pypdf = MagicMock()
        mock_pypdf.PdfReader.return_value = reader_inst

        file_path = tmp_path / "empty.pdf"
        file_path.write_bytes(b"fake")

        with patch.dict(sys.modules, {"pypdf": mock_pypdf}):
            from llm_patch.sources.pdf import _read_pdf

            doc = _read_pdf(file_path, tmp_path)

        assert doc.content == ""  # None → ""
