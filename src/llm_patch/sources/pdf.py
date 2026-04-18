"""PDF directory data source.

Reads PDF files from a directory.  Each PDF becomes one ``DocumentContext``
with page count stored in metadata.

Requires the ``pypdf`` package (install via ``pip install 'llm-patch[pdf]'``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from llm_patch.core.interfaces import IDataSource
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)


def _read_pdf(file_path: Path, base_dir: Path) -> DocumentContext:
    """Extract text from a PDF and return a DocumentContext."""
    from pypdf import PdfReader  # lazy import — optional dep

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    relative = file_path.relative_to(base_dir)
    document_id = relative.with_suffix("").as_posix()

    return DocumentContext(
        document_id=document_id,
        content="\n\n".join(pages),
        metadata={
            "source_path": str(file_path),
            "page_count": len(reader.pages),
        },
    )


class PdfDataSource(IDataSource):
    """Reads all PDF files from a directory.

    Args:
        directory: Root directory to scan.
        recursive: Recurse into subdirectories.
    """

    def __init__(self, directory: Path, *, recursive: bool = True) -> None:
        self._base_dir = Path(directory)
        self._recursive = recursive

    @property
    def name(self) -> str:
        return "pdf"

    def fetch_all(self) -> Iterable[DocumentContext]:
        if not self._base_dir.exists():
            return

        glob_pattern = "**/*.pdf" if self._recursive else "*.pdf"
        for file_path in sorted(self._base_dir.glob(glob_pattern)):
            if file_path.is_file():
                try:
                    yield _read_pdf(file_path, self._base_dir)
                except Exception:
                    logger.exception("Failed to read PDF: %s", file_path)
