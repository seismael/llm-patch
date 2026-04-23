"""JSONL (JSON Lines) data source.

Reads a ``.jsonl`` file where each line is a JSON object.  Configurable
``text_field`` and ``id_field`` determine which keys map to
``DocumentContext.content`` and ``DocumentContext.document_id``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from llm_patch.core.interfaces import IDataSource
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)


class JsonlDataSource(IDataSource):
    """Reads documents from a JSONL file.

    Args:
        path: Path to the ``.jsonl`` file.
        text_field: JSON key used as the document text (default ``'text'``).
        id_field: JSON key used as the document ID (default ``'id'``).
            If the field is missing, the 0-based line index is used.
    """

    def __init__(
        self,
        path: Path,
        *,
        text_field: str = "text",
        id_field: str = "id",
    ) -> None:
        self._path = Path(path)
        self._text_field = text_field
        self._id_field = id_field

    @property
    def name(self) -> str:
        return "jsonl"

    def fetch_all(self) -> Iterable[DocumentContext]:
        if not self._path.is_file():
            return

        with self._path.open(encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping invalid JSON at line %d in %s", idx, self._path)
                    continue

                text = record.get(self._text_field)
                if text is None:
                    logger.warning(
                        "Missing text field '%s' at line %d in %s",
                        self._text_field,
                        idx,
                        self._path,
                    )
                    continue

                doc_id = str(record.get(self._id_field, idx))
                metadata = {
                    k: v for k, v in record.items() if k not in (self._text_field, self._id_field)
                }
                metadata["source_path"] = str(self._path)
                metadata["line_number"] = idx

                yield DocumentContext(
                    document_id=doc_id,
                    content=str(text),
                    metadata=metadata,
                )
