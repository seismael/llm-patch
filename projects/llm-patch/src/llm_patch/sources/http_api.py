"""HTTP API data source.

Fetches documents from a REST endpoint.  The response is expected to be
a JSON array of objects; configurable ``text_path`` and ``id_path``
extract text and ID from each record.

Requires the ``httpx`` package (install via ``pip install 'llm-patch[http]'``).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from llm_patch.core.interfaces import IDataSource
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)


def _extract(record: dict[str, Any], path: str) -> Any:
    """Extract a value via a dot-delimited path (e.g. ``'data.text'``)."""
    obj: Any = record
    for key in path.split("."):
        if isinstance(obj, dict):
            obj = obj.get(key)
        else:
            return None
    return obj


class HttpApiDataSource(IDataSource):
    """Fetches documents from an HTTP JSON API.

    Args:
        url: Endpoint URL returning a JSON array of records.
        headers: Additional HTTP headers (e.g. Authorization).
        text_path: Dot-path into each record for the text field.
        id_path: Dot-path into each record for the document ID field.
    """

    def __init__(
        self,
        url: str,
        *,
        headers: dict[str, str] | None = None,
        text_path: str = "text",
        id_path: str = "id",
    ) -> None:
        self._url = url
        self._headers = headers or {}
        self._text_path = text_path
        self._id_path = id_path

    @property
    def name(self) -> str:
        return "http"

    def fetch_all(self) -> Iterable[DocumentContext]:
        import httpx  # lazy import — optional dep

        try:
            resp = httpx.get(self._url, headers=self._headers, timeout=30)
            resp.raise_for_status()
        except Exception:
            logger.exception("HTTP request to %s failed", self._url)
            return

        data = resp.json()
        if not isinstance(data, list):
            data = [data]

        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                continue

            text = _extract(record, self._text_path)
            if text is None:
                logger.warning("Missing text at path '%s' in record %d", self._text_path, idx)
                continue

            doc_id = str(_extract(record, self._id_path) or idx)
            metadata = {
                k: v
                for k, v in record.items()
                if k not in (self._text_path.split(".")[0], self._id_path.split(".")[0])
            }
            metadata["source_url"] = self._url

            yield DocumentContext(
                document_id=doc_id,
                content=str(text),
                metadata=metadata,
            )
