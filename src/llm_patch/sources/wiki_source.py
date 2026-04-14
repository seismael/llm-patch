"""Wiki knowledge source for LLM Wiki Agent output (Observer Pattern).

Watches wiki directories (sources/, entities/, concepts/) for structured
markdown pages with YAML frontmatter and [[wikilinks]], producing enriched
DocumentContext instances suitable for LoRA weight generation.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from llm_patch.core.config import WatcherConfig
from llm_patch.core.interfaces import IKnowledgeSource
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split YAML frontmatter from body content.

    Uses simple ``key: value`` parsing to avoid a PyYAML dependency.

    Returns:
        A tuple of (metadata dict, body text without frontmatter).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()

    body = text[match.end() :]
    return meta, body


def _extract_wikilinks(text: str) -> list[str]:
    """Extract all ``[[wikilink]]`` targets from markdown text."""
    return _WIKILINK_RE.findall(text)


def _matches_patterns(filename: str, patterns: list[str]) -> bool:
    """Check if a filename matches any of the glob patterns."""
    from fnmatch import fnmatch

    return any(fnmatch(filename, pat) for pat in patterns)


def _derive_document_id(file_path: Path, base_dir: Path) -> str:
    """Derive a document ID from a file path relative to the base directory."""
    relative = file_path.relative_to(base_dir)
    return relative.with_suffix("").as_posix()


def _read_wiki_document(file_path: Path, base_dir: Path) -> DocumentContext:
    """Read a wiki page and construct a DocumentContext with frontmatter metadata."""
    raw = file_path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw)
    document_id = _derive_document_id(file_path, base_dir)

    metadata: dict[str, object] = {
        "source_path": str(file_path),
        "modified_time": file_path.stat().st_mtime,
        **frontmatter,
    }

    wikilinks = _extract_wikilinks(body)
    if wikilinks:
        metadata["wikilinks"] = wikilinks

    return DocumentContext(
        document_id=document_id,
        content=body,
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# WikiDocumentAggregator
# ---------------------------------------------------------------------------


class WikiDocumentAggregator:
    """Follows ``[[wikilinks]]`` to build an enriched document from linked pages.

    Given a wiki base directory, resolves wikilinks found in a source page,
    reads the linked entity/concept pages, and concatenates their content
    into a single :class:`DocumentContext` for richer weight generation.
    """

    def __init__(self, wiki_dir: Path) -> None:
        self._wiki_dir = wiki_dir

    def aggregate(self, source_doc: DocumentContext) -> DocumentContext:
        """Return an enriched document that appends linked page content.

        If no wikilinks are found or none resolve, returns *source_doc* unchanged.
        """
        wikilinks: list[str] = source_doc.metadata.get("wikilinks", [])
        if not wikilinks:
            return source_doc

        parts: list[str] = [source_doc.content]
        resolved: list[str] = []

        for link in wikilinks:
            linked_path = self._resolve_link(link)
            if linked_path is None:
                continue
            try:
                raw = linked_path.read_text(encoding="utf-8")
                _, body = _parse_frontmatter(raw)
                parts.append(f"\n\n## {link}\n\n{body}")
                resolved.append(link)
            except Exception:
                logger.warning("Could not read linked page: %s", link)

        if not resolved:
            return source_doc

        merged_metadata = dict(source_doc.metadata)
        merged_metadata["resolved_links"] = resolved
        merged_metadata["aggregated"] = True

        return DocumentContext(
            document_id=source_doc.document_id,
            content="\n".join(parts),
            metadata=merged_metadata,
        )

    def _resolve_link(self, link: str) -> Path | None:
        """Try to find a markdown file matching the wikilink target."""
        slug = link.strip().replace(" ", "-").lower()
        # Search in common wiki subdirectories
        for subdir in ("entities", "concepts", "sources"):
            candidate = self._wiki_dir / subdir / f"{slug}.md"
            if candidate.is_file():
                return candidate
        # Fallback: search top-level
        candidate = self._wiki_dir / f"{slug}.md"
        if candidate.is_file():
            return candidate
        return None


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------


class _WikiEventHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Internal watchdog handler for wiki directory changes."""

    def __init__(
        self,
        base_dir: Path,
        patterns: list[str],
        debounce_seconds: float,
        callbacks: list[Callable[[DocumentContext], None]],
        aggregator: WikiDocumentAggregator | None,
    ) -> None:
        super().__init__()
        self._base_dir = base_dir
        self._patterns = patterns
        self._debounce_seconds = debounce_seconds
        self._callbacks = callbacks
        self._aggregator = aggregator
        self._last_fired: dict[str, float] = {}

    def _should_process(self, src_path: str) -> bool:
        file_path = Path(src_path)
        if file_path.is_dir():
            return False
        if not _matches_patterns(file_path.name, self._patterns):
            return False
        now = time.monotonic()
        last = self._last_fired.get(src_path, 0.0)
        if now - last < self._debounce_seconds:
            return False
        self._last_fired[src_path] = now
        return True

    def _handle_event(self, src_path: str) -> None:
        if not self._should_process(src_path):
            return

        file_path = Path(src_path)
        if not file_path.exists():
            return

        try:
            context = _read_wiki_document(file_path, self._base_dir)
        except Exception:
            logger.exception("Failed to read wiki document: %s", src_path)
            return

        if self._aggregator is not None:
            try:
                context = self._aggregator.aggregate(context)
            except Exception:
                logger.exception("Aggregation failed for %s", context.document_id)

        for callback in self._callbacks:
            try:
                callback(context)
            except Exception:
                logger.exception("Callback error for document %s", context.document_id)

    def on_created(self, event: FileCreatedEvent) -> None:
        self._handle_event(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        self._handle_event(event.src_path)


# ---------------------------------------------------------------------------
# WikiKnowledgeSource
# ---------------------------------------------------------------------------


class WikiKnowledgeSource(IKnowledgeSource):
    """Watches an LLM Wiki Agent output directory for structured wiki pages.

    Parses YAML frontmatter, extracts ``[[wikilinks]]``, and optionally
    aggregates linked pages into enriched :class:`DocumentContext` objects.

    Args:
        config: Watcher configuration (directory, patterns, etc.).
        aggregate: If ``True``, follow wikilinks and concatenate linked content.
    """

    def __init__(self, config: WatcherConfig, *, aggregate: bool = False) -> None:
        self._config = config
        self._base_dir = Path(config.directory)
        self._callbacks: list[Callable[[DocumentContext], None]] = []
        self._observer: Observer | None = None
        self._aggregator = WikiDocumentAggregator(self._base_dir) if aggregate else None

    def register_callback(self, callback: Callable[[DocumentContext], None]) -> None:
        """Register a callback for wiki document change notifications."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the filesystem observer thread."""
        if self._observer is not None:
            return

        handler = _WikiEventHandler(
            base_dir=self._base_dir,
            patterns=self._config.patterns,
            debounce_seconds=self._config.debounce_seconds,
            callbacks=self._callbacks,
            aggregator=self._aggregator,
        )

        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(self._base_dir),
            recursive=self._config.recursive,
        )
        self._observer.start()
        logger.info("Started watching wiki at %s", self._base_dir)

    def stop(self) -> None:
        """Stop the filesystem observer thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching wiki at %s", self._base_dir)

    def scan_existing(self) -> list[DocumentContext]:
        """Scan for all existing wiki pages in the watched directory."""
        documents: list[DocumentContext] = []

        if not self._base_dir.exists():
            return documents

        glob_pattern = "**/*" if self._config.recursive else "*"
        for file_path in sorted(self._base_dir.glob(glob_pattern)):
            if file_path.is_file() and _matches_patterns(file_path.name, self._config.patterns):
                try:
                    doc = _read_wiki_document(file_path, self._base_dir)
                    if self._aggregator is not None:
                        doc = self._aggregator.aggregate(doc)
                    documents.append(doc)
                except Exception:
                    logger.exception("Failed to read wiki page: %s", file_path)

        return documents

    def __enter__(self) -> WikiKnowledgeSource:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
