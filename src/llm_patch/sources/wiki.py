"""Wiki knowledge source — structured markdown with frontmatter and wikilinks.

Provides:
- ``WikiDataSource`` — pull-based ``IDataSource`` for batch wiki ingestion.
- ``WikiWatcher`` — push-based ``IKnowledgeStream`` for live wiki monitoring.
- ``WikiDocumentAggregator`` — follows ``[[wikilinks]]`` to enrich documents.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from llm_patch.core.interfaces import IDataSource, IKnowledgeStream
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)---\s*\n", re.DOTALL)
_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md)\)")
_YAML_LIST_RE = re.compile(r"^\s*\[(.*)\]\s*$")


def _parse_yaml_value(raw: str) -> str | list[str]:
    """Parse a YAML value, handling inline lists like ``[a, b, c]``."""
    m = _YAML_LIST_RE.match(raw)
    if m:
        return [item.strip().strip('"').strip("'") for item in m.group(1).split(",") if item.strip()]
    return raw


def _parse_frontmatter(text: str) -> tuple[dict[str, str | list[str]], str]:
    """Split YAML frontmatter from body content.

    Uses simple ``key: value`` parsing to avoid a PyYAML dependency.
    Handles inline YAML lists like ``tags: [grpc, protobuf]``.

    Returns:
        A tuple of (metadata dict, body text without frontmatter).
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    meta: dict[str, str | list[str]] = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip()] = _parse_yaml_value(value.strip())

    body = text[match.end() :]
    return meta, body


def _extract_wikilinks(text: str) -> list[str]:
    """Extract link targets from ``[[wikilinks]]`` and ``[text](path.md)`` links."""
    targets: list[str] = []
    # Classic [[wikilinks]]
    targets.extend(_WIKILINK_RE.findall(text))
    # Standard markdown links to .md files
    for _label, href in _MD_LINK_RE.findall(text):
        targets.append(href)
    return targets


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
        """Try to find a markdown file matching the link target.

        Handles both wikilink names ("Self Attention") and relative
        paths ("patterns/cqrs.md").
        """
        # If it looks like a relative path, resolve directly
        if link.endswith(".md"):
            candidate = (self._wiki_dir / link).resolve()
            if candidate.is_file():
                return candidate

        slug = link.strip().replace(" ", "-").lower()
        # Search in common wiki subdirectories
        for subdir in ("entities", "concepts", "sources", "patterns", "pillars", "decisions", "summaries"):
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


class WikiDataSource(IDataSource):
    """Reads all structured wiki pages from a directory (batch, pull-based).

    Parses YAML frontmatter, extracts ``[[wikilinks]]``, and optionally
    aggregates linked pages into enriched ``DocumentContext`` objects.

    Args:
        directory: Root wiki directory to scan.
        patterns: Glob patterns for file selection (default ``['*.md']``).
        recursive: Recurse into subdirectories.
        aggregate: If ``True``, follow wikilinks and concatenate linked content.
    """

    def __init__(
        self,
        directory: Path,
        *,
        patterns: list[str] | None = None,
        recursive: bool = True,
        aggregate: bool = False,
    ) -> None:
        self._base_dir = Path(directory)
        self._patterns = patterns or ["*.md"]
        self._recursive = recursive
        self._aggregator = WikiDocumentAggregator(self._base_dir) if aggregate else None

    @property
    def name(self) -> str:
        return "wiki"

    def fetch_all(self) -> Iterable[DocumentContext]:
        if not self._base_dir.exists():
            return

        glob_pattern = "**/*" if self._recursive else "*"
        for file_path in sorted(self._base_dir.glob(glob_pattern)):
            if file_path.is_file() and _matches_patterns(file_path.name, self._patterns):
                try:
                    doc = _read_wiki_document(file_path, self._base_dir)
                    if self._aggregator is not None:
                        doc = self._aggregator.aggregate(doc)
                    yield doc
                except Exception:
                    logger.exception("Failed to read wiki page: %s", file_path)


# ── Push-based watcher ────────────────────────────────────────────────


class WikiWatcher(IKnowledgeStream):
    """Watches a wiki directory for live changes and notifies subscribers."""

    def __init__(
        self,
        directory: Path,
        *,
        patterns: list[str] | None = None,
        recursive: bool = True,
        debounce_seconds: float = 0.5,
        aggregate: bool = False,
    ) -> None:
        self._base_dir = Path(directory)
        self._patterns = patterns or ["*.md"]
        self._recursive = recursive
        self._debounce_seconds = debounce_seconds
        self._callbacks: list[Callable[[DocumentContext], None]] = []
        self._aggregator = WikiDocumentAggregator(self._base_dir) if aggregate else None
        self._observer: Observer | None = None

    def subscribe(self, callback: Callable[[DocumentContext], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._observer is not None:
            return

        handler = _WikiEventHandler(
            base_dir=self._base_dir,
            patterns=self._patterns,
            debounce_seconds=self._debounce_seconds,
            callbacks=self._callbacks,
            aggregator=self._aggregator,
        )

        self._observer = Observer()
        self._observer.schedule(handler, str(self._base_dir), recursive=self._recursive)
        self._observer.start()
        logger.info("Started watching wiki at %s", self._base_dir)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching wiki at %s", self._base_dir)

    def __enter__(self) -> WikiWatcher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ── Backward-compat alias ────────────────────────────────────────────


class WikiKnowledgeSource(WikiDataSource, WikiWatcher):  # type: ignore[misc]
    """Legacy shim combining wiki data-source + watcher.

    New code should use ``WikiDataSource`` (batch) and/or
    ``WikiWatcher`` (live) independently.
    """

    def __init__(self, config: "WatcherConfig", *, aggregate: bool = False) -> None:  # noqa: F821
        from llm_patch.core.config import WatcherConfig as _WC  # noqa: F811

        WikiDataSource.__init__(
            self,
            directory=config.directory,
            patterns=config.patterns,
            recursive=config.recursive,
            aggregate=aggregate,
        )
        WikiWatcher.__init__(
            self,
            directory=config.directory,
            patterns=config.patterns,
            recursive=config.recursive,
            debounce_seconds=config.debounce_seconds,
            aggregate=aggregate,
        )
        self._config = config

    def register_callback(self, callback: Callable[[DocumentContext], None]) -> None:
        self.subscribe(callback)

    def scan_existing(self) -> list[DocumentContext]:
        return list(self.fetch_all())
