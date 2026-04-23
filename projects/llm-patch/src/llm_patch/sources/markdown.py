"""Markdown directory data source and watcher.

Provides:
- ``MarkdownDataSource`` — pull-based ``IDataSource`` for batch ingestion.
- ``MarkdownWatcher`` — push-based ``IKnowledgeStream`` for live monitoring.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from llm_patch.core.config import WatcherConfig
from llm_patch.core.interfaces import IDataSource, IKnowledgeStream
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)


# ── Shared helpers ────────────────────────────────────────────────────


def _matches_patterns(filename: str, patterns: list[str]) -> bool:
    """Check if a filename matches any of the glob patterns."""
    from fnmatch import fnmatch

    return any(fnmatch(filename, pat) for pat in patterns)


def _derive_document_id(file_path: Path, base_dir: Path) -> str:
    """Derive a document ID from a file path relative to the base directory."""
    relative = file_path.relative_to(base_dir)
    return relative.with_suffix("").as_posix()


def _read_document(file_path: Path, base_dir: Path) -> DocumentContext:
    """Read a file and construct a DocumentContext."""
    content = file_path.read_text(encoding="utf-8")
    document_id = _derive_document_id(file_path, base_dir)
    return DocumentContext(
        document_id=document_id,
        content=content,
        metadata={
            "source_path": str(file_path),
            "modified_time": file_path.stat().st_mtime,
        },
    )


# ── Pull-based data source ───────────────────────────────────────────


class MarkdownDataSource(IDataSource):
    """Reads all matching markdown files from a directory (batch, pull-based).

    Args:
        directory: Root directory to scan.
        patterns: Glob patterns for file selection (default ``['*.md']``).
        recursive: Recurse into subdirectories.
    """

    def __init__(
        self,
        directory: Path,
        *,
        patterns: list[str] | None = None,
        recursive: bool = True,
    ) -> None:
        self._base_dir = Path(directory)
        self._patterns = patterns or ["*.md"]
        self._recursive = recursive

    @property
    def name(self) -> str:
        return "markdown"

    def fetch_all(self) -> Iterable[DocumentContext]:
        if not self._base_dir.exists():
            return

        glob_pattern = "**/*" if self._recursive else "*"
        for file_path in sorted(self._base_dir.glob(glob_pattern)):
            if file_path.is_file() and _matches_patterns(file_path.name, self._patterns):
                try:
                    yield _read_document(file_path, self._base_dir)
                except Exception:
                    logger.exception("Failed to read: %s", file_path)

    def fetch_one(self, document_id: str) -> DocumentContext | None:
        candidate = self._base_dir / f"{document_id}.md"
        if candidate.is_file():
            try:
                return _read_document(candidate, self._base_dir)
            except Exception:
                logger.exception("Failed to read: %s", candidate)
        return super().fetch_one(document_id)


# ── Push-based watcher (live monitoring) ──────────────────────────────


class _MarkdownEventHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Internal watchdog handler that invokes callbacks on matching file changes."""

    def __init__(
        self,
        base_dir: Path,
        patterns: list[str],
        debounce_seconds: float,
        callbacks: list[Callable[[DocumentContext], None]],
    ) -> None:
        super().__init__()
        self._base_dir = base_dir
        self._patterns = patterns
        self._debounce_seconds = debounce_seconds
        self._callbacks = callbacks
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
            context = _read_document(file_path, self._base_dir)
        except Exception:
            logger.exception("Failed to read document: %s", src_path)
            return

        for callback in self._callbacks:
            try:
                callback(context)
            except Exception:
                logger.exception("Callback error for document %s", context.document_id)

    def on_created(self, event: FileCreatedEvent) -> None:
        self._handle_event(event.src_path)

    def on_modified(self, event: FileModifiedEvent) -> None:
        self._handle_event(event.src_path)


class MarkdownWatcher(IKnowledgeStream):
    """Watches a directory for markdown file changes and notifies subscribers.

    Uses the watchdog library for efficient filesystem monitoring with
    debouncing to prevent duplicate callbacks on rapid save events.
    """

    def __init__(
        self,
        directory: Path,
        *,
        patterns: list[str] | None = None,
        recursive: bool = True,
        debounce_seconds: float = 0.5,
    ) -> None:
        self._base_dir = Path(directory)
        self._patterns = patterns or ["*.md"]
        self._recursive = recursive
        self._debounce_seconds = debounce_seconds
        self._callbacks: list[Callable[[DocumentContext], None]] = []
        self._observer: Observer | None = None

    def subscribe(self, callback: Callable[[DocumentContext], None]) -> None:
        self._callbacks.append(callback)

    def start(self) -> None:
        if self._observer is not None:
            return

        handler = _MarkdownEventHandler(
            base_dir=self._base_dir,
            patterns=self._patterns,
            debounce_seconds=self._debounce_seconds,
            callbacks=self._callbacks,
        )

        self._observer = Observer()
        self._observer.schedule(handler, str(self._base_dir), recursive=self._recursive)
        self._observer.start()
        logger.info("Started watching %s", self._base_dir)

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching %s", self._base_dir)

    def __enter__(self) -> MarkdownWatcher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


# ── Backward-compat alias (used by existing tests / imports) ──────────


class MarkdownDirectoryWatcher(MarkdownDataSource, MarkdownWatcher):
    """Legacy shim combining data-source + watcher into one object.

    New code should use ``MarkdownDataSource`` (batch) and/or
    ``MarkdownWatcher`` (live) independently.
    """

    def __init__(self, config: WatcherConfig) -> None:

        MarkdownDataSource.__init__(
            self,
            directory=config.directory,
            patterns=config.patterns,
            recursive=config.recursive,
        )
        MarkdownWatcher.__init__(
            self,
            directory=config.directory,
            patterns=config.patterns,
            recursive=config.recursive,
            debounce_seconds=config.debounce_seconds,
        )
        self._config = config

    # Legacy API surface expected by old tests
    def register_callback(self, callback: Callable[[DocumentContext], None]) -> None:
        self.subscribe(callback)

    def scan_existing(self) -> list[DocumentContext]:
        return list(self.fetch_all())
