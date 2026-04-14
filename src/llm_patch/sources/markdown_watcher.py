"""Markdown directory watcher using watchdog (Observer Pattern)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileCreatedEvent, FileModifiedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from llm_patch.core.config import WatcherConfig
from llm_patch.core.interfaces import IKnowledgeSource
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)


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

        # Debounce: Skip if fired recently for the same path
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


class MarkdownDirectoryWatcher(IKnowledgeSource):
    """Watches a directory for markdown file changes and notifies observers.

    Uses the watchdog library for efficient filesystem monitoring with
    debouncing to prevent duplicate callbacks on rapid save events.
    """

    def __init__(self, config: WatcherConfig) -> None:
        self._config = config
        self._base_dir = Path(config.directory)
        self._callbacks: list[Callable[[DocumentContext], None]] = []
        self._observer: Observer | None = None

    def register_callback(self, callback: Callable[[DocumentContext], None]) -> None:
        """Register a callback for document change notifications."""
        self._callbacks.append(callback)

    def start(self) -> None:
        """Start the filesystem observer thread."""
        if self._observer is not None:
            return

        handler = _MarkdownEventHandler(
            base_dir=self._base_dir,
            patterns=self._config.patterns,
            debounce_seconds=self._config.debounce_seconds,
            callbacks=self._callbacks,
        )

        self._observer = Observer()
        self._observer.schedule(
            handler,
            str(self._base_dir),
            recursive=self._config.recursive,
        )
        self._observer.start()
        logger.info("Started watching %s", self._base_dir)

    def stop(self) -> None:
        """Stop the filesystem observer thread."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Stopped watching %s", self._base_dir)

    def scan_existing(self) -> list[DocumentContext]:
        """Scan for all existing matching documents in the watched directory."""
        documents: list[DocumentContext] = []

        if not self._base_dir.exists():
            return documents

        glob_pattern = "**/*" if self._config.recursive else "*"
        for file_path in sorted(self._base_dir.glob(glob_pattern)):
            if file_path.is_file() and _matches_patterns(file_path.name, self._config.patterns):
                try:
                    doc = _read_document(file_path, self._base_dir)
                    documents.append(doc)
                except Exception:
                    logger.exception("Failed to read: %s", file_path)

        return documents

    def __enter__(self) -> MarkdownDirectoryWatcher:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
