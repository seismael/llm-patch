"""Unit tests for MarkdownDirectoryWatcher (Observer Pattern)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from llm_patch.core.config import WatcherConfig
from llm_patch.sources.markdown_watcher import (
    MarkdownDirectoryWatcher,
    _derive_document_id,
    _matches_patterns,
    _read_document,
)


class TestHelperFunctions:
    def test_matches_patterns_md(self) -> None:
        assert _matches_patterns("readme.md", ["*.md"]) is True

    def test_matches_patterns_no_match(self) -> None:
        assert _matches_patterns("readme.txt", ["*.md"]) is False

    def test_matches_patterns_multiple(self) -> None:
        assert _matches_patterns("doc.txt", ["*.md", "*.txt"]) is True

    def test_derive_document_id_simple(self, tmp_path: Path) -> None:
        file_path = tmp_path / "api_v2.md"
        assert _derive_document_id(file_path, tmp_path) == "api_v2"

    def test_derive_document_id_nested(self, tmp_path: Path) -> None:
        file_path = tmp_path / "docs" / "api_v2.md"
        assert _derive_document_id(file_path, tmp_path) == "docs/api_v2"

    def test_read_document(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.md"
        file_path.write_text("# Test\nContent here.", encoding="utf-8")

        doc = _read_document(file_path, tmp_path)
        assert doc.document_id == "test"
        assert doc.content == "# Test\nContent here."
        assert "source_path" in doc.metadata


class TestScanExisting:
    def test_returns_all_markdown_files(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        docs = watcher.scan_existing()
        ids = {d.document_id for d in docs}

        assert len(docs) == 3
        assert ids == {"api_v1", "api_v2", "guide"}

    def test_ignores_non_markdown_files(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        docs = watcher.scan_existing()
        ids = {d.document_id for d in docs}

        # notes.txt should NOT be included
        assert "notes" not in ids

    def test_returns_document_content(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        docs = watcher.scan_existing()
        api_v1 = next(d for d in docs if d.document_id == "api_v1")
        assert "API v1" in api_v1.content

    def test_empty_directory(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        config = WatcherConfig(directory=empty)
        watcher = MarkdownDirectoryWatcher(config)

        assert watcher.scan_existing() == []

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        config = WatcherConfig(directory=tmp_path / "missing")
        watcher = MarkdownDirectoryWatcher(config)

        assert watcher.scan_existing() == []


class TestCallbackRegistration:
    def test_register_callback_stores_it(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        callback = MagicMock()
        watcher.register_callback(callback)
        assert callback in watcher._callbacks

    def test_multiple_callbacks(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        cb1 = MagicMock()
        cb2 = MagicMock()
        watcher.register_callback(cb1)
        watcher.register_callback(cb2)
        assert len(watcher._callbacks) == 2


class TestWatcherLifecycle:
    def test_start_and_stop(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        watcher.start()
        assert watcher._observer is not None
        watcher.stop()
        assert watcher._observer is None

    def test_double_start_is_idempotent(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        watcher.start()
        observer1 = watcher._observer
        watcher.start()  # Should not create a second observer
        assert watcher._observer is observer1
        watcher.stop()

    def test_context_manager(self, markdown_dir: Path) -> None:
        config = WatcherConfig(directory=markdown_dir)
        watcher = MarkdownDirectoryWatcher(config)

        with watcher:
            assert watcher._observer is not None
        assert watcher._observer is None
