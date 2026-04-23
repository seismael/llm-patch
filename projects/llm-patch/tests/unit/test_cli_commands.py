"""Tests for llm_patch.cli subcommands — source, model, serve."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    from click.testing import CliRunner
except ImportError:
    pytest.skip("click not installed", allow_module_level=True)


# ── Source CLI ────────────────────────────────────────────────────────────


class TestSourceCLI:
    @pytest.fixture()
    def runner(self):
        return CliRunner()

    def test_source_list_markdown(self, runner, tmp_path):
        from llm_patch.cli.source import source

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\nContent", encoding="utf-8")
        (docs / "b.md").write_text("# B\nOther", encoding="utf-8")

        result = runner.invoke(source, ["list", "--kind", "markdown", "--path", str(docs)])
        assert result.exit_code == 0
        assert "2 document(s) found" in result.output

    def test_source_count_markdown(self, runner, tmp_path):
        from llm_patch.cli.source import source

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("# A\nContent", encoding="utf-8")

        result = runner.invoke(source, ["count", "--kind", "markdown", "--path", str(docs)])
        assert result.exit_code == 0
        assert "1 document(s)" in result.output

    def test_source_preview_markdown(self, runner, tmp_path):
        from llm_patch.cli.source import source

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "hello.md").write_text("# Hello\nWorld", encoding="utf-8")

        result = runner.invoke(
            source,
            [
                "preview",
                "--kind",
                "markdown",
                "--path",
                str(docs),
                "hello",
            ],
        )
        assert result.exit_code == 0
        assert "Hello" in result.output

    def test_source_preview_not_found(self, runner, tmp_path):
        from llm_patch.cli.source import source

        docs = tmp_path / "docs"
        docs.mkdir()

        result = runner.invoke(
            source,
            [
                "preview",
                "--kind",
                "markdown",
                "--path",
                str(docs),
                "missing",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_source_list_jsonl(self, runner, tmp_path):
        import json

        from llm_patch.cli.source import source

        f = tmp_path / "data.jsonl"
        f.write_text(
            json.dumps({"id": "a", "text": "hello"}) + "\n",
            encoding="utf-8",
        )

        result = runner.invoke(source, ["list", "--kind", "jsonl", "--path", str(f)])
        assert result.exit_code == 0
        assert "1 document(s) found" in result.output

    def test_source_unknown_kind(self, runner, tmp_path):
        from llm_patch.cli.source import source

        result = runner.invoke(source, ["list", "--kind", "unknown", "--path", str(tmp_path)])
        assert result.exit_code != 0


# ── Model CLI ──────────────────────────────────────────────────────────────


class TestModelCLI:
    @pytest.fixture()
    def runner(self):
        return CliRunner()

    def test_model_info(self, runner, tmp_path):
        from llm_patch.cli.model import model

        # Create an adapter dir with nothing
        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()

        result = runner.invoke(model, ["info", "--adapter-dir", str(adapter_dir)])
        assert result.exit_code == 0
        assert "No adapters found" in result.output


# ── Serve CLI ──────────────────────────────────────────────────────────────


class TestServeCLI:
    @pytest.fixture()
    def runner(self):
        return CliRunner()

    def test_serve_calls_uvicorn(self, runner, tmp_path):
        from llm_patch.cli.serve import serve

        mock_uvicorn = MagicMock()

        adapter_dir = tmp_path / "adapters"
        adapter_dir.mkdir()

        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            result = runner.invoke(
                serve,
                [
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "9000",
                    "--adapter-dir",
                    str(adapter_dir),
                ],
            )
        assert result.exit_code == 0
        mock_uvicorn.run.assert_called_once_with(
            "llm_patch.server.app:app",
            host="0.0.0.0",
            port=9000,
            reload=False,
        )

    def test_serve_missing_uvicorn(self, runner):
        from llm_patch.cli.serve import serve

        with patch.dict("sys.modules", {"uvicorn": None}):
            result = runner.invoke(serve)
            # Should fail gracefully when uvicorn is not importable
            assert result.exit_code != 0
