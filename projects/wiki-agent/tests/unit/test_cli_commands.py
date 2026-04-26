"""Tests for the new CLI subcommands (daemon, serve, mcp)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner
from pytest import MonkeyPatch

from llm_patch_wiki_agent.cli import main
from llm_patch_wiki_agent.daemon.runner import DaemonResult


class _FakeDaemon:
    instances: list["_FakeDaemon"] = []

    def __init__(self) -> None:
        _FakeDaemon.instances.append(self)
        self.started = False
        self.stopped = False

    def run_once(self) -> DaemonResult:
        import llm_patch as engine

        from llm_patch_wiki_agent.registry import AdapterMetadata

        manifest = engine.AdapterManifest(
            adapter_id="api-v2-auth",
            rank=8,
            target_modules=["q_proj"],
            storage_uri="adapters/api-v2-auth",
        )
        metadata = AdapterMetadata(adapter_id="api-v2-auth", context_id="api-v2-auth")
        return DaemonResult(manifests=(manifest,), metadata=(metadata,))

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True


def test_daemon_command_once_prints_summary(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()

    _FakeDaemon.instances.clear()

    def _fake_from_config(*_args: Any, **_kwargs: Any) -> _FakeDaemon:
        return _FakeDaemon()

    monkeypatch.setattr(
        "llm_patch_wiki_agent.daemon.WikiCompileDaemon.from_config",
        staticmethod(_fake_from_config),
    )

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "daemon",
            "--wiki-dir",
            str(wiki_dir),
            "--adapter-dir",
            str(tmp_path / "adapters"),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Compiled 1 adapter(s)." in result.output
    assert "api-v2-auth" in result.output
    assert "context_id=api-v2-auth" in result.output


def test_daemon_command_help_lists_watch_flag() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["daemon", "--help"])
    assert result.exit_code == 0
    assert "--watch" in result.output
    assert "--once" in result.output


def test_serve_command_help_advertises_options() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "--adapter-dir" in result.output
    assert "--model-id" in result.output


def test_mcp_command_help_lists_transports() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["mcp", "--help"])
    assert result.exit_code == 0
    assert "stdio" in result.output
    assert "sse" in result.output
