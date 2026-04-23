"""Tests for the wiki-agent CLI surface."""

from __future__ import annotations

from pathlib import Path

import llm_patch as engine
from click.testing import CliRunner
from pytest import MonkeyPatch

from llm_patch_wiki_agent.agent import WikiAgentInfo
from llm_patch_wiki_agent.cli import main


class FakeCliAgent:
    def __init__(self, config: object) -> None:
        self.config = config

    def compile(self) -> list[engine.AdapterManifest]:
        return [
            engine.AdapterManifest(
                adapter_id="wiki-transformer",
                rank=8,
                target_modules=["q_proj"],
                storage_uri="adapters/wiki-transformer",
            )
        ]

    def chat(self, prompt: str, *, adapter_ids: tuple[str, ...] | None = None) -> str:
        if adapter_ids is not None:
            assert isinstance(adapter_ids, tuple)
        return f"reply: {prompt}"

    def describe(self) -> WikiAgentInfo:
        return WikiAgentInfo(
            adapter_dir=Path("adapters"),
            adapter_count=1,
            adapter_ids=("wiki-transformer",),
            wiki_dir=Path("wiki"),
            model_id="demo/model",
            checkpoint_dir=Path("checkpoint"),
        )


def test_compile_command_prints_manifest_summary(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    wiki_dir = tmp_path / "wiki"
    wiki_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    adapter_dir = tmp_path / "adapters"
    monkeypatch.setattr("llm_patch_wiki_agent.cli.WikiAgent", FakeCliAgent)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "compile",
            "--wiki-dir",
            str(wiki_dir),
            "--adapter-dir",
            str(adapter_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
    )

    assert result.exit_code == 0
    assert "Compiled 1 adapter(s)." in result.output
    assert "wiki-transformer" in result.output


def test_chat_command_prints_reply(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()
    monkeypatch.setattr("llm_patch_wiki_agent.cli.WikiAgent", FakeCliAgent)
    runner = CliRunner()

    result = runner.invoke(
        main,
        [
            "chat",
            "--adapter-dir",
            str(adapter_dir),
            "--model-id",
            "demo/model",
            "Explain transformers",
        ],
    )

    assert result.exit_code == 0
    assert "reply: Explain transformers" in result.output


def test_info_command_prints_runtime_summary(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    adapter_dir = tmp_path / "adapters"
    monkeypatch.setattr("llm_patch_wiki_agent.cli.WikiAgent", FakeCliAgent)
    runner = CliRunner()

    result = runner.invoke(main, ["info", "--adapter-dir", str(adapter_dir)])

    assert result.exit_code == 0
    assert "Adapter dir:" in result.output
    assert "wiki-transformer" in result.output
