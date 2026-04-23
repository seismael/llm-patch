"""Smoke tests for ``llm_patch_wiki_agent`` package and CLI scaffolding."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

import llm_patch_wiki_agent
from llm_patch_wiki_agent import WikiAgent, WikiAgentConfig, WikiAgentInfo
from llm_patch_wiki_agent.cli import main


def test_version_exposed() -> None:
    assert isinstance(llm_patch_wiki_agent.__version__, str)


def test_agent_construction(tmp_path: Path) -> None:
    cfg = WikiAgentConfig(
        adapter_dir=tmp_path / "adapters",
        wiki_dir=tmp_path / "wiki",
        model_id="google/gemma-2-2b-it",
    )
    agent = WikiAgent(cfg)
    assert agent.config is cfg


def test_top_level_info_type_exposed(tmp_path: Path) -> None:
    info = WikiAgentInfo(adapter_dir=tmp_path / "adapters", adapter_count=0, adapter_ids=())
    assert info.adapter_count == 0


def test_cli_info(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["info", "--adapter-dir", str(tmp_path / "adapters")])
    assert result.exit_code == 0
    assert "llm-patch-wiki-agent" in result.output


def test_cli_version() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
