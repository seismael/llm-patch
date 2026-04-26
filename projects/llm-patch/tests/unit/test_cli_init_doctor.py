"""Tests for ``llm-patch init`` and ``llm-patch doctor``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from click.testing import CliRunner

from llm_patch.cli.doctor import doctor
from llm_patch.cli.init import init
from llm_patch.cli.version import version


def test_init_writes_toml_non_interactive(tmp_path: Path) -> None:
    runner = CliRunner()
    cfg = tmp_path / ".llm-patch.toml"
    result = runner.invoke(
        init,
        [
            "--path",
            str(cfg),
            "--name",
            "demo",
            "--source",
            "./src",
            "--output",
            "./out",
            "--base-model",
            "google/gemma-2-2b-it",
            "--non-interactive",
        ],
    )
    assert result.exit_code == 0, result.output
    assert cfg.exists()
    text = cfg.read_text(encoding="utf-8")
    assert 'name = "demo"' in text
    assert 'source = "./src"' in text
    assert 'output = "./out"' in text
    assert "google/gemma-2-2b-it" in text


def test_init_refuses_overwrite_without_force(tmp_path: Path) -> None:
    cfg = tmp_path / ".llm-patch.toml"
    cfg.write_text("existing\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        init, ["--path", str(cfg), "--non-interactive", "--name", "demo"]
    )
    assert result.exit_code == 1
    assert "Refusing to overwrite" in result.output


def test_init_overwrites_with_force(tmp_path: Path) -> None:
    cfg = tmp_path / ".llm-patch.toml"
    cfg.write_text("existing\n", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(
        init,
        ["--path", str(cfg), "--non-interactive", "--name", "demo", "--force"],
    )
    assert result.exit_code == 0
    assert "existing" not in cfg.read_text(encoding="utf-8")


def test_doctor_text_output_runs() -> None:
    result = CliRunner().invoke(doctor, [])
    assert result.exit_code == 0
    assert "Python" in result.output
    assert "Optional extras" in result.output


def test_doctor_json_output_is_valid_json() -> None:
    result = CliRunner().invoke(doctor, ["--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "python" in payload
    assert "torch" in payload
    assert "registry_plugin" in payload
    assert isinstance(payload["extras"], list)


def test_doctor_quiet_one_line() -> None:
    result = CliRunner().invoke(doctor, ["--quiet"])
    assert result.exit_code == 0
    assert result.output.startswith("OK:")
    assert result.output.count("\n") <= 1


def test_version_text() -> None:
    result = CliRunner().invoke(version, [])
    assert result.exit_code == 0
    assert "llm-patch" in result.output
    assert sys.version.split()[0] in result.output


def test_version_json() -> None:
    result = CliRunner().invoke(version, ["--json"])
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "llm_patch" in payload
    assert "python" in payload
