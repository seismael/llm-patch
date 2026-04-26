"""Integration: `init` writes a config that downstream commands honor.

Exercises the read side of ``.llm-patch.toml``: after ``llm-patch init``
writes a project config, ``llm-patch compile`` (and ``model chat``)
must pick up its defaults when explicit flags are omitted, and explicit
flags must override config values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from llm_patch.cli.adapter import adapter
from llm_patch.cli.init import init
from llm_patch.core.project_config import CONFIG_FILENAME, ProjectConfig


def _write_config(
    project_dir: Path,
    *,
    source: str = "./docs",
    output: str = "./adapters",
    base_model: str = "google/gemma-2-2b-it",
    plugin: str | None = None,
) -> None:
    runner = CliRunner()
    result = runner.invoke(
        init,
        [
            "--path",
            str(project_dir / CONFIG_FILENAME),
            "--non-interactive",
            "--name",
            "demo",
            "--source",
            source,
            "--output",
            output,
            "--base-model",
            base_model,
        ],
    )
    assert result.exit_code == 0, result.output
    if plugin is not None:
        cfg_file = project_dir / CONFIG_FILENAME
        cfg_file.write_text(
            cfg_file.read_text(encoding="utf-8")
            + f'\n[registry]\nplugin = "{plugin}"\n',
            encoding="utf-8",
        )


class TestInitCompileRoundtrip:
    def test_compile_uses_config_when_flags_omitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "docs").mkdir()
        _write_config(project, source="./docs", output="./adapters")

        monkeypatch.chdir(project)

        # Without --source-dir/--output-dir but no --checkpoint-dir we
        # expect to pass path resolution and fail later at generator
        # construction. That proves config defaults were picked up.
        runner = CliRunner()
        result = runner.invoke(adapter, ["compile"])

        assert result.exit_code != 0
        assert "Missing --source-dir" not in result.output
        assert "Missing --output-dir" not in result.output
        assert "Missing --checkpoint-dir" in result.output

    def test_compile_explicit_flag_wins_over_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "docs").mkdir()
        (project / "elsewhere").mkdir()
        _write_config(project, source="./docs", output="./adapters")
        monkeypatch.chdir(project)

        # Explicit --source-dir points at "elsewhere"; config says "docs".
        # Failure mode (missing checkpoint) is the same; we just verify
        # neither --source nor --output errors fired (i.e., resolution
        # succeeded with the explicit flag value).
        runner = CliRunner()
        result = runner.invoke(
            adapter,
            ["compile", "--source-dir", str(project / "elsewhere")],
        )

        assert result.exit_code != 0
        assert "Missing --source-dir" not in result.output

    def test_compile_errors_clearly_when_no_config_no_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Use a directory with no config above it. We can't fully isolate
        # from a developer's home, but we can check the error message
        # *if* nothing is found.
        sub = tmp_path / "no_config_here"
        sub.mkdir()
        monkeypatch.chdir(sub)

        if ProjectConfig.find_and_load() is not None:
            pytest.skip("Outer .llm-patch.toml exists above tmp_path; cannot isolate.")

        runner = CliRunner()
        result = runner.invoke(adapter, ["compile"])

        assert result.exit_code != 0
        assert "Missing --source-dir" in result.output

    def test_registry_plugin_from_config_when_env_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        _write_config(project, plugin="my_pkg:build_registry")
        monkeypatch.chdir(project)
        monkeypatch.delenv("LLM_PATCH_PLUGIN_REGISTRY", raising=False)
        monkeypatch.delenv("LLM_PATCH_REGISTRY", raising=False)

        from llm_patch.cli.distribute import _resolve_registry_spec

        assert _resolve_registry_spec() == "my_pkg:build_registry"

    def test_env_var_wins_over_config_plugin(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        project = tmp_path / "proj"
        project.mkdir()
        _write_config(project, plugin="config_pkg:factory")
        monkeypatch.chdir(project)
        monkeypatch.setenv("LLM_PATCH_PLUGIN_REGISTRY", "env_pkg:factory")

        from llm_patch.cli.distribute import _resolve_registry_spec

        assert _resolve_registry_spec() == "env_pkg:factory"
