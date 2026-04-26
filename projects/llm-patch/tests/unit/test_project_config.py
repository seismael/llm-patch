"""Tests for ``llm_patch.core.project_config``.

Covers parsing, upward-walk discovery, env-var precedence, and graceful
handling of missing/empty/malformed sections.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_patch.core.project_config import (
    CONFIG_FILENAME,
    ProjectConfig,
)


# ── ProjectConfig.load ────────────────────────────────────────────────


class TestProjectConfigLoad:
    def test_parses_all_sections(self, tmp_path: Path) -> None:
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text(
            """
[project]
name = "demo"
description = "An llm-patch project."

[compile]
source = "./docs"
output = "./adapters"

[runtime]
base_model = "google/gemma-2-2b-it"

[registry]
plugin = "my_pkg:build_registry"
""",
            encoding="utf-8",
        )

        loaded = ProjectConfig.load(cfg)

        assert loaded.name == "demo"
        assert loaded.description == "An llm-patch project."
        assert loaded.compile.source == (tmp_path / "docs").resolve()
        assert loaded.compile.output == (tmp_path / "adapters").resolve()
        assert loaded.runtime.base_model == "google/gemma-2-2b-it"
        assert loaded.registry.plugin == "my_pkg:build_registry"

    def test_missing_sections_default_to_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text("[project]\nname = \"x\"\n", encoding="utf-8")

        loaded = ProjectConfig.load(cfg)

        assert loaded.compile.source is None
        assert loaded.compile.output is None
        assert loaded.runtime.base_model is None
        assert loaded.registry.plugin is None

    def test_empty_strings_become_none(self, tmp_path: Path) -> None:
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text(
            "[project]\nname = \"\"\n[runtime]\nbase_model = \"   \"\n",
            encoding="utf-8",
        )

        loaded = ProjectConfig.load(cfg)

        assert loaded.name is None
        assert loaded.runtime.base_model is None

    def test_absolute_paths_are_preserved(self, tmp_path: Path) -> None:
        absolute = (tmp_path / "elsewhere").resolve()
        cfg = tmp_path / CONFIG_FILENAME
        cfg.write_text(
            f"[compile]\nsource = \"{absolute.as_posix()}\"\n",
            encoding="utf-8",
        )

        loaded = ProjectConfig.load(cfg)

        assert loaded.compile.source == absolute


# ── ProjectConfig.find ────────────────────────────────────────────────


class TestProjectConfigFind:
    def test_finds_config_in_starting_directory(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text("[project]\n", encoding="utf-8")

        located = ProjectConfig.find(tmp_path)

        assert located == (tmp_path / CONFIG_FILENAME).resolve()

    def test_walks_upward_to_find_config(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text("[project]\n", encoding="utf-8")
        nested = tmp_path / "a" / "b" / "c"
        nested.mkdir(parents=True)

        located = ProjectConfig.find(nested)

        assert located == (tmp_path / CONFIG_FILENAME).resolve()

    def test_returns_none_when_no_config_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Walk upward will eventually hit a directory with no config.
        # Use a path under tmp_path; any ``.llm-patch.toml`` higher up
        # (e.g. in the developer's home) would still be returned, so we
        # only assert that we either get None or something *outside*
        # tmp_path that we don't control.
        sub = tmp_path / "deep"
        sub.mkdir()

        located = ProjectConfig.find(sub)

        if located is not None:
            # If something existed higher up (CI worker config, dev box),
            # at least it must not be inside tmp_path.
            assert tmp_path not in located.parents


# ── apply_registry_env ────────────────────────────────────────────────


class TestApplyRegistryEnv:
    def test_sets_var_when_unset(self) -> None:
        env: dict[str, str] = {}
        cfg = ProjectConfig(
            path=Path("/tmp/x"),
            registry=ProjectConfig.__dataclass_fields__["registry"].default.__class__(
                plugin="my_pkg:build"
            ),
        )

        applied = cfg.apply_registry_env(env=env)

        assert applied is True
        assert env["LLM_PATCH_PLUGIN_REGISTRY"] == "my_pkg:build"

    def test_does_not_overwrite_existing_var(self) -> None:
        env = {"LLM_PATCH_PLUGIN_REGISTRY": "user_set:factory"}
        cfg = ProjectConfig(
            path=Path("/tmp/x"),
            registry=ProjectConfig.__dataclass_fields__["registry"].default.__class__(
                plugin="my_pkg:build"
            ),
        )

        applied = cfg.apply_registry_env(env=env)

        assert applied is False
        assert env["LLM_PATCH_PLUGIN_REGISTRY"] == "user_set:factory"

    def test_no_op_when_plugin_unset(self) -> None:
        env: dict[str, str] = {}
        cfg = ProjectConfig(path=Path("/tmp/x"))

        applied = cfg.apply_registry_env(env=env)

        assert applied is False
        assert "LLM_PATCH_PLUGIN_REGISTRY" not in env


# ── find_and_load convenience ─────────────────────────────────────────


class TestFindAndLoad:
    def test_returns_none_when_nothing_found(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Re-anchor cwd to an empty directory to avoid finding higher-up
        # configs on the developer machine. A loaded config from above
        # is acceptable as long as it parses; only assert *some* outcome.
        monkeypatch.chdir(tmp_path / "deep" if (tmp_path / "deep").exists() else tmp_path)
        sub = tmp_path / "isolated"
        sub.mkdir()

        result = ProjectConfig.find_and_load(sub)

        # Either no config found, or one was found above tmp_path.
        if result is not None:
            assert tmp_path not in result.path.parents

    def test_returns_loaded_config_when_present(self, tmp_path: Path) -> None:
        (tmp_path / CONFIG_FILENAME).write_text(
            "[project]\nname = \"hello\"\n", encoding="utf-8"
        )

        result = ProjectConfig.find_and_load(tmp_path)

        assert result is not None
        assert result.name == "hello"
