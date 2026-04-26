"""Project-level configuration loaded from ``.llm-patch.toml``.

A small, dependency-free loader that walks upward from a starting
directory looking for a ``.llm-patch.toml`` file (the same name written
by ``llm-patch init``). The file is parsed with the stdlib ``tomllib``
module — no third-party dependencies, no I/O at import time.

The loader is intentionally **forgiving**: missing sections, unknown
keys, and unreadable files all degrade gracefully. The CLI consumes
``ProjectConfig`` purely as a *fallback* when explicit flags are not
supplied; explicit flags must always win, and environment variables set
by the user (e.g. ``LLM_PATCH_PLUGIN_REGISTRY``) must never be
overwritten by the config — see :meth:`ProjectConfig.apply_registry_env`.

The module is part of :mod:`llm_patch.core` and follows the engine
boundary rules: no imports from use-cases, no heavy dependencies.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Final

CONFIG_FILENAME: Final = ".llm-patch.toml"


@dataclass(frozen=True, slots=True)
class CompileSection:
    """Defaults for ``llm-patch compile`` / ``watch``."""

    source: Path | None = None
    output: Path | None = None


@dataclass(frozen=True, slots=True)
class RuntimeSection:
    """Defaults for ``llm-patch chat`` / ``generate``."""

    base_model: str | None = None


@dataclass(frozen=True, slots=True)
class RegistrySection:
    """Defaults for ``llm-patch push`` / ``pull``."""

    plugin: str | None = None


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Parsed view of a ``.llm-patch.toml`` file.

    All path fields are resolved relative to the config file's parent
    directory at load time so that callers can use them directly.
    """

    path: Path
    name: str | None = None
    description: str | None = None
    compile: CompileSection = CompileSection()
    runtime: RuntimeSection = RuntimeSection()
    registry: RegistrySection = RegistrySection()

    @classmethod
    def find(cls, start: Path | None = None) -> Path | None:
        """Walk upward from ``start`` looking for ``.llm-patch.toml``.

        Returns the absolute path to the first match, or ``None`` if no
        config file exists between ``start`` and the filesystem root.
        """
        cursor = (start or Path.cwd()).resolve()
        if cursor.is_file():
            cursor = cursor.parent
        for directory in (cursor, *cursor.parents):
            candidate = directory / CONFIG_FILENAME
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def load(cls, path: Path) -> "ProjectConfig":
        """Load and parse a config file. Raises ``OSError`` if unreadable."""
        with path.open("rb") as fh:
            raw = tomllib.load(fh)
        base_dir = path.parent.resolve()
        project = raw.get("project", {}) if isinstance(raw.get("project"), dict) else {}
        compile_raw = raw.get("compile", {}) if isinstance(raw.get("compile"), dict) else {}
        runtime_raw = raw.get("runtime", {}) if isinstance(raw.get("runtime"), dict) else {}
        registry_raw = raw.get("registry", {}) if isinstance(raw.get("registry"), dict) else {}

        return cls(
            path=path.resolve(),
            name=_str_or_none(project.get("name")),
            description=_str_or_none(project.get("description")),
            compile=CompileSection(
                source=_resolve_path(compile_raw.get("source"), base_dir),
                output=_resolve_path(compile_raw.get("output"), base_dir),
            ),
            runtime=RuntimeSection(
                base_model=_str_or_none(runtime_raw.get("base_model")),
            ),
            registry=RegistrySection(
                plugin=_str_or_none(registry_raw.get("plugin")),
            ),
        )

    @classmethod
    def find_and_load(cls, start: Path | None = None) -> "ProjectConfig | None":
        """Convenience: locate ``.llm-patch.toml`` and parse it."""
        located = cls.find(start)
        if located is None:
            return None
        return cls.load(located)

    def apply_registry_env(
        self, *, env: dict[str, str] | None = None, var: str = "LLM_PATCH_PLUGIN_REGISTRY"
    ) -> bool:
        """Populate the registry env var **only if it is not already set**.

        Returns ``True`` if the env var was set by this call, ``False``
        otherwise. Operator/CI-supplied values always win over project
        config.
        """
        if self.registry.plugin is None:
            return False
        target = env if env is not None else os.environ
        if var in target and target[var]:
            return False
        target[var] = self.registry.plugin
        return True


def _str_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _resolve_path(value: object, base_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


__all__ = [
    "CONFIG_FILENAME",
    "CompileSection",
    "ProjectConfig",
    "RegistrySection",
    "RuntimeSection",
]
