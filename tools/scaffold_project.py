"""Scaffold a new project under ``projects/<name>/``.

Materializes the standardized layout described in ``SPEC.md §9``:

    projects/<name>/
        src/<package>/__init__.py
        src/<package>/py.typed
        tests/unit/test_smoke.py
        tests/integration/__init__.py
        docs/.gitkeep
        data/.gitkeep
        artifacts/.gitkeep
        examples/.gitkeep
        pyproject.toml
        AGENTS.md
        README.md
        CHANGELOG.md

The new project is automatically picked up by the workspace because
``pyproject.toml`` declares ``members = ["projects/*"]``.

Usage::

    python tools/scaffold_project.py <project-name> [--package <import_name>]

Example::

    python tools/scaffold_project.py chat-web --package llm_patch_chat_web
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"


def _default_package(project_name: str) -> str:
    return "llm_patch_" + project_name.replace("-", "_")


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def scaffold(project_name: str, package: str) -> int:
    project_dir = PROJECTS_DIR / project_name
    if project_dir.exists():
        print(f"scaffold: {project_dir} already exists, refusing to overwrite", file=sys.stderr)
        return 1

    _write(
        project_dir / "pyproject.toml",
        dedent(f"""\
        [project]
        name = "{project_name}"
        version = "0.1.0"
        description = "TODO: short description of {project_name}"
        license = {{ text = "Apache-2.0" }}
        readme = "README.md"
        requires-python = ">=3.11"
        dependencies = [
            "llm-patch",
            "llm-patch-shared",
        ]

        [build-system]
        requires = ["hatchling"]
        build-backend = "hatchling.build"

        [tool.hatch.build.targets.wheel]
        packages = ["src/{package}"]

        [tool.mypy]
        python_version = "3.11"
        strict = true
        files = ["src/{package}", "tests"]

        [tool.pytest.ini_options]
        testpaths = ["tests"]
        addopts = "--strict-markers -v"
    """),
    )

    _write(
        project_dir / "src" / package / "__init__.py",
        dedent(f'''\
        """{package} — TODO: one-line summary.

        See AGENTS.md for the per-project contract and the root SPEC.md for
        the governing engineering specification.
        """

        __version__ = "0.1.0"

        __all__ = ["__version__"]
    '''),
    )
    _write(project_dir / "src" / package / "py.typed", "")

    _write(
        project_dir / "tests" / "unit" / "test_smoke.py",
        dedent(f"""\
        \"\"\"Smoke test — verifies the package imports.\"\"\"

        from __future__ import annotations

        import {package}


        def test_version_exposed() -> None:
            assert isinstance({package}.__version__, str)
    """),
    )
    _write(project_dir / "tests" / "integration" / "__init__.py", "")

    for sub in ("docs", "data", "artifacts", "examples"):
        _write(project_dir / sub / ".gitkeep", "")

    _write(
        project_dir / "README.md",
        dedent(f"""\
        # {project_name}

        TODO: describe this use-case and how it composes the engine.

        See the root [SPEC.md](../../SPEC.md) and per-project
        [AGENTS.md](AGENTS.md).
    """),
    )

    _write(
        project_dir / "AGENTS.md",
        dedent(f"""\
        # AGENTS — `{project_name}`

        Per-project agent contract. Read with the root
        [SPEC.md](../../SPEC.md) and [AGENTS.md](../../AGENTS.md).

        ## Goal

        TODO.

        ## Public API

        TODO — symbols re-exported from `{package}.__init__`.

        ## Allowed Dependencies

        - `llm-patch` (workspace, public API only).
        - `llm-patch-shared` (workspace).
        - Anything else requires an ADR.

        ## Test

        ```pwsh
        uv run --package {project_name} pytest -q
        ```
    """),
    )

    _write(
        project_dir / "CHANGELOG.md",
        dedent(f"""\
        # Changelog — {project_name}

        Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
        versioning: [SemVer](https://semver.org/).

        ## [Unreleased]

        ## [0.1.0]

        ### Added
        - Initial scaffold.
    """),
    )

    print(f"scaffold: created {project_dir}")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new project under projects/.")
    parser.add_argument("name", help="Project (distribution) name, e.g. 'chat-web'.")
    parser.add_argument(
        "--package",
        default=None,
        help="Python import name (default: derived from <name> as 'llm_patch_<snake>').",
    )
    args = parser.parse_args(argv[1:])
    package = args.package or _default_package(args.name)
    return scaffold(args.name, package)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
