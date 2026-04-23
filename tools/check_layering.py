"""Lightweight architectural fitness check.

Scans every project under ``projects/<name>/src`` and verifies that no
``.py`` file contains an import that violates the workspace's layering
rules defined in ``SPEC.md`` and ADR-0002:

    use-cases  ───►  engine  ───►  shared-utils

Forbidden imports (any of the following → exit code 1):

* Engine importing from any use-case package.
* ``llm-patch-shared`` importing from the engine or any use-case.
* A use-case importing from another use-case.
* A use-case importing from a non-public engine module
  (i.e., anything other than ``llm_patch`` itself).

Run:

    python tools/check_layering.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"

# Map: project directory name -> import-name of the package it owns.
PROJECT_PACKAGE: dict[str, str] = {
    "llm-patch": "llm_patch",
    "shared-utils": "llm_patch_shared",
    "wiki-agent": "llm_patch_wiki_agent",
}

# Map: package import-name -> layer (1=shared, 2=engine, 3=use-case).
LAYER: dict[str, int] = {
    "llm_patch_shared": 1,
    "llm_patch": 2,
    "llm_patch_wiki_agent": 3,
}

ALL_PACKAGES = set(PROJECT_PACKAGE.values())


def _imported_top_levels(tree: ast.AST) -> set[str]:
    """Collect all top-level module names imported by ``tree``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


def _full_imports(tree: ast.AST) -> list[str]:
    """Collect fully-qualified ``from x.y import z`` source modules."""
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            out.append(node.module)
    return out


def _violations_for_file(path: Path, owning_pkg: str) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [f"{path}: syntax error ({exc.msg})"]

    own_layer = LAYER[owning_pkg]
    issues: list[str] = []

    for top in _imported_top_levels(tree):
        if top not in ALL_PACKAGES or top == owning_pkg:
            continue
        other_layer = LAYER[top]
        if other_layer >= own_layer:
            issues.append(
                f"{path}: '{owning_pkg}' (layer {own_layer}) "
                f"imports from '{top}' (layer {other_layer}) — forbidden"
            )

    # Use-cases must consume the engine only via its top-level package.
    if own_layer == 3:
        for full in _full_imports(tree):
            head = full.split(".")[0]
            if head == "llm_patch" and "." in full:
                issues.append(
                    f"{path}: use-case imports engine internal '{full}' — "
                    "use the public top-level 'llm_patch' API only (ADR-0003)"
                )
    return issues


def main() -> int:
    if not PROJECTS_DIR.is_dir():
        print(f"check_layering: no projects/ directory at {PROJECTS_DIR}", file=sys.stderr)
        return 0

    all_issues: list[str] = []
    for project_dir in sorted(PROJECTS_DIR.iterdir()):
        if not project_dir.is_dir():
            continue
        owning_pkg = PROJECT_PACKAGE.get(project_dir.name)
        if owning_pkg is None:
            continue
        src_root = project_dir / "src" / owning_pkg
        if not src_root.is_dir():
            continue
        for py_file in src_root.rglob("*.py"):
            all_issues.extend(_violations_for_file(py_file, owning_pkg))

    if all_issues:
        print("Layering violations:", file=sys.stderr)
        for issue in all_issues:
            print(f"  - {issue}", file=sys.stderr)
        return 1

    print("check_layering: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
