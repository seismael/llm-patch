"""Enforce per-project ``CHANGELOG.md`` ``Unreleased`` updates.

Used by CI on pull requests. Compares the changed file list (passed as
arguments or read from stdin one-per-line) against the project layout
under ``projects/<name>/``. For each project that has any modified
file, requires that the project's ``CHANGELOG.md`` was also touched and
contains a non-empty ``[Unreleased]`` entry.

Usage::

    git diff --name-only origin/main... | python tools/check_changelog.py
    python tools/check_changelog.py path/to/file path/to/other
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROJECTS_DIR = ROOT / "projects"
UNRELEASED_RE = re.compile(r"^##\s*\[Unreleased\]\s*$", re.MULTILINE)


def _project_of(rel_path: Path) -> str | None:
    parts = rel_path.parts
    if len(parts) >= 2 and parts[0] == "projects":
        return parts[1]
    return None


def _read_changed_paths(argv: list[str]) -> list[Path]:
    if len(argv) > 1:
        return [Path(p) for p in argv[1:]]
    return [Path(line.strip()) for line in sys.stdin if line.strip()]


def main(argv: list[str]) -> int:
    changed = _read_changed_paths(argv)
    affected_projects: set[str] = set()
    changelog_touched: set[str] = set()

    for p in changed:
        proj = _project_of(p)
        if proj is None:
            continue
        affected_projects.add(proj)
        if p.name == "CHANGELOG.md":
            changelog_touched.add(proj)

    failed = False
    for proj in sorted(affected_projects):
        changelog = PROJECTS_DIR / proj / "CHANGELOG.md"
        if not changelog.exists():
            continue
        if proj not in changelog_touched:
            print(
                f"check_changelog: project '{proj}' was modified but its "
                f"CHANGELOG.md was not updated",
                file=sys.stderr,
            )
            failed = True
            continue
        text = changelog.read_text(encoding="utf-8")
        match = UNRELEASED_RE.search(text)
        if not match:
            print(
                f"check_changelog: '{changelog}' has no '## [Unreleased]' section",
                file=sys.stderr,
            )
            failed = True
            continue
        # Verify there is at least one non-blank, non-heading line under [Unreleased].
        tail = text[match.end() :]
        next_section = re.search(r"^##\s+", tail, re.MULTILINE)
        body = tail[: next_section.start()] if next_section else tail
        if not any(line.strip() and not line.startswith("##") for line in body.splitlines()):
            print(
                f"check_changelog: '{changelog}' has an empty [Unreleased] section",
                file=sys.stderr,
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
