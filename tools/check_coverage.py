"""Per-package coverage threshold enforcer.

Reads a ``coverage.xml`` (Cobertura format) and fails if the per-package
branch coverage of any configured "core" package is below its threshold.

Configured here so it can evolve without touching ``pyproject.toml``.

Usage::

    coverage xml -o coverage.xml
    python tools/check_coverage.py coverage.xml
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# package import-prefix -> minimum branch-rate (0..1)
THRESHOLDS: dict[str, float] = {
    "llm_patch.core": 0.85,
}


def _coverage_name_candidates(prefix: str) -> set[str]:
    """Return normalized Cobertura package-name candidates for an import prefix.

    Coverage XML generated from a project-local run often emits package names
    relative to the configured ``source`` root (for example ``core`` instead of
    ``llm_patch.core``). Matching both forms keeps the gate stable across
    platforms and invocation styles.
    """
    normalized = prefix.replace("\\", ".").replace("/", ".").strip(".")
    parts = [part for part in normalized.split(".") if part]
    return {
        ".".join(parts[index:])
        for index in range(len(parts))
        if ".".join(parts[index:])
    }


def _branch_rate_for(tree: ET.ElementTree, prefix: str) -> float | None:
    """Return aggregate branch-rate for all packages whose name starts with ``prefix``."""
    total_branches = 0
    covered_branches = 0
    found = False
    candidates = _coverage_name_candidates(prefix)
    for pkg in tree.iterfind(".//package"):
        name = pkg.get("name", "").replace("\\", ".").replace("/", ".").strip(".")
        if not any(name == candidate or name.startswith(candidate + ".") for candidate in candidates):
            continue
        found = True
        # Sum branches across classes in the package.
        for cls in pkg.iterfind(".//class"):
            for line in cls.iterfind(".//line"):
                if line.get("branch") == "true":
                    cond = line.get("condition-coverage", "0% (0/0)")
                    # "50% (1/2)" -> covered=1, total=2
                    try:
                        frac = cond.split("(")[1].rstrip(")")
                        cov, tot = frac.split("/")
                        total_branches += int(tot)
                        covered_branches += int(cov)
                    except (IndexError, ValueError):
                        continue
    if not found or total_branches == 0:
        return None
    return covered_branches / total_branches


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: check_coverage.py <coverage.xml>", file=sys.stderr)
        return 2
    xml_path = Path(argv[1])
    if not xml_path.exists():
        print(f"check_coverage: {xml_path} not found", file=sys.stderr)
        return 2

    tree = ET.parse(xml_path)
    failed = False
    for prefix, floor in THRESHOLDS.items():
        rate = _branch_rate_for(tree, prefix)
        if rate is None:
            print(f"check_coverage: no coverage data for '{prefix}' (skipped)")
            continue
        pct = rate * 100
        floor_pct = floor * 100
        status = "OK" if rate >= floor else "FAIL"
        print(f"check_coverage: {prefix} branch={pct:.1f}% (floor {floor_pct:.0f}%) {status}")
        if rate < floor:
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
