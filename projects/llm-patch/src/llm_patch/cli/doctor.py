"""``llm-patch doctor`` — environment + dependency probe.

Reports Python version, optional extras, torch/CUDA, and registry-plugin
availability. Designed to be the first command users run after install.

Heavy ML deps (torch) are imported lazily through
:func:`llm_patch.runtime.preflight.probe`.
"""

from __future__ import annotations

import json as _json
import os
import platform
import sys
from importlib import import_module
from importlib.util import find_spec
from typing import Any

import click

from llm_patch.runtime.preflight import probe

_OPTIONAL_EXTRAS = (
    "click",       # CLI
    "fastapi",     # serve
    "mcp",         # MCP server
    "peft",        # adapter loader
    "transformers",
    "torch",
    "pydantic",
    "yaml",        # wiki frontmatter
)


def _check_module(name: str) -> dict[str, Any]:
    spec = find_spec(name)
    if spec is None:
        return {"name": name, "installed": False, "version": None}
    try:
        mod = import_module(name)
        version = getattr(mod, "__version__", None)
    except Exception as exc:  # pragma: no cover - defensive
        return {"name": name, "installed": True, "version": None, "error": str(exc)}
    return {"name": name, "installed": True, "version": version}


def _check_registry_plugin() -> dict[str, Any]:
    spec = os.environ.get("LLM_PATCH_PLUGIN_REGISTRY", "")
    if not spec:
        return {"configured": False, "spec": None}
    return {"configured": True, "spec": spec}


def _build_report() -> dict[str, Any]:
    report = probe()
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "extras": [_check_module(m) for m in _OPTIONAL_EXTRAS],
        "torch": {
            "imported": report.torch_imported,
            "cuda_available": report.cuda_available,
            "device_count": report.device_count,
            "device_name": report.device_name,
            "free_vram_gib": round(report.free_vram_bytes / (1024**3), 2),
            "total_vram_gib": round(report.total_vram_bytes / (1024**3), 2),
        },
        "registry_plugin": _check_registry_plugin(),
    }


def _render_text(report: dict[str, Any]) -> str:
    lines: list[str] = []
    py = report["python"]
    lines.append(f"Python {py['version']} ({py['platform']})")
    lines.append(f"  {py['executable']}")
    lines.append("")
    lines.append("Optional extras:")
    for entry in report["extras"]:
        mark = "OK " if entry["installed"] else "-- "
        ver = entry["version"] or ("not installed" if not entry["installed"] else "?")
        lines.append(f"  [{mark}] {entry['name']:<14} {ver}")
    lines.append("")
    t = report["torch"]
    if not t["imported"]:
        lines.append("Torch: not installed (CPU-only execution).")
    elif not t["cuda_available"]:
        lines.append("Torch: installed; CUDA unavailable (CPU-only execution).")
    else:
        lines.append(
            f"Torch: CUDA OK | {t['device_count']} device(s) | "
            f"{t['device_name']} | {t['free_vram_gib']} / {t['total_vram_gib']} GiB free"
        )
    lines.append("")
    reg = report["registry_plugin"]
    if reg["configured"]:
        lines.append(f"Registry plugin: {reg['spec']}")
    else:
        lines.append(
            "Registry plugin: not configured "
            "(set LLM_PATCH_PLUGIN_REGISTRY=module:factory to enable push/pull)."
        )
    return "\n".join(lines)


@click.command("doctor")
@click.option("--json", "as_json", is_flag=True, help="Emit a JSON report instead of text.")
@click.option("--quiet", is_flag=True, help="Only print the summary line.")
def doctor(as_json: bool, quiet: bool) -> None:
    """Check the runtime environment for known issues."""
    report = _build_report()
    if as_json:
        click.echo(_json.dumps(report, indent=2, sort_keys=True))
        return
    if quiet:
        t = report["torch"]
        if t["cuda_available"]:
            click.echo(f"OK: Python {report['python']['version']} | CUDA on {t['device_name']}.")
        else:
            click.echo(f"OK: Python {report['python']['version']} | CPU-only.")
        return
    click.echo(_render_text(report))


__all__ = ["doctor"]
