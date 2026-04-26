"""``llm-patch version`` — print engine version (text or JSON)."""

from __future__ import annotations

import json as _json
import platform
import sys

import click

from llm_patch import __version__


@click.command("version")
@click.option("--json", "as_json", is_flag=True, help="Emit JSON.")
def version(as_json: bool) -> None:
    """Print the installed llm-patch version."""
    if as_json:
        click.echo(
            _json.dumps(
                {
                    "llm_patch": __version__,
                    "python": platform.python_version(),
                    "executable": sys.executable,
                },
                indent=2,
            )
        )
        return
    click.echo(f"llm-patch {__version__} (Python {platform.python_version()})")


__all__ = ["version"]
