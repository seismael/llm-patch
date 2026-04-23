"""Unified CLI for llm-patch.

Provides the top-level ``llm-patch`` command with subgroups:

* ``llm-patch wiki``    — knowledge wiki management.
* ``llm-patch adapter`` — LoRA adapter generation pipeline (legacy).
* ``llm-patch source``  — data source inspection / preview.
* ``llm-patch model``   — load model, attach adapters, generate / chat.
* ``llm-patch serve``   — start the HTTP API server.

Each subgroup lives in its own module so the dependency surface stays
narrow.
"""

from __future__ import annotations

import sys

try:
    import click
except ImportError:
    print(
        "CLI requires click. Install with: pip install 'llm-patch[cli]'",
        file=sys.stderr,
    )
    raise SystemExit(1)

from llm_patch.cli.adapter import adapter
from llm_patch.cli.model import model
from llm_patch.cli.serve import serve
from llm_patch.cli.source import source
from llm_patch.cli.wiki import wiki


@click.group()
@click.version_option(package_name="llm-patch")
def cli() -> None:
    """llm-patch — Ingest → Compile → Attach → Use.

    Convert text documents into LoRA adapter weights, attach them to any
    HuggingFace model, and serve the patched model for inference.
    """


cli.add_command(wiki)
cli.add_command(adapter)
cli.add_command(source)
cli.add_command(model)
cli.add_command(serve)


def main() -> None:
    """Entry point for the ``llm-patch`` console script."""
    cli()
