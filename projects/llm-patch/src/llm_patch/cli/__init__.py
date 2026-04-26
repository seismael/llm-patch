"""Unified CLI for llm-patch.

Primary verbs (always shown in ``--help``):

* ``init``     — scaffold a project and write ``.llm-patch.toml``.
* ``doctor``   — verify Python / torch / CUDA / extras / registry plugin.
* ``compile``  — folder of docs → LoRA adapter.
* ``watch``    — same as compile, but re-runs on file changes.
* ``chat``     — load base + adapter and start an interactive session.
* ``push`` / ``pull`` / ``hub`` — adapter market commands.
* ``serve``    — start the HTTP API server.
* ``version``  — print the installed engine version.

Advanced groups (``adapter``, ``model``, ``source``, ``wiki``) remain
fully wired but are hidden in ``--help`` unless the user opts in by
setting ``LLM_PATCH_ADVANCED=1``.
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

from llm_patch.cli._registry import CommandRegistry
from llm_patch.cli.adapter import adapter
from llm_patch.cli.adapter import compile as _adapter_compile
from llm_patch.cli.adapter import watch as _adapter_watch
from llm_patch.cli.distribute import hub, pull, push
from llm_patch.cli.doctor import doctor
from llm_patch.cli.init import init
from llm_patch.cli.model import chat as _model_chat
from llm_patch.cli.model import model
from llm_patch.cli.serve import serve
from llm_patch.cli.source import source
from llm_patch.cli.version import version
from llm_patch.cli.wiki import wiki


@click.group()
@click.version_option(package_name="llm-patch")
@click.option("--quiet", is_flag=True, help="Suppress non-essential output.")
@click.option("--json", "as_json", is_flag=True, help="Prefer JSON output where supported.")
@click.option("--no-color", is_flag=True, help="Disable ANSI colors in output.")
@click.pass_context
def cli(ctx: click.Context, quiet: bool, as_json: bool, no_color: bool) -> None:
    """llm-patch — turn any folder into a LoRA adapter in seconds.

    Run ``llm-patch doctor`` once after installing to verify your
    environment, then ``llm-patch init`` to scaffold a project.

    Set ``LLM_PATCH_ADVANCED=1`` to reveal advanced subgroups
    (``adapter``, ``model``, ``source``, ``wiki``) in ``--help``.
    """
    ctx.ensure_object(dict)
    ctx.obj["quiet"] = quiet
    ctx.obj["json"] = as_json
    ctx.obj["no_color"] = no_color


_registry = CommandRegistry(group=cli)

# Primary verbs (always visible).
_registry.register(init)
_registry.register(doctor)
_registry.register(_adapter_compile, name="compile")
_registry.register(_adapter_watch, name="watch")
_registry.register(_model_chat, name="chat")
_registry.register(push)
_registry.register(pull)
_registry.register(hub)
_registry.register(serve)
_registry.register(version)

# Advanced groups (hidden unless LLM_PATCH_ADVANCED=1).
_registry.register(adapter, hidden=True)
_registry.register(model, hidden=True)
_registry.register(source, hidden=True)
_registry.register(wiki, hidden=True)


def main() -> None:
    """Entry point for the ``llm-patch`` console script."""
    cli()
