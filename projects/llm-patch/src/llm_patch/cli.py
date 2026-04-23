"""CLI entry point for ``llm-patch wiki`` subcommands.

Requires the ``click`` package (included in the ``cli`` extra).
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import click
except ImportError as exc:
    print("CLI requires click. Install with: pip install 'llm-patch[cli]'", file=sys.stderr)
    raise SystemExit(1) from exc

from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.schema import WikiSchema


def _make_agent(
    agent_type: str,
    api_key: str | None,
    model: str | None,
    schema: WikiSchema | None,
):
    """Create a wiki agent based on the --agent flag."""
    if agent_type == "mock":
        return MockWikiAgent(schema)

    if agent_type == "litellm":
        try:
            from llm_patch.wiki.agents.litellm_agent import LiteLLMWikiAgent
        except ImportError as exc:
            click.echo(
                "litellm package not installed. "
                "Run: pip install 'llm-patch[llm]'\n"
                "Or use --agent mock to run without an LLM.",
                err=True,
            )
            raise SystemExit(1) from exc

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        if schema:
            kwargs["schema"] = schema
        return LiteLLMWikiAgent(**kwargs)

    # Claude agent (default)
    try:
        from llm_patch.wiki.agents.anthropic_agent import AnthropicWikiAgent
    except ImportError as exc:
        click.echo(
            "anthropic package not installed. "
            "Run: pip install 'llm-patch[anthropic]'\n"
            "Or use --agent mock to run without an LLM.",
            err=True,
        )
        raise SystemExit(1) from exc

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if model:
        kwargs["model"] = model
    if schema:
        kwargs["schema"] = schema
    return AnthropicWikiAgent(**kwargs)


def _make_wiki(
    base_dir: Path,
    schema_path: Path | None,
    agent_type: str,
    api_key: str | None,
    model: str | None,
) -> WikiManager:
    schema: WikiSchema | None = None
    if schema_path and schema_path.exists():
        schema = WikiSchema.from_file(schema_path)
    agent = _make_agent(agent_type, api_key, model, schema)
    return WikiManager(agent=agent, base_dir=base_dir, schema=schema)


@click.group()
@click.option(
    "--base-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
    help="Root directory containing raw/ and wiki/.",
)
@click.option(
    "--schema",
    "schema_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a CLAUDE.md-style wiki schema file.",
)
@click.option(
    "--agent",
    "agent_type",
    type=click.Choice(["claude", "litellm", "mock"], case_sensitive=False),
    default="claude",
    help="Agent backend: claude (default), litellm (Gemini/OpenAI/etc.), or mock.",
)
@click.option(
    "--api-key",
    default=None,
    help="LLM API key (default: provider-specific env var).",
)
@click.option(
    "--model",
    default=None,
    help="LLM model name (e.g. gemini/gemini-2.5-pro, claude-sonnet-4-20250514).",
)
@click.pass_context
def cli(
    ctx: click.Context,
    base_dir: Path,
    schema_path: Path | None,
    agent_type: str,
    api_key: str | None,
    model: str | None,
) -> None:
    """llm-patch wiki — manage your LLM knowledge wiki."""
    ctx.ensure_object(dict)
    ctx.obj["wiki"] = _make_wiki(base_dir, schema_path, agent_type, api_key, model)


@cli.command()
@click.option("--obsidian", is_flag=True, help="Also initialise as an Obsidian vault.")
@click.pass_context
def init(ctx: click.Context, obsidian: bool) -> None:
    """Initialize the wiki directory structure."""
    wiki: WikiManager = ctx.obj["wiki"]
    wiki.init(obsidian=obsidian)
    click.echo("Wiki initialized.")
    if obsidian:
        click.echo("Obsidian vault configured (.obsidian/ created).")


@cli.command()
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.pass_context
def ingest(ctx: click.Context, source: Path) -> None:
    """Ingest a raw source file into the wiki."""
    wiki: WikiManager = ctx.obj["wiki"]
    result = wiki.ingest(source.resolve())
    click.echo(f"Created: {', '.join(result.pages_created) or 'none'}")
    click.echo(f"Updated: {', '.join(result.pages_updated) or 'none'}")
    click.echo(f"Entities: {len(result.entities_extracted)}")


@cli.command()
@click.argument("question")
@click.option("--save", is_flag=True, help="Save the answer as a synthesis page.")
@click.pass_context
def query(ctx: click.Context, question: str, save: bool) -> None:
    """Query the wiki with a natural language question."""
    wiki: WikiManager = ctx.obj["wiki"]
    result = wiki.query(question, save_as_synthesis=save)
    click.echo(result.answer)
    if result.cited_pages:
        click.echo(f"\nCited: {', '.join(result.cited_pages)}")
    if result.filed_as:
        click.echo(f"Saved as: {result.filed_as}")


@cli.command()
@click.pass_context
def lint(ctx: click.Context) -> None:
    """Run a health check on the wiki."""
    wiki: WikiManager = ctx.obj["wiki"]
    report = wiki.lint()
    if not report.issues:
        click.echo("No issues found.")
        return
    for issue in report.issues:
        click.echo(f"[{issue.category}] {issue.page}: {issue.description}")
    click.echo(f"\nTotal: {report.issue_count} issue(s)")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show wiki status."""
    wiki: WikiManager = ctx.obj["wiki"]
    for key, value in wiki.status().items():
        click.echo(f"{key.replace('_', ' ').title()}: {value}")


@cli.command()
@click.pass_context
def compile(ctx: click.Context) -> None:
    """Batch ingest all unprocessed raw sources."""
    wiki: WikiManager = ctx.obj["wiki"]
    results = wiki.compile_all()
    click.echo(f"Compiled {len(results)} source(s).")
    for result in results:
        click.echo(f"  {result.source_path}: {len(result.pages_created)} created")


# -----------------------------------------------------------------------
# Obsidian subcommand group
# -----------------------------------------------------------------------


@cli.group()
def obsidian() -> None:
    """Obsidian vault management commands."""


@obsidian.command("init")
@click.option(
    "--attachment-folder",
    default="raw/assets",
    help="Relative path for downloaded images/assets.",
)
@click.option(
    "--ignore",
    "ignore_filters",
    multiple=True,
    help="Additional paths to exclude from Obsidian graph (repeatable).",
)
@click.pass_context
def obsidian_init(
    ctx: click.Context,
    attachment_folder: str,
    ignore_filters: tuple[str, ...],
) -> None:
    """Set up the wiki directory as an Obsidian vault."""
    from llm_patch.wiki.obsidian import ObsidianConfig

    wiki: WikiManager = ctx.obj["wiki"]

    filters = [
        ".claude",
        "raw",
        ".git",
        "src",
        "tests",
        "docs",
        "examples",
        ".venv",
        "__pycache__",
        ".pytest_tmp",
        "node_modules",
        ".github",
        *ignore_filters,
    ]
    cfg = ObsidianConfig(
        enabled=True,
        attachment_folder=attachment_folder,
        ignore_filters=filters,
    )
    vault = wiki.enable_obsidian(cfg)
    click.echo(f"Obsidian vault initialised at {vault.root}")
    click.echo(f"  Attachment folder: {attachment_folder}")
    click.echo(f"  Ignore filters: {', '.join(filters)}")


@obsidian.command("graph")
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Write graph JSON to a file.",
)
@click.option("--summary", is_flag=True, help="Print summary counts only.")
@click.pass_context
def obsidian_graph(ctx: click.Context, output: Path | None, summary: bool) -> None:
    """Export the wiki knowledge graph."""
    wiki: WikiManager = ctx.obj["wiki"]
    graph_data = wiki.graph()

    if summary:
        click.echo(f"Nodes: {graph_data.node_count}")
        click.echo(f"Edges: {graph_data.edge_count}")
        return

    json_str = wiki.export_graph(output)
    if output:
        click.echo(
            f"Graph exported to {output} "
            f"({graph_data.node_count} nodes, {graph_data.edge_count} edges)"
        )
    else:
        click.echo(json_str)


@obsidian.command("status")
@click.pass_context
def obsidian_status(ctx: click.Context) -> None:
    """Show Obsidian vault status."""
    wiki: WikiManager = ctx.obj["wiki"]
    vault = wiki.obsidian
    if vault is None or not vault.is_vault:
        click.echo("Obsidian vault: not configured")
        click.echo("Run 'llm-patch-wiki obsidian init' to set up.")
        return

    app_cfg = vault.read_app_config()
    click.echo("Obsidian vault: active")
    click.echo(f"  Root: {vault.root}")
    attach = app_cfg.get("attachmentFolderPath", "(default)")
    click.echo(f"  Attachment folder: {attach}")
    filters = app_cfg.get("userIgnoreFilters", [])
    click.echo(f"  Ignore filters: {', '.join(filters) if filters else '(none)'}")

    graph_data = wiki.graph()
    click.echo(f"  Graph nodes: {graph_data.node_count}")
    click.echo(f"  Graph edges: {graph_data.edge_count}")


def main() -> None:
    """Entry point for ``llm-patch-wiki`` console script."""
    cli()


if __name__ == "__main__":
    main()
