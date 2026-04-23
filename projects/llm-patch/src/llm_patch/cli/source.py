"""Source management subcommands for ``llm-patch source``.

Provides commands to list, preview, and validate configured data sources.
"""

from __future__ import annotations

from pathlib import Path

import click

from llm_patch.core.interfaces import IDataSource


def _build_source(
    kind: str,
    path: str | None,
    *,
    patterns: tuple[str, ...] = ("*.md",),
) -> IDataSource:
    """Factory: build an IDataSource from a kind and path."""
    source_path = Path(path or ".")

    if kind == "markdown":
        from llm_patch.sources.markdown import MarkdownDataSource

        return MarkdownDataSource(source_path, patterns=list(patterns))
    if kind == "wiki":
        from llm_patch.sources.wiki import WikiDataSource

        return WikiDataSource(source_path, patterns=list(patterns))
    if kind == "pdf":
        from llm_patch.sources.pdf import PdfDataSource

        return PdfDataSource(source_path)
    if kind == "jsonl":
        from llm_patch.sources.jsonl import JsonlDataSource

        return JsonlDataSource(source_path)

    raise click.ClickException(f"Unknown source kind: {kind}")


@click.group()
def source() -> None:
    """Data source management.

    Inspect and preview data sources that feed into the adapter
    compilation pipeline.
    """


@source.command("list")
@click.option(
    "--kind",
    type=click.Choice(["markdown", "wiki", "pdf", "jsonl"]),
    required=True,
    help="Type of data source.",
)
@click.option("--path", type=click.Path(exists=True), required=True, help="Path to source data.")
@click.option("--pattern", "patterns", multiple=True, default=["*.md"], help="File glob patterns.")
def list_docs(kind: str, path: str, patterns: tuple[str, ...]) -> None:
    """List all document IDs from a data source."""
    src = _build_source(kind, path, patterns=patterns)
    count = 0
    for doc in src.fetch_all():
        click.echo(f"  {doc.document_id}  ({len(doc.content)} chars)")
        count += 1
    click.echo(f"\n{count} document(s) found.")


@source.command()
@click.option(
    "--kind",
    type=click.Choice(["markdown", "wiki", "pdf", "jsonl"]),
    required=True,
    help="Type of data source.",
)
@click.option("--path", type=click.Path(exists=True), required=True, help="Path to source data.")
@click.argument("doc_id")
def preview(kind: str, path: str, doc_id: str) -> None:
    """Preview the content of a single document by ID."""
    src = _build_source(kind, path)
    doc = src.fetch_one(doc_id)
    if doc is None:
        raise click.ClickException(f"Document not found: {doc_id}")
    click.echo(f"--- {doc.document_id} ---")
    click.echo(doc.content[:2000])
    if len(doc.content) > 2000:
        click.echo(f"\n... ({len(doc.content) - 2000} more chars)")


@source.command()
@click.option(
    "--kind",
    type=click.Choice(["markdown", "wiki", "pdf", "jsonl"]),
    required=True,
    help="Type of data source.",
)
@click.option("--path", type=click.Path(exists=True), required=True, help="Path to source data.")
def count(kind: str, path: str) -> None:
    """Count documents in a data source."""
    src = _build_source(kind, path)
    n = sum(1 for _ in src.fetch_all())
    click.echo(f"{n} document(s).")
