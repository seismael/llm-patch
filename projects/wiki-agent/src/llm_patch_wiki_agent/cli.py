"""CLI entry point for ``llm-patch-wiki-agent``."""

from __future__ import annotations

from pathlib import Path

import click
from llm_patch_shared import LlmPatchError

from llm_patch_wiki_agent import __version__
from llm_patch_wiki_agent.agent import WikiAgent, WikiAgentConfig


def _build_agent(
    *,
    adapter_dir: Path,
    wiki_dir: Path | None = None,
    model_id: str | None = None,
    checkpoint_dir: Path | None = None,
    patterns: tuple[str, ...] = ("*.md",),
    recursive: bool = True,
    aggregate_links: bool = True,
    generator_device: str = "cuda",
    model_device_map: str = "auto",
    model_dtype: str = "float16",
    system_prompt: str | None = None,
) -> WikiAgent:
    config = WikiAgentConfig(
        adapter_dir=adapter_dir,
        wiki_dir=wiki_dir,
        model_id=model_id,
        checkpoint_dir=checkpoint_dir,
        source_patterns=patterns,
        recursive=recursive,
        aggregate_links=aggregate_links,
        generator_device=generator_device,
        model_device_map=model_device_map,
        model_dtype=model_dtype,
        system_prompt=system_prompt,
    )
    return WikiAgent(config)


def _run_agent_action(action: str, callback: click.utils.LazyFile | object) -> None:
    try:
        if not callable(callback):
            raise click.ClickException(f"{action} failed: internal callback is not callable")
        callback()
    except LlmPatchError as exc:
        raise click.ClickException(str(exc)) from exc


@click.group(help="Wiki-specialized agent built on the llm-patch engine.")
@click.version_option(__version__, prog_name="llm-patch-wiki-agent")
def main() -> None:
    """Top-level command group."""


@main.command(help="Compile wiki pages into LoRA adapters.")
@click.option(
    "--wiki-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing the wiki markdown content.",
)
@click.option(
    "--adapter-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory where compiled adapters will be stored.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Sakana Text-to-LoRA checkpoint directory.",
)
@click.option("--generator-device", default="cuda", show_default=True, help="PyTorch device.")
@click.option(
    "--pattern", "patterns", multiple=True, default=["*.md"], help="Wiki file glob patterns."
)
@click.option("--recursive/--no-recursive", default=True, help="Recurse into subdirectories.")
@click.option(
    "--aggregate-links/--no-aggregate-links",
    default=True,
    help="Enrich pages by following wiki links.",
)
def compile(
    wiki_dir: Path,
    adapter_dir: Path,
    checkpoint_dir: Path,
    generator_device: str,
    patterns: tuple[str, ...],
    recursive: bool,
    aggregate_links: bool,
) -> None:
    """Compile wiki knowledge into LoRA adapters."""

    def _callback() -> None:
        agent = _build_agent(
            adapter_dir=adapter_dir,
            wiki_dir=wiki_dir,
            checkpoint_dir=checkpoint_dir,
            patterns=patterns,
            recursive=recursive,
            aggregate_links=aggregate_links,
            generator_device=generator_device,
        )
        manifests = agent.compile()
        click.echo(f"Compiled {len(manifests)} adapter(s).")
        for manifest in manifests:
            click.echo(f"  {manifest.adapter_id}: {manifest.storage_uri}")

    _run_agent_action("compile", _callback)


@main.command(help="Run a single chat turn against the compiled wiki adapters.")
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing compiled adapters.",
)
@click.option("--model-id", required=True, help="Base HuggingFace model ID or local path.")
@click.option("--adapter-id", "adapter_ids", multiple=True, help="Specific adapter IDs to attach.")
@click.option("--device-map", default="auto", show_default=True, help="HuggingFace device map.")
@click.option("--dtype", default="float16", show_default=True, help="Torch dtype string.")
@click.option("--system", default=None, help="Optional system prompt.")
@click.argument("prompt")
def chat(
    adapter_dir: Path,
    model_id: str,
    adapter_ids: tuple[str, ...],
    device_map: str,
    dtype: str,
    system: str | None,
    prompt: str,
) -> None:
    """Run a one-shot chat request through the patched model."""

    def _callback() -> None:
        agent = _build_agent(
            adapter_dir=adapter_dir,
            model_id=model_id,
            model_device_map=device_map,
            model_dtype=dtype,
            system_prompt=system,
        )
        reply = agent.chat(prompt, adapter_ids=adapter_ids or None)
        click.echo(reply)

    _run_agent_action("chat", _callback)


@main.command(help="Print wiki-agent runtime info.")
@click.option(
    "--adapter-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory containing compiled adapters.",
)
@click.option(
    "--wiki-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing the wiki markdown content.",
)
@click.option("--model-id", default=None, help="Configured base model ID.")
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Configured T2L checkpoint directory.",
)
def info(
    adapter_dir: Path,
    wiki_dir: Path | None,
    model_id: str | None,
    checkpoint_dir: Path | None,
) -> None:
    """Print current wiki-agent configuration and adapter inventory."""

    def _callback() -> None:
        agent = _build_agent(
            adapter_dir=adapter_dir,
            wiki_dir=wiki_dir,
            model_id=model_id,
            checkpoint_dir=checkpoint_dir,
        )
        summary = agent.describe()
        click.echo(f"llm-patch-wiki-agent {__version__}")
        click.echo(f"Adapter dir: {summary.adapter_dir}")
        click.echo(f"Wiki dir: {summary.wiki_dir or '(not configured)'}")
        click.echo(f"Model id: {summary.model_id or '(not configured)'}")
        click.echo(f"Checkpoint dir: {summary.checkpoint_dir or '(not configured)'}")
        click.echo(f"Adapters: {summary.adapter_count}")
        if summary.adapter_ids:
            click.echo(f"Adapter IDs: {', '.join(summary.adapter_ids)}")

    _run_agent_action("info", _callback)


if __name__ == "__main__":  # pragma: no cover
    main()
