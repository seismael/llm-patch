"""CLI entry point for ``llm-patch-wiki-agent``."""

from __future__ import annotations

import signal
from pathlib import Path
from typing import TYPE_CHECKING

import click
from llm_patch_utils import DependencyError, LlmPatchError

from llm_patch_wiki_agent import __version__
from llm_patch_wiki_agent.agent import WikiAgent, WikiAgentConfig

if TYPE_CHECKING:
    pass


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


# ── Phase 1: compile daemon ───────────────────────────────────────────


@main.command(help="Run the wiki compile daemon (batch by default; --watch for live).")
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
    help="Directory where compiled adapters and metadata sidecars are stored.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Sakana Text-to-LoRA checkpoint directory.",
)
@click.option(
    "--watch/--once",
    default=False,
    help="Keep running and compile on filesystem changes (default: --once).",
)
@click.option("--generator-device", default="cuda", show_default=True)
@click.option("--pattern", "patterns", multiple=True, default=["*.md"])
@click.option("--recursive/--no-recursive", default=True)
@click.option("--aggregate-links/--no-aggregate-links", default=True)
def daemon(
    wiki_dir: Path,
    adapter_dir: Path,
    checkpoint_dir: Path,
    watch: bool,
    generator_device: str,
    patterns: tuple[str, ...],
    recursive: bool,
    aggregate_links: bool,
) -> None:
    """Compile a wiki into adapters and write metadata sidecars."""
    from llm_patch_wiki_agent.daemon import WikiCompileDaemon

    def _callback() -> None:
        config = WikiAgentConfig(
            adapter_dir=adapter_dir,
            wiki_dir=wiki_dir,
            checkpoint_dir=checkpoint_dir,
            source_patterns=patterns,
            recursive=recursive,
            aggregate_links=aggregate_links,
            generator_device=generator_device,
        )
        wiki_daemon = WikiCompileDaemon.from_config(config)

        if not watch:
            result = wiki_daemon.run_once()
            click.echo(f"Compiled {len(result.manifests)} adapter(s).")
            for manifest, metadata in zip(result.manifests, result.metadata, strict=True):
                click.echo(
                    f"  {manifest.adapter_id}  context_id={metadata.context_id}  "
                    f"→ {manifest.storage_uri}"
                )
            return

        result = wiki_daemon.run_once()
        click.echo(f"Initial pass: compiled {len(result.manifests)} adapter(s). Watching…")
        wiki_daemon.start()

        def _stop(_signum: int, _frame: object) -> None:
            click.echo("Stopping daemon…")
            wiki_daemon.stop()
            raise SystemExit(0)

        signal.signal(signal.SIGINT, _stop)
        try:
            signal.signal(signal.SIGTERM, _stop)
        except ValueError:
            # Windows: SIGTERM may be unavailable in some contexts.
            pass
        signal.pause() if hasattr(signal, "pause") else _block_forever()

    _run_agent_action("daemon", _callback)


def _block_forever() -> None:  # pragma: no cover - Windows-only fallback
    import time

    while True:
        time.sleep(3600)


# ── Phase 2: HTTP gateway ─────────────────────────────────────────────


@main.command(help="Run the FastAPI inference gateway with metadata-based routing.")
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option("--model-id", required=True, help="Base HuggingFace model ID or local path.")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8765, show_default=True, type=int)
@click.option("--device-map", default="auto", show_default=True)
@click.option("--dtype", default="float16", show_default=True)
@click.option("--reload/--no-reload", default=False)
def serve(
    adapter_dir: Path,
    model_id: str,
    host: str,
    port: int,
    device_map: str,
    dtype: str,
    reload: bool,
) -> None:
    """Launch the wiki-agent inference gateway (uvicorn)."""

    def _callback() -> None:
        try:
            import uvicorn
        except ImportError as exc:
            raise DependencyError(
                "uvicorn is not installed. Install with: pip install 'llm-patch-wiki-agent[server]'"
            ) from exc

        from llm_patch_wiki_agent.gateway import GatewayContext, create_app

        config = WikiAgentConfig(
            adapter_dir=adapter_dir,
            model_id=model_id,
            model_device_map=device_map,
            model_dtype=dtype,
        )
        context = GatewayContext.from_config(config)
        app = create_app(context)
        click.echo(f"Serving wiki-agent gateway on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port, reload=reload)

    _run_agent_action("serve", _callback)


# ── Phase 3: MCP server ───────────────────────────────────────────────


@main.command(help="Run the wiki-agent MCP server (exposes internalize_knowledge).")
@click.option(
    "--adapter-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--wiki-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
)
@click.option("--model-id", default=None)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"], case_sensitive=False),
    default="stdio",
    show_default=True,
)
def mcp(
    adapter_dir: Path,
    wiki_dir: Path | None,
    checkpoint_dir: Path | None,
    model_id: str | None,
    transport: str,
) -> None:
    """Launch the wiki-agent MCP server."""

    def _callback() -> None:
        try:
            import mcp  # noqa: F401
        except ImportError as exc:
            raise DependencyError(
                "mcp is not installed. Install with: pip install 'llm-patch-wiki-agent[mcp]'"
            ) from exc

        from llm_patch_wiki_agent.gateway import GatewayContext
        from llm_patch_wiki_agent.mcp_server import build_server

        config = WikiAgentConfig(
            adapter_dir=adapter_dir,
            wiki_dir=wiki_dir,
            checkpoint_dir=checkpoint_dir,
            model_id=model_id,
        )
        context = GatewayContext.from_config(config)
        server = build_server(context)
        click.echo(f"Starting wiki-agent MCP server (transport={transport})…")
        server.run(transport=transport)

    _run_agent_action("mcp", _callback)


if __name__ == "__main__":  # pragma: no cover
    main()
