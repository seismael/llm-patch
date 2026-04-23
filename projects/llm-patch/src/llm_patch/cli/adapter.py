"""Adapter subcommands for ``llm-patch adapter``.

Wraps the core adapter pipeline (``KnowledgeFusionOrchestrator``) to
compile Markdown documents into LoRA adapter weights.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import click

from llm_patch.core.interfaces import IAdapterRepository, IWeightGenerator

if TYPE_CHECKING:
    from llm_patch.orchestrator import KnowledgeFusionOrchestrator

# ── Helpers ────────────────────────────────────────────────────────────


def _resolve_source(source_dir: Path, patterns: list[str], recursive: bool) -> object:
    """Create a knowledge source from the given directory."""
    from llm_patch.core.config import WatcherConfig
    from llm_patch.sources.markdown_watcher import MarkdownDirectoryWatcher

    cfg = WatcherConfig(directory=source_dir, patterns=patterns, recursive=recursive)
    return MarkdownDirectoryWatcher(cfg)


def _resolve_generator(checkpoint_dir: Path | None, device: str) -> IWeightGenerator:
    """Create a weight generator — requires a T2L checkpoint."""
    if checkpoint_dir is None:
        raise click.ClickException(
            "Missing --checkpoint-dir. Provide a T2L hypernetwork checkpoint directory."
        )

    from llm_patch.core.config import GeneratorConfig
    from llm_patch.generators.sakana_t2l import SakanaT2LGenerator

    cfg = GeneratorConfig(checkpoint_dir=checkpoint_dir, device=device)
    return SakanaT2LGenerator(cfg)


def _resolve_repository(output_dir: Path) -> IAdapterRepository:
    """Create a storage backend for persisting adapters."""
    from llm_patch.core.config import StorageConfig
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

    cfg = StorageConfig(output_dir=output_dir)
    return LocalSafetensorsRepository(cfg)


def _build_orchestrator(
    source_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path | None,
    device: str,
    patterns: list[str],
    recursive: bool,
) -> KnowledgeFusionOrchestrator:
    """Assemble the full adapter pipeline."""
    from llm_patch.orchestrator import KnowledgeFusionOrchestrator

    source = _resolve_source(source_dir, patterns, recursive)
    generator = _resolve_generator(checkpoint_dir, device)
    repository = _resolve_repository(output_dir)
    return KnowledgeFusionOrchestrator(source=source, generator=generator, repository=repository)


def _pause_for_watch_loop() -> None:
    """Block until interrupted, using ``signal.pause`` when available."""
    import signal

    pause = getattr(signal, "pause", None)
    if pause is None:
        raise AttributeError("signal.pause is unavailable on this platform")
    pause()


# ── Top-level ``adapter`` group ────────────────────────────────────────


@click.group()
def adapter() -> None:
    """LoRA adapter generation pipeline.

    Convert Markdown documents into LoRA adapter weights using the
    Sakana AI Text-to-LoRA hypernetwork.
    """


@adapter.command()
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing source Markdown documents.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for storing generated adapter weights.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="T2L hypernetwork checkpoint directory.",
)
@click.option("--device", default="cuda", help="PyTorch device (default: cuda).")
@click.option("--pattern", "patterns", multiple=True, default=["*.md"], help="File glob patterns.")
@click.option("--recursive/--no-recursive", default=True, help="Recurse into subdirs.")
def compile(
    source_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path | None,
    device: str,
    patterns: tuple[str, ...],
    recursive: bool,
) -> None:
    """Batch compile all documents into adapter weights."""
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = _build_orchestrator(
        source_dir, output_dir, checkpoint_dir, device, list(patterns), recursive
    )
    manifests = orchestrator.compile_all()
    click.echo(f"Compiled {len(manifests)} adapter(s).")
    for m in manifests:
        click.echo(f"  {m.adapter_id}: {m.storage_uri}")


@adapter.command()
@click.option(
    "--source-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory to watch for document changes.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
    help="Directory for storing generated adapter weights.",
)
@click.option(
    "--checkpoint-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="T2L hypernetwork checkpoint directory.",
)
@click.option("--device", default="cuda", help="PyTorch device (default: cuda).")
@click.option("--pattern", "patterns", multiple=True, default=["*.md"], help="File glob patterns.")
@click.option("--recursive/--no-recursive", default=True, help="Recurse into subdirs.")
def watch(
    source_dir: Path,
    output_dir: Path,
    checkpoint_dir: Path | None,
    device: str,
    patterns: tuple[str, ...],
    recursive: bool,
) -> None:
    """Watch a directory and compile adapters on change (Ctrl-C to stop)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    orchestrator = _build_orchestrator(
        source_dir, output_dir, checkpoint_dir, device, list(patterns), recursive
    )
    click.echo(f"Watching {source_dir} for changes … (Ctrl-C to stop)")
    try:
        with orchestrator:
            _pause_for_watch_loop()
    except (KeyboardInterrupt, AttributeError):
        # AttributeError: signal.pause not available on Windows
        pass
    finally:
        click.echo("Stopped.")


@adapter.command()
@click.option(
    "--output-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Adapter storage directory to inspect.",
)
def status(output_dir: Path) -> None:
    """List adapters in the output directory."""
    repository = _resolve_repository(output_dir)
    adapters = repository.list_adapters()
    if not adapters:
        click.echo("No adapters found.")
        return
    click.echo(f"{len(adapters)} adapter(s):")
    for m in adapters:
        click.echo(f"  {m.adapter_id} — rank={m.rank}, path={m.storage_uri}")
