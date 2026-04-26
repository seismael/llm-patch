"""Model and agent subcommands for ``llm-patch model``.

Provides commands to load a base model, attach adapters, and run
interactive chat or single-prompt generation.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
def model() -> None:
    """Model loading, adapter attachment, and inference.

    Load a HuggingFace model, attach compiled LoRA adapters, and
    interact via chat or single-prompt generation.
    """


@model.command()
@click.option("--model-id", required=True, help="HuggingFace model ID or local path.")
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing compiled adapters.",
)
@click.option("--adapter-id", "adapter_ids", multiple=True, help="Specific adapter IDs to attach.")
@click.option("--device", default="auto", help="Device map (default: auto).")
@click.argument("prompt")
def generate(
    model_id: str,
    adapter_dir: Path,
    adapter_ids: tuple[str, ...],
    device: str,
    prompt: str,
) -> None:
    """Generate text from a prompt using a patched model."""
    from llm_patch.attach import HFModelProvider, PeftAdapterLoader
    from llm_patch.core.config import StorageConfig
    from llm_patch.pipelines.use import UsePipeline
    from llm_patch.runtime.agent import PeftAgentRuntime
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

    repo = LocalSafetensorsRepository(StorageConfig(output_dir=adapter_dir))
    pipeline = UsePipeline(
        model_provider=HFModelProvider(),
        adapter_loader=PeftAdapterLoader(),
        repository=repo,
    )
    ids = list(adapter_ids) if adapter_ids else None
    handle = pipeline.load_and_attach(model_id, adapter_ids=ids, device_map=device)
    runtime = PeftAgentRuntime(handle)
    result = runtime.generate(prompt)
    click.echo(result)


@model.command()
@click.option(
    "--model-id",
    default=None,
    help="HuggingFace model ID or local path (default: [runtime].base_model from .llm-patch.toml).",
)
@click.option(
    "--adapter-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory containing compiled adapters (default: [compile].output from .llm-patch.toml).",
)
@click.option("--adapter-id", "adapter_ids", multiple=True, help="Specific adapter IDs to attach.")
@click.option("--device", default="auto", help="Device map (default: auto).")
@click.option("--system", default=None, help="System prompt for the conversation.")
def chat(
    model_id: str | None,
    adapter_dir: Path | None,
    adapter_ids: tuple[str, ...],
    device: str,
    system: str | None,
) -> None:
    """Interactive chat with a patched model (Ctrl-C to exit)."""
    from llm_patch.attach import HFModelProvider, PeftAdapterLoader
    from llm_patch.core.config import StorageConfig
    from llm_patch.core.project_config import ProjectConfig
    from llm_patch.pipelines.use import UsePipeline
    from llm_patch.runtime.agent import PeftAgentRuntime
    from llm_patch.runtime.session import ChatSession
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

    config = ProjectConfig.find_and_load()
    if model_id is None:
        model_id = config.runtime.base_model if config else None
    if adapter_dir is None:
        adapter_dir = config.compile.output if config else None
    if model_id is None:
        raise click.ClickException(
            "Missing --model-id. Provide --model-id or set [runtime] base_model "
            "in .llm-patch.toml."
        )
    if adapter_dir is None:
        raise click.ClickException(
            "Missing --adapter-dir. Provide --adapter-dir or set [compile] output "
            "in .llm-patch.toml."
        )
    if not adapter_dir.exists() or not adapter_dir.is_dir():
        raise click.ClickException(f"--adapter-dir does not exist: {adapter_dir}")

    repo = LocalSafetensorsRepository(StorageConfig(output_dir=adapter_dir))
    pipeline = UsePipeline(
        model_provider=HFModelProvider(),
        adapter_loader=PeftAdapterLoader(),
        repository=repo,
    )
    ids = list(adapter_ids) if adapter_ids else None
    handle = pipeline.load_and_attach(model_id, adapter_ids=ids, device_map=device)
    runtime = PeftAgentRuntime(handle)
    session = ChatSession(runtime, system_prompt=system)

    click.echo("Chat session started (Ctrl-C to exit).\n")
    try:
        while True:
            user_input = click.prompt("You", prompt_suffix="> ")
            if not user_input.strip():
                continue
            reply = session.say(user_input)
            click.echo(f"Assistant> {reply}\n")
    except (KeyboardInterrupt, EOFError):
        click.echo("\nSession ended.")


@model.command("info")
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing compiled adapters.",
)
def info(adapter_dir: Path) -> None:
    """List adapters available in an adapter directory."""
    from llm_patch.core.config import StorageConfig
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

    repo = LocalSafetensorsRepository(StorageConfig(output_dir=adapter_dir))
    adapters = repo.list_adapters()
    if not adapters:
        click.echo("No adapters found.")
        return
    click.echo(f"{len(adapters)} adapter(s):")
    for m in adapters:
        click.echo(f"  {m.adapter_id} — rank={m.rank}, path={m.storage_uri}")
