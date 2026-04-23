"""Server subcommand for ``llm-patch serve``.

Launches the FastAPI HTTP server for the llm-patch API.
"""

from __future__ import annotations

from pathlib import Path

import click


@click.command()
@click.option("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
@click.option("--port", default=8000, type=int, help="Bind port (default: 8000).")
@click.option("--model-id", default=None, help="Pre-load a HuggingFace model ID.")
@click.option(
    "--adapter-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=None,
    help="Directory containing compiled adapters.",
)
@click.option("--reload/--no-reload", default=False, help="Enable auto-reload (dev mode).")
def serve(
    host: str,
    port: int,
    model_id: str | None,
    adapter_dir: Path | None,
    reload: bool,
) -> None:
    """Start the llm-patch HTTP API server.

    Requires the ``server`` extra: pip install 'llm-patch[server]'
    """
    try:
        import uvicorn
    except ImportError:
        raise click.ClickException(
            "Server requires uvicorn and fastapi. Install with: pip install 'llm-patch[server]'"
        )

    # Pass config via environment so the FastAPI app can pick it up
    import os

    if model_id:
        os.environ["LLM_PATCH_MODEL_ID"] = model_id
    if adapter_dir:
        os.environ["LLM_PATCH_ADAPTER_DIR"] = str(adapter_dir)

    uvicorn.run(
        "llm_patch.server.app:app",
        host=host,
        port=port,
        reload=reload,
    )
