"""Top-level distribution commands: ``llm-patch push`` / ``llm-patch pull``.

Both commands route through :class:`IAdapterRegistryClient`. The engine
itself ships **no** concrete clients; CLI users must wire one (HF Hub,
S3, custom REST hub) via the ``LLM_PATCH_PLUGIN_REGISTRY`` environment
variable (canonical) or its deprecated alias ``LLM_PATCH_REGISTRY``
(removal scheduled for v2.0.0; see ``docs/ROADMAP.md``).

Target URI grammar
------------------
* ``hub://owner/name[:version]``  — canonical Adapter Market URI
* ``hf://owner/repo``             — Hugging Face repository (deferred)
* ``s3://bucket/key``             — S3 bucket (deferred)

This module is import-light: the heavy registry implementation stays
behind a lazy import so ``llm-patch --help`` is fast.
"""

from __future__ import annotations

import json as _json
import warnings
from pathlib import Path
from typing import Final

import click

from llm_patch_shared import RegistryUnavailableError

CANONICAL_REGISTRY_ENV: Final = "LLM_PATCH_PLUGIN_REGISTRY"
LEGACY_REGISTRY_ENV: Final = "LLM_PATCH_REGISTRY"


def _emit(payload: dict[str, object], *, as_json: bool, quiet: bool) -> None:
    if as_json:
        click.echo(_json.dumps(payload, sort_keys=True))
        return
    if quiet:
        return
    for key, value in payload.items():
        click.echo(f"{key}: {value}")


def _resolve_registry_spec() -> str:
    """Return the registry factory spec, preferring the canonical env var.

    Reads ``LLM_PATCH_PLUGIN_REGISTRY`` first; falls back to the
    deprecated ``LLM_PATCH_REGISTRY`` alias with a one-time warning.
    Removal of the alias is scheduled for v2.0.0 (see ROADMAP).
    """
    import os

    from llm_patch.core.project_config import ProjectConfig

    spec = os.environ.get(CANONICAL_REGISTRY_ENV, "").strip()
    if spec:
        return spec
    # Project-level fallback: only honored if no env var is set.
    config = ProjectConfig.find_and_load()
    if config is not None and config.registry.plugin:
        return config.registry.plugin
    legacy = os.environ.get(LEGACY_REGISTRY_ENV, "").strip()
    if legacy:
        warnings.warn(
            f"{LEGACY_REGISTRY_ENV} is deprecated and will be removed in v2.0.0; "
            f"use {CANONICAL_REGISTRY_ENV} instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return legacy
    raise RegistryUnavailableError(
        f"No registry client configured. Set {CANONICAL_REGISTRY_ENV}="
        "'module:factory' or see docs/REGISTRY_PROTOCOL.md."
    )


def _load_registry() -> object:
    """Return the user-configured registry client.

    The engine deliberately ships no concrete client. Users plug one in
    by setting :data:`CANONICAL_REGISTRY_ENV` (or the deprecated
    :data:`LEGACY_REGISTRY_ENV` alias) to a ``module:callable`` factory
    that returns an
    :class:`~llm_patch.core.interfaces.IAdapterRegistryClient`.
    """
    import importlib

    spec = _resolve_registry_spec()
    if ":" not in spec:
        raise RegistryUnavailableError(
            f"{CANONICAL_REGISTRY_ENV} must be 'module:callable', got {spec!r}"
        )
    module_path, _, attr = spec.partition(":")
    module = importlib.import_module(module_path)
    factory = getattr(module, attr)
    return factory()


@click.command("push")
@click.argument(
    "adapter_path",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--target", required=True, help="Destination URI (hub://, hf://, s3://).")
@click.option("--quiet", is_flag=True, help="Suppress non-essential output.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-parseable JSON.")
def push(adapter_path: Path, target: str, quiet: bool, as_json: bool) -> None:
    """Upload a locally-compiled adapter to a registry target."""
    from llm_patch.core.models import AdapterRef

    try:
        registry = _load_registry()
    except RegistryUnavailableError as exc:
        raise click.ClickException(str(exc)) from exc

    if target.startswith("hub://"):
        ref = AdapterRef.parse(target)
    elif target.startswith(("hf://", "s3://")):
        # Deferred: the engine recognizes the schemes but defers actual
        # HF / S3 transport to user-provided clients. See
        # ``docs/REGISTRY_PROTOCOL.md``.
        raise click.ClickException(
            f"Target scheme not yet implemented in core: {target!r}. "
            "Wire a client via LLM_PATCH_PLUGIN_REGISTRY (see docs/REGISTRY_PROTOCOL.md)."
        )
    else:
        raise click.ClickException(
            f"Unsupported target scheme: {target!r}. "
            "Expected hub://, hf://, or s3://."
        )

    from llm_patch.core.interfaces import IAdapterRegistryClient

    if not isinstance(registry, IAdapterRegistryClient):
        raise click.ClickException(
            "Configured registry does not implement IAdapterRegistryClient."
        )

    manifest = registry.push(adapter_path.name, ref)
    _emit(
        {
            "pushed": ref.to_uri(),
            "adapter_id": manifest.adapter_id,
            "checksum_sha256": manifest.checksum_sha256 or "",
        },
        as_json=as_json,
        quiet=quiet,
    )


@click.command("pull")
@click.argument("ref")
@click.option("--quiet", is_flag=True, help="Suppress non-essential output.")
@click.option("--json", "as_json", is_flag=True, help="Emit machine-parseable JSON.")
def pull(ref: str, quiet: bool, as_json: bool) -> None:
    """Download an adapter from a registry target.

    REF must be a ``hub://owner/name[:version]`` URI.
    """
    from llm_patch.core.models import AdapterRef

    try:
        registry = _load_registry()
    except RegistryUnavailableError as exc:
        raise click.ClickException(str(exc)) from exc

    parsed = AdapterRef.parse(ref)

    from llm_patch.core.interfaces import IAdapterRegistryClient

    if not isinstance(registry, IAdapterRegistryClient):
        raise click.ClickException(
            "Configured registry does not implement IAdapterRegistryClient."
        )

    manifest = registry.pull(parsed)
    _emit(
        {
            "pulled": parsed.to_uri(),
            "adapter_id": manifest.adapter_id,
            "storage_uri": manifest.storage_uri,
            "checksum_sha256": manifest.checksum_sha256 or "",
        },
        as_json=as_json,
        quiet=quiet,
    )


@click.group("hub")
def hub() -> None:
    """Read-only registry discovery commands."""


@hub.command("search")
@click.argument("query")
@click.option("--limit", type=int, default=10, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-parseable JSON.")
def hub_search(query: str, limit: int, as_json: bool) -> None:
    """Search the configured registry for adapters."""
    try:
        registry = _load_registry()
    except RegistryUnavailableError as exc:
        raise click.ClickException(str(exc)) from exc

    from llm_patch.core.interfaces import IAdapterRegistryClient

    if not isinstance(registry, IAdapterRegistryClient):
        raise click.ClickException(
            "Configured registry does not implement IAdapterRegistryClient."
        )

    results = registry.search(query, limit=limit)
    if as_json:
        click.echo(_json.dumps([r.model_dump(mode="json") for r in results]))
        return
    if not results:
        click.echo("No matches.")
        return
    for r in results:
        click.echo(f"{r.adapter_id}\t{r.namespace or '-'}\t{r.version or '-'}")


@hub.command("info")
@click.argument("ref")
@click.option("--json", "as_json", is_flag=True)
def hub_info(ref: str, as_json: bool) -> None:
    """Resolve a ``hub://`` reference to its manifest (no download)."""
    from llm_patch.core.models import AdapterRef

    try:
        registry = _load_registry()
    except RegistryUnavailableError as exc:
        raise click.ClickException(str(exc)) from exc

    from llm_patch.core.interfaces import IAdapterRegistryClient

    if not isinstance(registry, IAdapterRegistryClient):
        raise click.ClickException(
            "Configured registry does not implement IAdapterRegistryClient."
        )

    manifest = registry.resolve(AdapterRef.parse(ref))
    if as_json:
        click.echo(_json.dumps(manifest.model_dump(mode="json")))
        return
    click.echo(f"adapter_id: {manifest.adapter_id}")
    click.echo(f"namespace: {manifest.namespace}")
    click.echo(f"version: {manifest.version}")
    click.echo(f"description: {manifest.description or ''}")


__all__ = ["hub", "pull", "push"]
