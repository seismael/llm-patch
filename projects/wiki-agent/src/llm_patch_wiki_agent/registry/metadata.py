"""Adapter metadata model + filesystem-backed sidecar registry.

Each adapter gets a ``<adapter_id>.meta.json`` sidecar file co-located with
its compiled artifact. This keeps the engine's :class:`llm_patch.AdapterManifest`
generic while allowing the wiki-agent gateway to attach routing-relevant
metadata (context_id, tags, summary).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_patch_shared import ConfigurationError, ResourceNotFoundError
from pydantic import BaseModel, Field

_SIDECAR_SUFFIX = ".meta.json"


class AdapterMetadata(BaseModel):
    """Wiki-agent-side metadata describing a compiled adapter.

    The engine's :class:`llm_patch.AdapterManifest` is intentionally generic.
    This model adds the routing/context information the gateway needs without
    modifying the engine.

    Attributes:
        adapter_id: Must match the adapter_id of the engine ``AdapterManifest``.
        context_id: Optional logical key used for exact-match routing
            (e.g. ``"api-v2-auth"``).
        tags: Free-form labels (e.g. ``("api", "v2", "auth")``).
        summary: Optional short description used for embedding-based routers.
        source_path: Optional original source file path used to compile the
            adapter, for traceability.
        created_at: UTC timestamp written when the sidecar was first saved.
    """

    model_config = {"frozen": True}

    adapter_id: str
    context_id: str | None = None
    tags: tuple[str, ...] = ()
    summary: str | None = None
    source_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class SidecarMetadataRegistry:
    """Directory-backed registry storing one JSON sidecar per adapter.

    The registry never mutates the underlying ``.safetensors`` artifacts — it
    only writes/reads ``<adapter_id>.meta.json`` files under ``directory``.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = Path(directory)

    @property
    def directory(self) -> Path:
        return self._directory

    def save(self, metadata: AdapterMetadata) -> Path:
        """Persist ``metadata`` and return the sidecar path."""
        if not metadata.adapter_id:
            raise ConfigurationError("AdapterMetadata.adapter_id must be non-empty.")
        self._directory.mkdir(parents=True, exist_ok=True)
        sidecar = self._sidecar_for(metadata.adapter_id)
        payload = metadata.model_dump(mode="json")
        sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return sidecar

    def load(self, adapter_id: str) -> AdapterMetadata:
        """Load ``adapter_id``'s metadata or raise :class:`ResourceNotFoundError`."""
        sidecar = self._sidecar_for(adapter_id)
        if not sidecar.exists():
            raise ResourceNotFoundError(
                f"No metadata sidecar found for adapter_id={adapter_id!r} in {self._directory}."
            )
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigurationError(
                f"Corrupt metadata sidecar for adapter_id={adapter_id!r}: {sidecar}"
            ) from exc
        return AdapterMetadata.model_validate(payload)

    def list_all(self) -> tuple[AdapterMetadata, ...]:
        """Return every readable sidecar in the directory, deterministically ordered."""
        if not self._directory.exists():
            return ()
        records: list[AdapterMetadata] = []
        for path in sorted(self._directory.glob(f"*{_SIDECAR_SUFFIX}")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                records.append(AdapterMetadata.model_validate(payload))
            except (OSError, json.JSONDecodeError, ValueError):
                # Skip malformed sidecars; they are surfaced via ``load`` on demand.
                continue
        return tuple(records)

    def delete(self, adapter_id: str) -> None:
        sidecar = self._sidecar_for(adapter_id)
        if not sidecar.exists():
            raise ResourceNotFoundError(
                f"No metadata sidecar to delete for adapter_id={adapter_id!r}."
            )
        sidecar.unlink()

    def exists(self, adapter_id: str) -> bool:
        return self._sidecar_for(adapter_id).exists()

    def _sidecar_for(self, adapter_id: str) -> Path:
        return self._directory / f"{adapter_id}{_SIDECAR_SUFFIX}"

    def __repr__(self) -> str:
        return f"SidecarMetadataRegistry(directory={self._directory!s})"

    # ── Convenience helpers ───────────────────────────────────────────

    def upsert_from_payload(self, adapter_id: str, payload: dict[str, Any]) -> AdapterMetadata:
        """Save metadata derived from a free-form payload (e.g. frontmatter)."""
        metadata = AdapterMetadata(
            adapter_id=adapter_id,
            context_id=payload.get("context_id"),
            tags=tuple(payload.get("tags", ())),
            summary=payload.get("summary"),
            source_path=payload.get("source_path"),
        )
        self.save(metadata)
        return metadata
