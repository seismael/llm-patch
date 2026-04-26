"""Pydantic domain models for the llm_patch library."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# Module-level regex used by ``AdapterRef.parse`` (kept off the class
# because Pydantic treats ``_NAME`` attributes as private model state).
_ADAPTER_REF_RE = re.compile(
    r"^hub://(?P<namespace>[a-z0-9][a-z0-9._\-]*)/"
    r"(?P<name>[a-z0-9][a-z0-9._\-]*)"
    r"(?::(?P<version>[A-Za-z0-9._\-+]+))?$"
)


class DocumentContext(BaseModel):
    """Immutable representation of ingested knowledge.

    Attributes:
        document_id: Unique identifier derived from document filename stem.
        content: Raw document text content.
        metadata: Arbitrary metadata (source path, modified time, etc.).
    """

    model_config = {"frozen": True}

    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class AdapterManifest(BaseModel):
    """Tracks a generated LoRA adapter and its storage location.

    Manifest version 2 (additive) introduces optional fields used by the
    distributed adapter registry / "Adapter Market" use case. All new
    fields default to safe values so v1 manifests on disk continue to
    deserialize unchanged.

    Attributes:
        adapter_id: Unique identifier matching the source document_id.
        rank: LoRA rank (r) used for the adapter.
        target_modules: List of model modules targeted by the adapter.
        storage_uri: Path or URI to the adapter directory.
        created_at: UTC timestamp of adapter creation.
        manifest_version: Schema version (1 = legacy, 2 = registry-aware).
        namespace: Registry namespace (e.g. ``"acme/react-19-docs"``).
        version: Semantic version string (e.g. ``"v1.2.0"``).
        checksum_sha256: 64-char hex SHA-256 of the packed weights.
        base_model_compatibility: Compatible base model IDs.
        tags: Free-form discoverability labels.
        description: Human-readable summary.
    """

    adapter_id: str
    rank: int
    target_modules: list[str]
    storage_uri: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # ── v2 (registry-aware, all optional) ─────────────────────────────
    manifest_version: int = 1
    namespace: str | None = None
    version: str | None = None
    checksum_sha256: str | None = None
    base_model_compatibility: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    description: str | None = None

    @field_validator("checksum_sha256")
    @classmethod
    def _validate_checksum(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.fullmatch(r"[0-9a-f]{64}", v):
            raise ValueError("checksum_sha256 must be 64 lowercase hex characters")
        return v

    @field_validator("version")
    @classmethod
    def _validate_version(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Permissive SemVer: optional 'v' prefix, MAJOR.MINOR.PATCH plus optional pre/build.
        if not re.fullmatch(r"v?\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?", v):
            raise ValueError(
                "version must be SemVer (e.g. '1.2.0', 'v1.2.0', '1.2.0-rc1')"
            )
        return v

    @field_validator("namespace")
    @classmethod
    def _validate_namespace(cls, v: str | None) -> str | None:
        if v is None:
            return v
        # Allow ``owner/name`` slugs with [a-z0-9._-] segments.
        if not re.fullmatch(r"[a-z0-9][a-z0-9._\-]*/[a-z0-9][a-z0-9._\-]*", v):
            raise ValueError(
                "namespace must look like 'owner/name' using [a-z0-9._-]"
            )
        return v


class AdapterRef(BaseModel):
    """Parsed reference to a registry adapter.

    The canonical string form is ``hub://{namespace}/{name}:{version}``.
    ``namespace`` is the publisher slug (``owner``); ``name`` is the
    adapter slug. ``version`` defaults to ``"latest"`` when omitted.

    This is a value object — equality is structural, not identity-based.
    """

    model_config = {"frozen": True}

    namespace: str
    name: str
    version: str = "latest"

    @classmethod
    def parse(cls, uri: str) -> AdapterRef:
        """Parse a ``hub://owner/name[:version]`` URI.

        Args:
            uri: The reference string.

        Returns:
            A frozen :class:`AdapterRef`.

        Raises:
            ValueError: If *uri* does not match the grammar.
        """
        match = _ADAPTER_REF_RE.match(uri)
        if match is None:
            raise ValueError(
                f"Invalid adapter reference: {uri!r}. "
                "Expected 'hub://owner/name[:version]'."
            )
        return cls(
            namespace=match["namespace"],
            name=match["name"],
            version=match["version"] or "latest",
        )

    def to_uri(self) -> str:
        """Serialize back to the canonical ``hub://`` form."""
        return f"hub://{self.namespace}/{self.name}:{self.version}"

    @property
    def adapter_id(self) -> str:
        """Stable local identifier derived from the reference."""
        return f"{self.namespace}__{self.name}__{self.version}".replace("/", "__")


# ── Model Handles ─────────────────────────────────────────────────────


class ModelHandle(BaseModel):
    """Lightweight wrapper around a loaded model + tokenizer.

    The ``model`` and ``tokenizer`` fields hold runtime objects (not
    serializable) so they use ``Any``.  Everything else is plain data.
    """

    model_config = {"frozen": True, "arbitrary_types_allowed": True}

    model: Any = Field(exclude=True)
    tokenizer: Any = Field(exclude=True)
    base_model_id: str
    attached_adapters: tuple[str, ...] = ()
    device: str = "cpu"


# ── Chat / Generation ─────────────────────────────────────────────────


class ChatRole(StrEnum):
    """Standard chat-message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """A single message in a multi-turn conversation."""

    model_config = {"frozen": True}

    role: ChatRole
    content: str


class GenerationOptions(BaseModel):
    """Knobs exposed to callers for text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


class ChatResponse(BaseModel):
    """Response from ``IAgentRuntime.chat``."""

    message: ChatMessage
    usage: dict[str, int] = Field(default_factory=dict)


# ── Data-Source Registry ──────────────────────────────────────────────


class DataSourceDescriptor(BaseModel):
    """Metadata entry describing an available data source type."""

    source_type: str
    description: str
    config_schema: dict[str, Any] = Field(default_factory=dict)
