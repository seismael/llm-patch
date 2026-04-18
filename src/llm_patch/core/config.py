"""Pydantic configuration models for llm_patch components."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


# ── Generator / Watcher / Storage (existing) ──────────────────────────


class GeneratorConfig(BaseModel):
    """Configuration for a weight generator backend.

    Attributes:
        checkpoint_dir: Directory containing hypermod.pt, args.yaml, adapter_config.json.
        device: PyTorch device string (e.g. 'cuda', 'cpu', 'cuda:0').
    """

    checkpoint_dir: Path
    device: str = "cuda"


class WatcherConfig(BaseModel):
    """Configuration for a document directory watcher.

    Attributes:
        directory: Directory to monitor for document changes.
        patterns: Glob patterns for files to watch.
        recursive: Whether to monitor subdirectories.
        debounce_seconds: Minimum interval between callbacks for the same file.
    """

    directory: Path
    patterns: list[str] = Field(default_factory=lambda: ["*.md"])
    recursive: bool = True
    debounce_seconds: float = 0.5


class StorageConfig(BaseModel):
    """Configuration for an adapter storage backend.

    Attributes:
        output_dir: Directory where adapter subdirectories will be created.
    """

    output_dir: Path


class WikiConfig(BaseModel):
    """Configuration for the LLM Wiki management layer.

    Attributes:
        base_dir: Root directory containing raw/ and wiki/ directories.
        schema_path: Path to the CLAUDE.md-style wiki schema file.  ``None`` uses defaults.
        obsidian: Whether to initialise the wiki as an Obsidian vault.
    """

    base_dir: Path
    schema_path: Path | None = None
    obsidian: bool = False


# ── Data Source Configs (discriminated union) ─────────────────────────


class MarkdownSourceConfig(BaseModel):
    """Config for the markdown directory data source."""

    type: Literal["markdown"] = "markdown"
    directory: Path
    patterns: list[str] = Field(default_factory=lambda: ["*.md"])
    recursive: bool = True


class WikiSourceConfig(BaseModel):
    """Config for the wiki data source."""

    type: Literal["wiki"] = "wiki"
    directory: Path
    patterns: list[str] = Field(default_factory=lambda: ["*.md"])
    recursive: bool = True
    aggregate: bool = False


class PdfSourceConfig(BaseModel):
    """Config for the PDF data source."""

    type: Literal["pdf"] = "pdf"
    directory: Path
    recursive: bool = True


class JsonlSourceConfig(BaseModel):
    """Config for the JSONL data source."""

    type: Literal["jsonl"] = "jsonl"
    path: Path
    text_field: str = "text"
    id_field: str = "id"


class HttpSourceConfig(BaseModel):
    """Config for the HTTP API data source."""

    type: Literal["http"] = "http"
    url: str
    headers: dict[str, str] = Field(default_factory=dict)
    text_path: str = "text"
    id_path: str = "id"


DataSourceConfig = (
    MarkdownSourceConfig
    | WikiSourceConfig
    | PdfSourceConfig
    | JsonlSourceConfig
    | HttpSourceConfig
)


# ── Model / Attach / Agent / Server ──────────────────────────────────


class ModelSpec(BaseModel):
    """Specification for loading a base model.

    Attributes:
        model_id: HuggingFace model ID or local path.
        dtype: Torch dtype string (``'float16'``, ``'bfloat16'``, ``'float32'``).
        device_map: Accelerate device-map hint (``'auto'``, ``'cpu'``, ``'cuda:0'``).
        trust_remote_code: Allow custom code in model repo.
    """

    model_id: str
    dtype: str = "float16"
    device_map: str = "auto"
    trust_remote_code: bool = False


class AttachConfig(BaseModel):
    """Configuration for attaching an adapter onto a model."""

    adapter_dir: Path
    adapter_name: str | None = None


class AgentConfig(BaseModel):
    """Configuration for an agent runtime session."""

    model_spec: ModelSpec
    adapter_ids: list[str] = Field(default_factory=list)
    generation_max_new_tokens: int = 256
    generation_temperature: float = 0.7
    system_prompt: str | None = None


class ServerConfig(BaseModel):
    """Configuration for the HTTP server frontend."""

    host: str = "127.0.0.1"
    port: int = 8000
    adapter_dir: Path = Path("adapters")
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
