"""Pydantic configuration models for llm_patch components."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


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
