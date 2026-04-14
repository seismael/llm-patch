#!/usr/bin/env python
"""Research Papers → Wiki → LoRA demo pipeline.

Two modes:
  batch  — Scan all existing wiki pages and compile adapters for each.
  watch  — Start a live watcher and generate adapters on every wiki change.

Usage:
    python research_pipeline.py batch --wiki-dir wiki/ --output-dir adapters/
    python research_pipeline.py watch --wiki-dir wiki/ --output-dir adapters/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure the src directory is importable when running as a script.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_patch.core.config import StorageConfig, WatcherConfig
from llm_patch.core.interfaces import IAdapterRepository, IWeightGenerator
from llm_patch.core.models import AdapterManifest, DocumentContext
from llm_patch.orchestrator import KnowledgeFusionOrchestrator
from llm_patch.sources.wiki_source import WikiKnowledgeSource

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MockWeightGenerator — used when torch / Sakana checkpoint unavailable
# ---------------------------------------------------------------------------

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False


class MockWeightGenerator(IWeightGenerator):
    """Generates deterministic placeholder weight dicts (no GPU required).

    Useful for testing the full pipeline when torch is broken or no Sakana
    checkpoint is available.
    """

    def __init__(self, *, rank: int = 8, hidden: int = 256) -> None:
        self._rank = rank
        self._hidden = hidden

    def generate(self, context: DocumentContext) -> dict[str, Any]:
        """Return placeholder weight tensors keyed by PEFT-format names."""
        if HAS_TORCH:
            return {
                "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(
                    self._rank, self._hidden
                ),
                "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.zeros(
                    self._hidden, self._rank
                ),
            }
        # Torch unavailable — return raw nested lists as stand-in
        return {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": [
                [0.0] * self._hidden for _ in range(self._rank)
            ],
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": [
                [0.0] * self._rank for _ in range(self._hidden)
            ],
        }

    def get_peft_config(self) -> dict[str, Any]:
        """Return a plain-dict PEFT config (not a real LoraConfig)."""
        return {
            "r": self._rank,
            "target_modules": ["q_proj", "v_proj"],
            "lora_alpha": self._rank * 2,
            "lora_dropout": 0.0,
            "bias": "none",
            "peft_type": "LORA",
        }


# ---------------------------------------------------------------------------
# MockAdapterRepository — filesystem-free stand-in
# ---------------------------------------------------------------------------


class MockAdapterRepository(IAdapterRepository):
    """In-memory adapter repository for demonstration without safetensors."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._store: dict[str, tuple[dict[str, Any], Any]] = {}

    def save(
        self,
        adapter_id: str,
        weights: dict[str, Any],
        peft_config: Any,
    ) -> AdapterManifest:
        self._store[adapter_id] = (weights, peft_config)
        storage_uri = str(self._output_dir / adapter_id)
        logger.info("  [mock-repo] Stored adapter '%s' → %s", adapter_id, storage_uri)

        rank = peft_config.get("r", 8) if isinstance(peft_config, dict) else 8
        target_modules = (
            peft_config.get("target_modules", []) if isinstance(peft_config, dict) else []
        )
        return AdapterManifest(
            adapter_id=adapter_id,
            rank=rank,
            target_modules=target_modules,
            storage_uri=storage_uri,
        )

    def load(self, adapter_id: str) -> dict[str, Any]:
        return self._store[adapter_id][0]

    def exists(self, adapter_id: str) -> bool:
        return adapter_id in self._store

    def list_adapters(self) -> list[AdapterManifest]:
        return [
            AdapterManifest(
                adapter_id=aid,
                rank=8,
                target_modules=["q_proj", "v_proj"],
                storage_uri=str(self._output_dir / aid),
            )
            for aid in self._store
        ]

    def delete(self, adapter_id: str) -> None:
        self._store.pop(adapter_id, None)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research Papers → Wiki → LoRA adapter pipeline demo.",
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    batch = sub.add_parser("batch", help="Compile adapters for all existing wiki pages.")
    batch.add_argument("--wiki-dir", type=Path, required=True, help="Wiki output directory.")
    batch.add_argument("--output-dir", type=Path, default=Path("adapters"), help="Adapter output.")
    batch.add_argument("--aggregate", action="store_true", help="Follow wikilinks during scan.")

    watch = sub.add_parser("watch", help="Watch wiki directory and compile on change.")
    watch.add_argument("--wiki-dir", type=Path, required=True, help="Wiki output directory.")
    watch.add_argument("--output-dir", type=Path, default=Path("adapters"), help="Adapter output.")
    watch.add_argument("--aggregate", action="store_true", help="Follow wikilinks during scan.")

    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    args = build_parser().parse_args(argv)
    wiki_dir: Path = args.wiki_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not wiki_dir.exists():
        logger.error("Wiki directory does not exist: %s", wiki_dir)
        sys.exit(1)

    watcher_config = WatcherConfig(directory=wiki_dir)
    storage_config = StorageConfig(output_dir=output_dir)  # noqa: F841

    source = WikiKnowledgeSource(watcher_config, aggregate=args.aggregate)
    generator = MockWeightGenerator()
    repository = MockAdapterRepository(output_dir)

    orchestrator = KnowledgeFusionOrchestrator(
        source=source,
        generator=generator,
        repository=repository,
    )

    if args.mode == "batch":
        logger.info("Batch mode — scanning %s", wiki_dir)
        manifests = orchestrator.compile_all()
        logger.info("Compiled %d adapters:", len(manifests))
        for m in manifests:
            logger.info("  %s → %s (rank=%d)", m.adapter_id, m.storage_uri, m.rank)
    else:
        logger.info("Watch mode — monitoring %s (Ctrl-C to stop)", wiki_dir)
        with orchestrator:
            try:
                import time

                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Interrupted — shutting down.")


if __name__ == "__main__":
    main()
