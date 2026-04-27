#!/usr/bin/env python
"""End-to-end runner: raw papers → wiki simulation → adapter generation → validation.

Since the LLM Wiki Agent (SamurAIGPT/llm-wiki-agent) is an external process,
this script *simulates* the wiki ingestion step by copying raw paper markdown
files into a wiki-style directory structure with added wiki metadata.

Usage:
    python examples/e2e/run_e2e.py
    python examples/e2e/run_e2e.py --raw-dir examples/data/papers \
        --wiki-dir examples/e2e/wiki --output-dir examples/e2e/adapters
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

# Allow sibling-script imports (research_pipeline lives next to this file).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Simulate LLM Wiki Agent ingestion
# ---------------------------------------------------------------------------

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _extract_title(raw: str, fallback: str) -> str:
    """Pull the title from YAML frontmatter if present."""
    m = _FRONTMATTER_RE.match(raw)
    if m:
        for line in m.group(1).splitlines():
            if line.startswith("title:"):
                return line.partition(":")[2].strip().strip("\"'")
    return fallback


def simulate_wiki_ingest(raw_dir: Path, wiki_dir: Path) -> list[Path]:
    """Copy raw papers into a wiki sources/ directory with frontmatter.

    Also creates stub entity pages so wikilink aggregation can demonstrate
    cross-referencing.
    """
    sources_dir = wiki_dir / "sources"
    entities_dir = wiki_dir / "entities"
    sources_dir.mkdir(parents=True, exist_ok=True)
    entities_dir.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    entity_slugs: set[str] = set()

    for paper in sorted(raw_dir.glob("*.md")):
        raw = paper.read_text(encoding="utf-8")
        title = _extract_title(raw, paper.stem)
        slug = paper.stem

        # Write source page (keep original content, ensure frontmatter)
        dest = sources_dir / f"{slug}.md"
        if not raw.startswith("---"):
            header = f"---\ntitle: {title}\ntype: source\n---\n\n"
            dest.write_text(header + raw, encoding="utf-8")
        else:
            dest.write_text(raw, encoding="utf-8")
        created.append(dest)
        logger.info("  [wiki] Ingested source: %s", dest.name)

        # Collect entity slugs from tags
        m = _FRONTMATTER_RE.match(raw)
        if m:
            for line in m.group(1).splitlines():
                if line.startswith("tags:"):
                    tags = [t.strip() for t in line.partition(":")[2].split(",")]
                    entity_slugs.update(tags)

    # Create stub entity pages so aggregation can resolve wikilinks
    for slug in sorted(entity_slugs):
        if not slug:
            continue
        entity_path = entities_dir / f"{slug}.md"
        if not entity_path.exists():
            entity_path.write_text(
                f"---\ntitle: {slug.replace('-', ' ').title()}\ntype: entity\n---\n\n"
                f"# {slug.replace('-', ' ').title()}\n\n"
                f"Stub entity page for the **{slug}** concept.\n",
                encoding="utf-8",
            )
            created.append(entity_path)
            logger.info("  [wiki] Created entity stub: %s", entity_path.name)

    return created


# ---------------------------------------------------------------------------
# Step 2 — Run pipeline
# ---------------------------------------------------------------------------


def run_pipeline(wiki_dir: Path, output_dir: Path, *, aggregate: bool) -> None:
    """Invoke research_pipeline in batch mode."""
    from research_pipeline import (
        MockAdapterRepository,
        MockWeightGenerator,
    )
    from llm_patch.core.config import WatcherConfig
    from llm_patch.orchestrator import KnowledgeFusionOrchestrator
    from llm_patch.sources.wiki_source import WikiKnowledgeSource

    config = WatcherConfig(directory=wiki_dir)
    source = WikiKnowledgeSource(config, aggregate=aggregate)
    generator = MockWeightGenerator()
    repository = MockAdapterRepository(output_dir)

    orchestrator = KnowledgeFusionOrchestrator(
        source=source,
        generator=generator,
        repository=repository,
    )

    manifests = orchestrator.compile_all()
    logger.info("Generated %d adapter(s):", len(manifests))
    for m in manifests:
        logger.info("  %-40s rank=%d  uri=%s", m.adapter_id, m.rank, m.storage_uri)


# ---------------------------------------------------------------------------
# Step 3 — Validate
# ---------------------------------------------------------------------------


def validate_adapters(output_dir: Path) -> None:
    """Quick sanity check — just verify manifests were created."""
    from research_pipeline import MockAdapterRepository

    _repo = MockAdapterRepository(output_dir)
    # Since MockAdapterRepository is in-memory, we can only confirm the
    # pipeline completed without error.  Real validation with safetensors
    # is in validate_adapter.py.
    logger.info("Validation: pipeline completed without errors.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="End-to-end research pipeline runner.")
    examples_root = Path(__file__).resolve().parent.parent
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=examples_root / "data" / "papers",
        help="Directory containing raw paper markdown files.",
    )
    p.add_argument(
        "--wiki-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "wiki",
        help="Wiki output directory (created/overwritten).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "adapters",
        help="Adapter output directory.",
    )
    p.add_argument("--aggregate", action="store_true", help="Enable wikilink aggregation.")
    p.add_argument("--clean", action="store_true", help="Remove wiki dir before starting.")
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    args = build_parser().parse_args(argv)
    raw_dir: Path = args.raw_dir.resolve()
    wiki_dir: Path = args.wiki_dir.resolve()
    output_dir: Path = args.output_dir.resolve()

    if not raw_dir.exists():
        logger.error("Raw papers directory not found: %s", raw_dir)
        sys.exit(1)

    if args.clean and wiki_dir.exists():
        shutil.rmtree(wiki_dir)
        logger.info("Cleaned wiki directory: %s", wiki_dir)

    # Phase 1: Simulate wiki ingestion
    logger.info("=" * 60)
    logger.info("Phase 1: Simulating LLM Wiki Agent ingestion")
    logger.info("=" * 60)
    pages = simulate_wiki_ingest(raw_dir, wiki_dir)
    logger.info("Created %d wiki pages.\n", len(pages))

    # Phase 2: Run pipeline
    logger.info("=" * 60)
    logger.info("Phase 2: Running adapter generation pipeline")
    logger.info("=" * 60)
    run_pipeline(wiki_dir, output_dir, aggregate=args.aggregate)
    print()

    # Phase 3: Validate
    logger.info("=" * 60)
    logger.info("Phase 3: Validation")
    logger.info("=" * 60)
    validate_adapters(output_dir)

    logger.info("\nDone. See examples/README.md for detailed walkthrough.")


if __name__ == "__main__":
    main()
