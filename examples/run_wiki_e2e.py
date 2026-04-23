#!/usr/bin/env python
"""Full E2E pipeline: raw papers → Claude wiki agent → structured wiki → validated output.

This script connects to your Claude account via the Anthropic API and runs
the full wiki pipeline against the example papers (Attention, GPT-3, LoRA).

Usage:
    # Set your API key:
    set ANTHROPIC_API_KEY=sk-ant-api03-...

    # Run with Claude (default):
    python examples/run_wiki_e2e.py

    # Run with mock agent (no API key needed):
    python examples/run_wiki_e2e.py --mock

    # Run against a custom directory:
    python examples/run_wiki_e2e.py --base-dir ./my-project

    # Specify a different Claude model:
    python examples/run_wiki_e2e.py --model claude-sonnet-4-20250514
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.schema import WikiSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wiki-e2e")


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _banner(text: str) -> None:
    width = max(len(text) + 4, 60)
    logger.info("=" * width)
    logger.info(f"  {text}")
    logger.info("=" * width)


def _section(text: str) -> None:
    logger.info("")
    logger.info(f"── {text} " + "─" * max(0, 55 - len(text)))


# ──────────────────────────────────────────────────────────────────────
# Pipeline stages
# ──────────────────────────────────────────────────────────────────────


def create_agent(args: argparse.Namespace):
    """Create the wiki agent based on CLI flags."""
    if args.mock:
        logger.info("Using MockWikiAgent (no LLM calls)")
        return MockWikiAgent()

    # Real Claude agent
    try:
        from llm_patch.wiki.agents.anthropic_agent import AnthropicWikiAgent
    except ImportError:
        logger.error(
            "anthropic package not installed. Run: pip install anthropic\n"
            "Or use --mock to run without an LLM."
        )
        sys.exit(1)

    model = args.model or "claude-sonnet-4-20250514"
    try:
        agent = AnthropicWikiAgent(
            api_key=args.api_key or None,
            model=model,
        )
    except ValueError as exc:
        logger.error(str(exc))
        logger.error("To run without an API key, use --mock.")
        sys.exit(1)

    logger.info("Using AnthropicWikiAgent (model=%s)", model)
    return agent


def run_pipeline(args: argparse.Namespace) -> bool:
    """Run the full pipeline. Returns True on success."""
    start_time = time.monotonic()

    _banner("LLM-Patch Wiki E2E Pipeline")

    # ── Setup ──────────────────────────────────────────────────────
    base_dir = Path(args.base_dir).resolve()
    raw_dir = base_dir / "raw"

    # If using the default directory, copy example papers there
    example_papers = Path(__file__).parent / "raw" / "papers"
    if not raw_dir.exists() and example_papers.exists():
        _section("Copying example papers to raw/")
        shutil.copytree(example_papers, raw_dir / "papers")
        logger.info("Copied %d papers", len(list((raw_dir / "papers").glob("*.md"))))

    if not raw_dir.exists():
        logger.error("No raw/ directory found at %s", base_dir)
        return False

    raw_files = sorted(raw_dir.rglob("*.md"))
    if not raw_files:
        logger.error("No markdown files found in %s", raw_dir)
        return False

    logger.info("Base directory: %s", base_dir)
    logger.info("Raw sources: %d files", len(raw_files))

    # ── Create agent & manager ─────────────────────────────────────
    agent = create_agent(args)
    schema = WikiSchema.default()
    if args.schema and Path(args.schema).exists():
        schema = WikiSchema.from_file(Path(args.schema))

    manager = WikiManager(agent=agent, base_dir=base_dir, schema=schema)

    # ── Phase 1: Initialize ────────────────────────────────────────
    _section("Phase 1: Initialize Wiki")
    manager.init()
    logger.info("Wiki directories created at %s", manager.wiki_dir)

    # ── Phase 2: Ingest all raw sources ────────────────────────────
    _section("Phase 2: Ingest Raw Sources")
    try:
        ingest_results = manager.compile_all()
    except RuntimeError as exc:
        logger.error("Pipeline failed during ingest: %s", exc)
        return False
    total_created = sum(len(r.pages_created) for r in ingest_results)
    total_updated = sum(len(r.pages_updated) for r in ingest_results)
    total_entities = sum(len(r.entities_extracted) for r in ingest_results)
    logger.info("Ingested %d sources", len(ingest_results))
    logger.info("  Pages created: %d", total_created)
    logger.info("  Pages updated: %d", total_updated)
    logger.info("  Entities extracted: %d", total_entities)

    for result in ingest_results:
        logger.info("  [%s]", Path(result.source_path).name)
        for page in result.pages_created:
            logger.info("    + %s", page)
        for page in result.pages_updated:
            logger.info("    ~ %s", page)

    # ── Phase 3: Query ─────────────────────────────────────────────
    _section("Phase 3: Query Wiki")
    queries = [
        "What is the Transformer architecture?",
        "How does LoRA achieve parameter-efficient fine-tuning?",
        "What are the scaling properties of GPT-3?",
    ]
    for q in queries:
        logger.info("Q: %s", q)
        result = manager.query(q, save_as_synthesis=True)
        answer_preview = result.answer[:200].replace("\n", " ")
        logger.info("A: %s...", answer_preview)
        logger.info("  Cited: %s", ", ".join(result.cited_pages[:5]) or "none")
        if result.filed_as:
            logger.info("  Saved: %s", result.filed_as)
        logger.info("")

    # ── Phase 4: Lint ──────────────────────────────────────────────
    _section("Phase 4: Lint Wiki")
    report = manager.lint()
    logger.info("Issues found: %d", report.issue_count)
    for issue in report.issues[:10]:
        logger.info("  [%s] %s: %s", issue.category, issue.page, issue.description)
    if report.suggestions:
        logger.info("Suggestions:")
        for s in report.suggestions[:5]:
            logger.info("  - %s", s)

    # ── Phase 5: Status ────────────────────────────────────────────
    _section("Phase 5: Final Status")
    status = manager.status()
    for key, value in status.items():
        logger.info("  %s: %d", key.replace("_", " ").title(), value)

    # ── Validation ─────────────────────────────────────────────────
    _section("Validation Checks")
    ok = True

    # Check: All raw sources ingested
    if len(ingest_results) != len(raw_files):
        logger.error(
            "FAIL: Expected %d ingest results, got %d", len(raw_files), len(ingest_results)
        )
        ok = False
    else:
        logger.info("PASS: All %d raw sources ingested", len(raw_files))

    # Check: Summary pages exist
    for result in ingest_results:
        if result.summary_page:
            summary_path = manager.wiki_dir / result.summary_page
            if summary_path.exists():
                logger.info("PASS: Summary exists — %s", result.summary_page)
            else:
                logger.error("FAIL: Summary missing — %s", result.summary_page)
                ok = False

    # Check: Index has entries
    if len(manager.index) >= len(ingest_results):
        logger.info(
            "PASS: Index has %d entries (>= %d sources)", len(manager.index), len(ingest_results)
        )
    else:
        logger.error("FAIL: Index has only %d entries", len(manager.index))
        ok = False

    # Check: Wiki pages on disk
    wiki_pages = list(manager.wiki_dir.rglob("*.md"))
    # Exclude index.md and log.md from wiki page count
    content_pages = [p for p in wiki_pages if p.name not in ("index.md", "log.md")]
    if content_pages:
        logger.info("PASS: %d wiki content pages on disk", len(content_pages))
    else:
        logger.error("FAIL: No wiki content pages on disk")
        ok = False

    # Check: At least some entities extracted
    if total_entities > 0:
        logger.info("PASS: %d entities extracted across all sources", total_entities)
    else:
        logger.warning("WARN: No entities extracted (may be expected with mock agent)")

    elapsed = time.monotonic() - start_time

    _section("Result")
    if ok:
        logger.info("ALL VALIDATION CHECKS PASSED (%.1fs)", elapsed)
    else:
        logger.error("SOME CHECKS FAILED (%.1fs)", elapsed)

    return ok


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full LLM Wiki E2E pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-dir",
        default=".",
        help="Root project directory (default: current dir)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockWikiAgent instead of Claude (no API key needed)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Claude model name (default: claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--schema",
        default=None,
        help="Path to a CLAUDE.md-style wiki schema file",
    )
    args = parser.parse_args()

    success = run_pipeline(args)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
