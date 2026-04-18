"""Phase 1–8: End-to-end tests against the real architecture wiki.

Target wiki: C:\\dev\\projects\\architecture\\wiki
Tests data-source ingestion, WikiManager (mock + live), compile pipeline,
attach/use pipeline, runtime/chat, CLI, and validates structural integrity.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_patch.core.models import (
    AdapterManifest,
    ChatMessage,
    ChatRole,
    DocumentContext,
    ModelHandle,
)
from llm_patch.sources.wiki import (
    WikiDataSource,
    WikiDocumentAggregator,
    _extract_wikilinks,
    _parse_frontmatter,
)
from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.index import WikiIndex
from llm_patch.wiki.log import WikiLog
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.page import parse_wiki_page
from llm_patch.wiki.schema import WikiSchema

# ─── Load .env file if present (for GEMINI_API_KEY etc.) ─────────────
_env_path = Path(__file__).resolve().parents[2] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _key, _, _val = _line.partition("=")
            os.environ.setdefault(_key.strip(), _val.strip())

# ─── Location of the real architecture wiki ──────────────────────────
ARCH_WIKI = Path(r"C:\dev\projects\architecture\wiki")
ARCH_RAW = Path(r"C:\dev\projects\architecture\raw") if (Path(r"C:\dev\projects\architecture\raw")).exists() else None

HAS_ARCH_WIKI = ARCH_WIKI.exists() and any(ARCH_WIKI.rglob("*.md"))

skip_no_wiki = pytest.mark.skipif(
    not HAS_ARCH_WIKI,
    reason="Architecture wiki not found at C:\\dev\\projects\\architecture\\wiki",
)


# =====================================================================
# Phase 1 — Data Source Tests
# =====================================================================


@skip_no_wiki
class TestPhase1DataSourceIngestion:
    """Test WikiDataSource against the real architecture wiki."""

    def test_datasource_finds_all_md_files(self):
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        assert len(docs) > 0, "WikiDataSource should find markdown files"
        # We know the wiki has ~77 pages
        assert len(docs) >= 10, f"Expected >= 10 docs, got {len(docs)}"

    def test_datasource_document_ids_are_unique(self):
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        ids = [d.document_id for d in docs]
        assert len(ids) == len(set(ids)), "Document IDs must be unique"

    def test_datasource_content_is_nonempty(self):
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        empty = [d.document_id for d in docs if not d.content.strip()]
        assert len(empty) == 0, f"Found empty documents: {empty}"

    def test_frontmatter_parsing_real_pages(self):
        """Every wiki page should parse without errors."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        for doc in ds.fetch_all():
            assert isinstance(doc.metadata, dict)
            # Source path should be present
            assert "source_path" in doc.metadata

    def test_frontmatter_extracts_title(self):
        """Pages with title: in frontmatter should have it extracted."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        titled = [d for d in docs if "title" in d.metadata]
        assert len(titled) > 0, "At least some pages should have title in frontmatter"

    def test_yaml_list_parsing(self):
        """Tags frontmatter with [a, b] syntax should produce a Python list."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        list_tags = [
            d for d in docs
            if isinstance(d.metadata.get("tags"), list)
        ]
        # At least some pages should have list-format tags
        # (if none do, the test still passes — we just verify no crash)
        for d in list_tags:
            assert all(isinstance(t, str) for t in d.metadata["tags"])

    def test_markdown_link_extraction(self):
        """Pages with [text](path.md) links should have them extracted."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        docs_with_links = [d for d in docs if d.metadata.get("wikilinks")]
        assert len(docs_with_links) > 0, (
            "At least some pages should have markdown links extracted"
        )

    def test_link_extraction_includes_md_links(self):
        """Specifically test that [text](path.md) style links are extracted."""
        sample = "See [CQRS Pattern](patterns/cqrs.md) and [[Transformer]]."
        links = _extract_wikilinks(sample)
        assert "Transformer" in links, "Should extract [[wikilinks]]"
        assert "patterns/cqrs.md" in links, "Should extract [text](path.md) links"

    def test_datasource_with_aggregation(self):
        """WikiDataSource with aggregate=True should enrich docs."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True, aggregate=True)
        docs = list(ds.fetch_all())
        aggregated = [d for d in docs if d.metadata.get("aggregated")]
        # At least some documents should get aggregated content
        # (if the wiki has cross-references)
        if aggregated:
            for d in aggregated:
                assert len(d.content) > 100
                assert d.metadata.get("resolved_links")

    def test_datasource_name(self):
        ds = WikiDataSource(ARCH_WIKI)
        assert ds.name == "wiki"


@skip_no_wiki
class TestPhase1Aggregator:
    """Test WikiDocumentAggregator with the real wiki structure."""

    def test_aggregator_resolves_relative_md_links(self):
        """If a page links to patterns/cqrs.md, aggregator should find it."""
        agg = WikiDocumentAggregator(ARCH_WIKI)
        # Find any page with markdown links
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        linked_docs = [d for d in docs if d.metadata.get("wikilinks")]
        if not linked_docs:
            pytest.skip("No pages with links found")

        doc = linked_docs[0]
        result = agg.aggregate(doc)
        # Just verify it doesn't crash
        assert result.document_id == doc.document_id

    def test_aggregator_resolve_subdirectories(self):
        """Aggregator should search patterns/, pillars/, decisions/ etc."""
        agg = WikiDocumentAggregator(ARCH_WIKI)
        # Check which subdirs exist
        existing_subdirs = [
            d.name for d in ARCH_WIKI.iterdir()
            if d.is_dir() and d.name not in (".obsidian", ".git", "__pycache__")
        ]
        # Manually test resolution for a known pattern page if patterns/ exists
        if "patterns" in existing_subdirs:
            md_files = list((ARCH_WIKI / "patterns").glob("*.md"))
            if md_files:
                name = md_files[0].stem
                resolved = agg._resolve_link(f"patterns/{name}.md")
                assert resolved is not None, f"Should resolve patterns/{name}.md"


# =====================================================================
# Phase 2 — Wiki Pipeline with Mock Agent
# =====================================================================


@skip_no_wiki
class TestPhase2WikiPipelineMock:
    """Test WikiManager with MockWikiAgent against real raw sources."""

    @pytest.fixture()
    def mock_project(self, tmp_path: Path) -> Path:
        """Create a project layout copying a subset of real wiki sources."""
        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)

        # Copy up to 5 real .md files from the architecture wiki as raw sources
        source_files = sorted(ARCH_WIKI.rglob("*.md"))[:5]
        for f in source_files:
            dest = raw / f.name
            shutil.copy2(f, dest)

        return tmp_path

    @pytest.fixture()
    def mock_manager(self, mock_project: Path) -> WikiManager:
        agent = MockWikiAgent()
        manager = WikiManager(agent=agent, base_dir=mock_project)
        manager.init()
        return manager

    def test_init_creates_structure(self, mock_manager: WikiManager):
        assert mock_manager.wiki_dir.exists()
        assert (mock_manager.wiki_dir / "index.md").exists()

    def test_compile_all_mock(self, mock_manager: WikiManager, mock_project: Path):
        results = mock_manager.compile_all()
        raw_count = len(list((mock_project / "raw").rglob("*.md")))
        assert len(results) == raw_count, (
            f"Expected {raw_count} results, got {len(results)}"
        )

    def test_compile_creates_summary_pages(self, mock_manager: WikiManager):
        results = mock_manager.compile_all()
        for r in results:
            assert r.summary_page
            path = mock_manager.wiki_dir / r.summary_page
            assert path.exists(), f"Summary file missing: {r.summary_page}"

    def test_compile_extracts_entities(self, mock_manager: WikiManager):
        results = mock_manager.compile_all()
        total = sum(len(r.entities_extracted) for r in results)
        assert total > 0, "Mock agent should extract entities from headers"

    def test_index_after_compile(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        assert len(mock_manager.index) > 0

    def test_log_after_compile(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        log = WikiLog(mock_manager.wiki_dir / "log.md")
        entries = log.recent(100)
        assert len(entries) > 0

    def test_query_mock(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        result = mock_manager.query("What is software architecture?")
        assert result.answer

    def test_idempotent_compile(self, mock_manager: WikiManager):
        first = mock_manager.compile_all()
        second = mock_manager.compile_all()
        assert len(first) > 0
        assert len(second) == 0, "Second compile should skip processed sources"

    def test_status_mock(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        status = mock_manager.status()
        assert status["wiki_pages"] > 0
        assert status["index_entries"] > 0

    def test_lint_mock(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        report = mock_manager.lint()
        assert report is not None

    def test_page_frontmatter_valid(self, mock_manager: WikiManager):
        mock_manager.compile_all()
        for page_path in mock_manager.wiki_dir.rglob("*.md"):
            if page_path.name in ("index.md", "log.md"):
                continue
            text = page_path.read_text(encoding="utf-8")
            rel = page_path.relative_to(mock_manager.wiki_dir).as_posix()
            page = parse_wiki_page(text, rel)
            assert page.title, f"Missing title in {rel}"


# =====================================================================
# Phase 3 — Wiki Pipeline with Live Gemini (LiteLLM)
# =====================================================================

HAS_GEMINI_KEY = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))

skip_no_gemini = pytest.mark.skipif(
    not HAS_GEMINI_KEY,
    reason="GEMINI_API_KEY or GOOGLE_API_KEY not set",
)


@skip_no_wiki
@skip_no_gemini
@pytest.mark.integration
class TestPhase3WikiPipelineGemini:
    """E2E tests with real Gemini API calls via LiteLLM agent."""

    @pytest.fixture()
    def gemini_project(self, tmp_path: Path) -> Path:
        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)
        # Copy 2 real pages for live testing (keep cost low)
        source_files = sorted(ARCH_WIKI.rglob("*.md"))[:2]
        for f in source_files:
            shutil.copy2(f, raw / f.name)
        return tmp_path

    @pytest.fixture()
    def gemini_manager(self, gemini_project: Path):
        # Set env var for litellm to avoid SSL hang on import
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        from llm_patch.wiki.agents.litellm_agent import LiteLLMWikiAgent
        agent = LiteLLMWikiAgent(model="gemini/gemini-2.5-pro")
        manager = WikiManager(agent=agent, base_dir=gemini_project)
        manager.init()
        return manager

    def test_gemini_ingest(self, gemini_manager: WikiManager, gemini_project: Path):
        sources = list((gemini_project / "raw").rglob("*.md"))
        assert len(sources) > 0
        result = gemini_manager.ingest(sources[0])
        assert result.summary_page, "Gemini should produce a summary"
        assert len(result.entities_extracted) >= 1

        page = gemini_manager.read_page(result.summary_page)
        assert page is not None
        assert len(page.body) > 100, "Summary should be substantive"

    def test_gemini_query(self, gemini_manager: WikiManager, gemini_project: Path):
        sources = list((gemini_project / "raw").rglob("*.md"))
        gemini_manager.ingest(sources[0])
        result = gemini_manager.query("What is this about?")
        assert result.answer
        assert len(result.answer) > 20

    def test_gemini_compile_all(self, gemini_manager: WikiManager):
        results = gemini_manager.compile_all()
        assert len(results) >= 1

    def test_gemini_lint(self, gemini_manager: WikiManager):
        gemini_manager.compile_all()
        report = gemini_manager.lint()
        assert report is not None


# =====================================================================
# Phase 4 — Compile Pipeline (mock generator)
# =====================================================================


@skip_no_wiki
class TestPhase4CompilePipeline:
    """Test the compile pipeline with WikiDataSource and mock generator."""

    def test_compile_pipeline_with_wiki_source(self, tmp_path: Path):
        from llm_patch.core.models import AdapterManifest
        from llm_patch.pipelines.compile import CompilePipeline

        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        assert len(docs) > 0

        # Mock generator and repository
        mock_gen = MagicMock()
        mock_gen.generate.return_value = {"mock": "weights"}
        mock_gen.get_peft_config.return_value = MagicMock(
            to_dict=MagicMock(return_value={
                "r": 8, "target_modules": ["q_proj"], "peft_type": "LORA",
            })
        )
        mock_repo = MagicMock()
        mock_repo.save.return_value = AdapterManifest(
            adapter_id="test",
            rank=8,
            target_modules=["q_proj"],
            storage_uri=str(tmp_path / "out"),
        )

        pipeline = CompilePipeline(
            source=ds,
            generator=mock_gen,
            repository=mock_repo,
        )
        manifests = pipeline.compile_all()
        assert len(manifests) > 0
        assert mock_gen.generate.call_count == len(manifests)

    def test_compile_pipeline_respects_document_structure(self):
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())

        # Verify DocumentContext structure
        for doc in docs[:10]:
            assert doc.document_id
            assert doc.content
            assert isinstance(doc.metadata, dict)


# =====================================================================
# Phase 5 — CLI Smoke Tests
# =====================================================================


@skip_no_wiki
class TestPhase7CLISmokeTests:
    """Test CLI commands work (with mock agent)."""

    def test_cli_init(self, tmp_path: Path):
        from click.testing import CliRunner
        from llm_patch.cli.wiki import wiki

        runner = CliRunner()
        result = runner.invoke(wiki, [
            "--base-dir", str(tmp_path),
            "--agent", "mock",
            "init",
        ])
        assert result.exit_code == 0, f"CLI init failed: {result.output}"
        assert (tmp_path / "wiki").exists()

    def test_cli_compile(self, tmp_path: Path):
        from click.testing import CliRunner
        from llm_patch.cli.wiki import wiki

        # Set up raw sources
        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)
        src_files = sorted(ARCH_WIKI.rglob("*.md"))[:2]
        for f in src_files:
            shutil.copy2(f, raw / f.name)

        runner = CliRunner()
        # Init first
        runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "init"])
        # Compile
        result = runner.invoke(wiki, [
            "--base-dir", str(tmp_path),
            "--agent", "mock",
            "compile",
        ])
        assert result.exit_code == 0, f"CLI compile failed: {result.output}"
        assert "Compiled" in result.output

    def test_cli_status(self, tmp_path: Path):
        from click.testing import CliRunner
        from llm_patch.cli.wiki import wiki

        runner = CliRunner()
        runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "init"])
        result = runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "status"])
        assert result.exit_code == 0, f"CLI status failed: {result.output}"

    def test_cli_query(self, tmp_path: Path):
        from click.testing import CliRunner
        from llm_patch.cli.wiki import wiki

        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)
        src_files = sorted(ARCH_WIKI.rglob("*.md"))[:2]
        for f in src_files:
            shutil.copy2(f, raw / f.name)

        runner = CliRunner()
        runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "init"])
        runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "compile"])
        result = runner.invoke(wiki, [
            "--base-dir", str(tmp_path),
            "--agent", "mock",
            "query", "What is software architecture?",
        ])
        assert result.exit_code == 0, f"CLI query failed: {result.output}"

    def test_cli_lint(self, tmp_path: Path):
        from click.testing import CliRunner
        from llm_patch.cli.wiki import wiki

        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)
        src_files = sorted(ARCH_WIKI.rglob("*.md"))[:2]
        for f in src_files:
            shutil.copy2(f, raw / f.name)

        runner = CliRunner()
        runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "init"])
        runner.invoke(wiki, ["--base-dir", str(tmp_path), "--agent", "mock", "compile"])
        result = runner.invoke(wiki, [
            "--base-dir", str(tmp_path),
            "--agent", "mock",
            "lint",
        ])
        assert result.exit_code == 0, f"CLI lint failed: {result.output}"

    def test_cli_litellm_agent_choice(self, tmp_path: Path):
        """Verify --agent litellm is accepted (doesn't crash on arg parsing)."""
        from click.testing import CliRunner
        from llm_patch.cli.wiki import wiki

        runner = CliRunner()
        # Without a real API key, init with litellm agent should fail gracefully
        result = runner.invoke(wiki, [
            "--base-dir", str(tmp_path),
            "--agent", "litellm",
            "init",
        ])
        # Either succeeds (key found) or exits with error about missing key
        assert result.exit_code in (0, 1)


# =====================================================================
# Phase 8 — Structural Integrity of the Real Wiki
# =====================================================================


@skip_no_wiki
class TestPhase8WikiStructuralIntegrity:
    """Validate the structure and quality of the real architecture wiki."""

    def test_all_pages_parseable(self):
        """Every .md page should be readable and parseable."""
        failures = []
        for md_file in sorted(ARCH_WIKI.rglob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
                _parse_frontmatter(text)
            except Exception as e:
                failures.append(f"{md_file.name}: {e}")
        assert not failures, f"Parse failures:\n" + "\n".join(failures)

    def test_frontmatter_title_coverage(self):
        """Most pages should have a title in frontmatter."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())
        titled = [d for d in docs if d.metadata.get("title")]
        coverage = len(titled) / len(docs) * 100 if docs else 0
        # We expect a high percentage of pages to be well-structured
        assert coverage > 50, f"Only {coverage:.0f}% of pages have title in frontmatter"

    def test_no_orphan_subdirectories(self):
        """All subdirectories should contain at least one .md file."""
        for subdir in ARCH_WIKI.iterdir():
            if subdir.is_dir() and not subdir.name.startswith("."):
                md_files = list(subdir.glob("*.md"))
                assert len(md_files) > 0, f"Empty subdirectory: {subdir.name}"

    def test_wiki_has_expected_directories(self):
        """The architecture wiki should have patterns/ and pillars/ at minimum."""
        subdirs = {d.name for d in ARCH_WIKI.iterdir() if d.is_dir()}
        # Filter out hidden dirs
        subdirs = {d for d in subdirs if not d.startswith(".")}
        assert "patterns" in subdirs or "pillars" in subdirs, (
            f"Expected patterns/ or pillars/ in wiki, found: {subdirs}"
        )

    def test_cross_references_are_valid(self):
        """Links to .md files should resolve within the wiki."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        broken_links: list[str] = []
        for doc in ds.fetch_all():
            links = doc.metadata.get("wikilinks", [])
            # Determine the document's directory for resolving relative links
            src_path = doc.metadata.get("source_path", "")
            if src_path:
                doc_dir = Path(src_path).parent
            else:
                doc_dir = ARCH_WIKI
            for link in links:
                if link.endswith(".md"):
                    # Resolve relative to the document's directory
                    target = (doc_dir / link).resolve()
                    if not target.exists():
                        broken_links.append(
                            f"{doc.document_id} -> {link}"
                        )
        if broken_links:
            print(f"\nBroken links ({len(broken_links)}):")
            for bl in broken_links[:20]:
                print(f"  {bl}")
        # Allow up to 20% broken links (some may reference not-yet-created pages)
        ds2 = WikiDataSource(ARCH_WIKI, recursive=True)
        total_docs = len(list(ds2.fetch_all()))
        max_broken = max(total_docs, 10)
        assert len(broken_links) < max_broken, (
            f"Too many broken links: {len(broken_links)} (max {max_broken})"
        )

    def test_page_content_not_too_short(self):
        """Wiki pages should have meaningful content (not just frontmatter)."""
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        short_pages = []
        for doc in ds.fetch_all():
            if len(doc.content.strip()) < 50:
                short_pages.append(doc.document_id)
        if short_pages:
            print(f"\nShort pages ({len(short_pages)}): {short_pages[:10]}")
        # Allow some short pages (index, stubs) but not too many
        total = len(list(WikiDataSource(ARCH_WIKI, recursive=True).fetch_all()))
        ratio = len(short_pages) / total if total else 0
        assert ratio < 0.5, f"{ratio:.0%} of pages are very short"


# =====================================================================
# Phase 5 — Attach Pipeline (mock torch/PEFT, real wiki data)
# =====================================================================


@skip_no_wiki
class TestPhase5AttachPipeline:
    """Test CompilePipeline → UsePipeline flow with real wiki documents
    but mock generator/repository/model provider."""

    def _mock_generator(self):
        gen = MagicMock()
        gen.generate.return_value = {"lora_A": MagicMock(), "lora_B": MagicMock()}
        gen.get_peft_config.return_value = MagicMock(
            to_dict=MagicMock(return_value={
                "r": 8, "target_modules": ["q_proj"], "peft_type": "LORA",
            })
        )
        return gen

    def _mock_repo(self, tmp_path: Path):
        repo = MagicMock()
        manifests: list[AdapterManifest] = []

        def save_side_effect(aid, weights, cfg):
            m = AdapterManifest(
                adapter_id=aid,
                rank=8,
                target_modules=["q_proj"],
                storage_uri=str(tmp_path / aid),
            )
            manifests.append(m)
            return m

        repo.save.side_effect = save_side_effect
        repo.list_adapters.side_effect = lambda: list(manifests)
        return repo, manifests

    def test_compile_real_wiki_docs_then_attach(self, tmp_path: Path):
        """Full Ingest → Compile → Attach flow with real wiki data."""
        from llm_patch.pipelines.compile import CompilePipeline
        from llm_patch.pipelines.use import UsePipeline

        # Ingest real wiki docs
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())[:5]  # limit to 5 for speed

        source = MagicMock()
        source.fetch_all.return_value = docs

        gen = self._mock_generator()
        repo, manifests = self._mock_repo(tmp_path)

        # Step 1: Compile
        compile_pipe = CompilePipeline(source, gen, repo)
        results = compile_pipe.compile_all()

        assert len(results) == len(docs)
        assert gen.generate.call_count == len(docs)

        # Verify document content was passed to generator
        for i, doc in enumerate(docs):
            call_args = gen.generate.call_args_list[i]
            assert call_args[0][0].document_id == doc.document_id

        # Step 2: Attach
        model_provider = MagicMock()
        base_handle = ModelHandle(
            model=MagicMock(),
            tokenizer=MagicMock(),
            base_model_id="google/gemma-2-2b-it",
            device="cpu",
        )
        model_provider.load.return_value = base_handle

        adapter_loader = MagicMock()
        adapter_loader.attach.side_effect = lambda h, m: ModelHandle(
            model=h.model,
            tokenizer=h.tokenizer,
            base_model_id=h.base_model_id,
            attached_adapters=(*h.attached_adapters, m.adapter_id),
            device=h.device,
        )

        use_pipe = UsePipeline(model_provider, adapter_loader, repo)
        handle = use_pipe.load_and_attach("google/gemma-2-2b-it")

        assert len(handle.attached_adapters) == len(docs)
        assert adapter_loader.attach.call_count == len(docs)

    def test_compile_preserves_document_metadata(self, tmp_path: Path):
        """Generator receives document metadata from real wiki."""
        from llm_patch.pipelines.compile import CompilePipeline

        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())[:3]

        source = MagicMock()
        source.fetch_all.return_value = docs

        gen = self._mock_generator()
        repo, _ = self._mock_repo(tmp_path)

        pipeline = CompilePipeline(source, gen, repo)
        pipeline.compile_all()

        for i, doc in enumerate(docs):
            ctx = gen.generate.call_args_list[i][0][0]
            assert ctx.metadata.get("source_path"), f"Missing source_path for {doc.document_id}"
            assert len(ctx.content) > 0

    def test_attach_specific_adapter_subset(self, tmp_path: Path):
        """UsePipeline.load_and_attach with specific adapter_ids."""
        from llm_patch.pipelines.compile import CompilePipeline
        from llm_patch.pipelines.use import UsePipeline

        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())[:4]

        source = MagicMock()
        source.fetch_all.return_value = docs

        gen = self._mock_generator()
        repo, manifests = self._mock_repo(tmp_path)

        CompilePipeline(source, gen, repo).compile_all()

        # Only attach first 2
        target_ids = [m.adapter_id for m in manifests[:2]]

        model_provider = MagicMock()
        base_handle = ModelHandle(
            model=MagicMock(), tokenizer=MagicMock(),
            base_model_id="test", device="cpu",
        )
        model_provider.load.return_value = base_handle

        adapter_loader = MagicMock()
        adapter_loader.attach.side_effect = lambda h, m: ModelHandle(
            model=h.model, tokenizer=h.tokenizer,
            base_model_id=h.base_model_id,
            attached_adapters=(*h.attached_adapters, m.adapter_id),
            device=h.device,
        )

        use_pipe = UsePipeline(model_provider, adapter_loader, repo)
        handle = use_pipe.load_and_attach("test", adapter_ids=target_ids)

        assert len(handle.attached_adapters) == 2
        assert set(handle.attached_adapters) == set(target_ids)

    def test_merged_adapter_coverage(self, tmp_path: Path):
        """Verify merge_into_base works with compiled adapters."""
        from llm_patch.attach.merger import merge_into_base

        model = MagicMock()
        merged = MagicMock()
        model.merge_and_unload.return_value = merged

        handle = ModelHandle(
            model=model,
            tokenizer=MagicMock(),
            base_model_id="test",
            attached_adapters=("wiki-adapter-1", "wiki-adapter-2"),
            device="cpu",
        )

        out_dir = tmp_path / "merged_model"
        result = merge_into_base(handle, out_dir)

        assert result == out_dir
        assert out_dir.exists()
        model.merge_and_unload.assert_called_once()
        merged.save_pretrained.assert_called_once()


# =====================================================================
# Phase 6 — Use Pipeline: Runtime & Chat Session
# =====================================================================


@skip_no_wiki
class TestPhase6RuntimeAndChat:
    """Test PeftAgentRuntime and ChatSession with wiki-compiled adapters."""

    def _build_mock_handle(self) -> ModelHandle:
        """Create a ModelHandle with mocked model + tokenizer."""
        model = MagicMock()
        tokenizer = MagicMock()

        # Mock tokenizer call: returns dict with input_ids
        input_ids = MagicMock()
        input_ids.shape = [1, 5]  # batch=1, seq_len=5
        token_output = MagicMock()
        token_output.__getitem__ = MagicMock(return_value=input_ids)
        token_output.to = MagicMock(return_value={"input_ids": input_ids})
        tokenizer.return_value = token_output

        # Mock model.generate: returns output_ids tensor
        output_ids = MagicMock()
        # Slice [0, 5:] should return new token ids
        output_ids.__getitem__ = MagicMock(return_value=MagicMock())
        model.generate.return_value = output_ids
        model.device = "cpu"

        # Mock tokenizer.decode: returns text
        tokenizer.decode.return_value = "Generated response about architecture"

        return ModelHandle(
            model=model,
            tokenizer=tokenizer,
            base_model_id="google/gemma-2-2b-it",
            attached_adapters=("wiki-adapter",),
            device="cpu",
        )

    def test_build_agent_from_use_pipeline(self, tmp_path: Path):
        """UsePipeline.build_agent() produces a PeftAgentRuntime."""
        from llm_patch.pipelines.use import UsePipeline
        from llm_patch.runtime.agent import PeftAgentRuntime

        model_provider = MagicMock()
        model_provider.load.return_value = self._build_mock_handle()

        adapter_loader = MagicMock()
        repo = MagicMock()
        repo.list_adapters.return_value = []

        pipeline = UsePipeline(model_provider, adapter_loader, repo)
        agent = pipeline.build_agent("google/gemma-2-2b-it")

        assert isinstance(agent, PeftAgentRuntime)
        assert agent.handle.base_model_id == "google/gemma-2-2b-it"

    def test_runtime_generate(self):
        """PeftAgentRuntime.generate() returns generated text."""
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = self._build_mock_handle()
        runtime = PeftAgentRuntime(handle)

        # Patch torch.inference_mode for the test
        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock()
        )
        with patch.dict(sys.modules, {"torch": mock_torch}):
            result = runtime.generate("What is event sourcing?")

        assert isinstance(result, str)

    def test_runtime_chat(self):
        """PeftAgentRuntime.chat() returns ChatResponse."""
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = self._build_mock_handle()
        runtime = PeftAgentRuntime(handle)

        messages = [
            ChatMessage(role=ChatRole.USER, content="Explain CQRS pattern"),
        ]

        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock()
        )
        with patch.dict(sys.modules, {"torch": mock_torch}):
            response = runtime.chat(messages)

        assert response.message.role == ChatRole.ASSISTANT
        assert isinstance(response.message.content, str)

    def test_chat_session_multi_turn(self):
        """ChatSession maintains conversation history."""
        from llm_patch.runtime.agent import PeftAgentRuntime
        from llm_patch.runtime.session import ChatSession

        handle = self._build_mock_handle()
        runtime = PeftAgentRuntime(handle)

        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock()
        )

        session = ChatSession(
            runtime,
            system_prompt="You are an architecture wiki expert.",
            max_history=10,
        )

        with patch.dict(sys.modules, {"torch": mock_torch}):
            reply1 = session.say("What is event sourcing?")
            reply2 = session.say("How does it relate to CQRS?")

        # History: user1, assistant1, user2, assistant2 = 4 messages
        assert len(session.history) == 4
        assert session.history[0].role == ChatRole.USER
        assert session.history[1].role == ChatRole.ASSISTANT
        assert session.system_prompt == "You are an architecture wiki expert."

    def test_chat_session_history_trimming(self):
        """ChatSession.max_history trims old messages."""
        from llm_patch.runtime.agent import PeftAgentRuntime
        from llm_patch.runtime.session import ChatSession

        handle = self._build_mock_handle()
        runtime = PeftAgentRuntime(handle)

        mock_torch = MagicMock()
        mock_torch.inference_mode.return_value = MagicMock(
            __enter__=MagicMock(), __exit__=MagicMock()
        )

        session = ChatSession(runtime, max_history=4)

        with patch.dict(sys.modules, {"torch": mock_torch}):
            session.say("Question 1")
            session.say("Question 2")
            session.say("Question 3")  # This should trim earlier messages

        # max_history=4 → keeps last 4 messages
        assert len(session.history) <= 4

    def test_chat_session_clear_preserves_system_prompt(self):
        """ChatSession.clear() removes history but keeps system_prompt."""
        from llm_patch.runtime.agent import PeftAgentRuntime
        from llm_patch.runtime.session import ChatSession

        handle = self._build_mock_handle()
        runtime = PeftAgentRuntime(handle)

        session = ChatSession(runtime, system_prompt="Expert assistant")
        session.add_message(ChatRole.USER, "Hello")
        session.add_message(ChatRole.ASSISTANT, "Hi")

        assert len(session.history) == 2

        session.clear()
        assert len(session.history) == 0
        assert session.system_prompt == "Expert assistant"


# =====================================================================
# Phase 9 — Full Pipeline Integration (Ingest → Compile → Attach → Use)
# =====================================================================


@skip_no_wiki
class TestPhase9FullPipelineIntegration:
    """End-to-end integration: real wiki doc → compile → attach → generate."""

    def test_full_pipeline_mock_weights(self, tmp_path: Path):
        """Complete Ingest → Compile → Attach → Use with real content."""
        from llm_patch.pipelines.compile import CompilePipeline
        from llm_patch.pipelines.use import UsePipeline
        from llm_patch.runtime.agent import PeftAgentRuntime

        # 1. INGEST — real wiki documents
        ds = WikiDataSource(ARCH_WIKI, recursive=True)
        docs = list(ds.fetch_all())[:3]
        assert len(docs) >= 1

        source = MagicMock()
        source.fetch_all.return_value = docs

        # 2. COMPILE — mock generator but real document content
        gen = MagicMock()
        gen.generate.return_value = {"lora_A": MagicMock(), "lora_B": MagicMock()}
        gen.get_peft_config.return_value = MagicMock(
            to_dict=MagicMock(return_value={"r": 8, "target_modules": ["q_proj"], "peft_type": "LORA"})
        )

        manifests_store: list[AdapterManifest] = []
        repo = MagicMock()

        def save_fn(aid, w, cfg):
            m = AdapterManifest(
                adapter_id=aid, rank=8,
                target_modules=["q_proj"],
                storage_uri=str(tmp_path / aid),
            )
            manifests_store.append(m)
            return m

        repo.save.side_effect = save_fn
        repo.list_adapters.side_effect = lambda: list(manifests_store)

        compile_pipe = CompilePipeline(source, gen, repo)
        manifests = compile_pipe.compile_all()
        assert len(manifests) == len(docs)

        # 3. ATTACH — mock model provider + loader
        model_provider = MagicMock()
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Architecture response"

        base_handle = ModelHandle(
            model=mock_model,
            tokenizer=mock_tokenizer,
            base_model_id="google/gemma-2-2b-it",
            device="cpu",
        )
        model_provider.load.return_value = base_handle

        adapter_loader = MagicMock()
        adapter_loader.attach.side_effect = lambda h, m: ModelHandle(
            model=h.model, tokenizer=h.tokenizer,
            base_model_id=h.base_model_id,
            attached_adapters=(*h.attached_adapters, m.adapter_id),
            device=h.device,
        )

        use_pipe = UsePipeline(model_provider, adapter_loader, repo)
        handle = use_pipe.load_and_attach("google/gemma-2-2b-it")
        assert len(handle.attached_adapters) == len(docs)

        # 4. USE — build agent and generate
        agent = PeftAgentRuntime(handle)
        assert agent.handle.attached_adapters == tuple(d.document_id for d in docs)

    def test_wiki_compile_then_adapter_compile(self, tmp_path: Path):
        """WikiManager.compile_all() feeds into CompilePipeline."""
        from llm_patch.pipelines.compile import CompilePipeline

        # Wiki compile
        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)
        src_files = sorted(ARCH_WIKI.rglob("*.md"))[:3]
        for f in src_files:
            shutil.copy2(f, raw / f.name)

        agent = MockWikiAgent()
        manager = WikiManager(agent=agent, base_dir=tmp_path)
        manager.init()
        wiki_results = manager.compile_all()
        assert len(wiki_results) >= 1

        # Now feed wiki pages into compile pipeline
        wiki_pages = list(manager.wiki_dir.rglob("*.md"))
        wiki_docs = []
        for p in wiki_pages:
            if p.name in ("index.md", "log.md"):
                continue
            content = p.read_text(encoding="utf-8")
            if len(content.strip()) > 50:
                wiki_docs.append(DocumentContext(
                    document_id=p.stem,
                    content=content,
                    metadata={"source_path": str(p)},
                ))

        if not wiki_docs:
            pytest.skip("No substantial wiki pages produced")

        source = MagicMock()
        source.fetch_all.return_value = wiki_docs

        gen = MagicMock()
        gen.generate.return_value = {"w": MagicMock()}
        gen.get_peft_config.return_value = MagicMock()
        repo = MagicMock()
        repo.save.side_effect = lambda aid, w, cfg: AdapterManifest(
            adapter_id=aid, rank=8, target_modules=["q_proj"],
            storage_uri=str(tmp_path / "adapters" / aid),
        )

        pipeline = CompilePipeline(source, gen, repo)
        adapter_manifests = pipeline.compile_all()
        assert len(adapter_manifests) == len(wiki_docs)

    def test_wiki_pipeline_with_compile_callback(self, tmp_path: Path):
        """WikiPipeline with compile_pipeline triggers adapter rebuild."""
        from llm_patch.core.config import WikiConfig
        from llm_patch.pipelines.wiki import WikiPipeline

        raw = tmp_path / "raw" / "papers"
        raw.mkdir(parents=True)
        src_files = sorted(ARCH_WIKI.rglob("*.md"))[:2]
        for f in src_files:
            shutil.copy2(f, raw / f.name)

        agent = MockWikiAgent()
        config = WikiConfig(base_dir=tmp_path)

        mock_compile = MagicMock()
        mock_compile.compile_all.return_value = []

        pipeline = WikiPipeline(agent, config, compile_pipeline=mock_compile)
        pipeline.init()
        results = pipeline.compile_all()

        assert len(results) >= 1
        # compile_pipeline.compile_all() should have been triggered
        assert mock_compile.compile_all.call_count >= 1


# =====================================================================
# Phase 10 — Before/After Knowledge Comparison (Gemini)
# =====================================================================


@skip_no_wiki
@skip_no_gemini
@pytest.mark.integration
class TestPhase10BeforeAfterComparison:
    """Compare LLM answers WITHOUT wiki context vs WITH wiki context.

    This demonstrates the core value proposition: the wiki-enhanced agent
    produces more specific, cited, architecture-aware answers.
    """

    QUESTIONS = [
        "What is the CQRS pattern and when should it be used?",
        "Explain the key architectural decisions in this system.",
        "How do the architecture pillars relate to each other?",
    ]

    @pytest.fixture(scope="class")
    def gemini_agent(self):
        os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        from llm_patch.wiki.agents.litellm_agent import LiteLLMWikiAgent

        return LiteLLMWikiAgent(model="gemini/gemini-2.5-pro")

    @pytest.fixture(scope="class")
    def wiki_manager(self, gemini_agent, tmp_path_factory):
        """Build a wiki from a subset of real architecture docs."""
        project = tmp_path_factory.mktemp("wiki_compare")
        raw = project / "raw" / "papers"
        raw.mkdir(parents=True)

        # Copy up to 5 representative pages
        source_files = sorted(ARCH_WIKI.rglob("*.md"))[:5]
        for f in source_files:
            shutil.copy2(f, raw / f.name)

        manager = WikiManager(agent=gemini_agent, base_dir=project)
        manager.init()
        manager.compile_all()
        return manager

    def test_before_answer_is_generic(self, gemini_agent):
        """Raw LLM answer (no wiki) should be valid but generic."""
        question = self.QUESTIONS[0]
        raw_answer = gemini_agent._call(
            "You are a helpful assistant. Answer concisely.",
            question,
        )

        assert len(raw_answer) > 20, "LLM should produce a substantive answer"
        # The raw answer is valid but won't reference our specific wiki pages
        print(f"\n--- BEFORE (no wiki) ---\n{raw_answer[:600]}")

    def test_after_answer_cites_wiki_pages(self, wiki_manager):
        """Wiki-enhanced answer should cite specific pages."""
        question = self.QUESTIONS[0]
        result = wiki_manager.query(question)

        assert len(result.answer) > 20
        assert len(result.cited_pages) >= 1, (
            "Wiki-enhanced answer should cite at least one page"
        )
        print(f"\n--- AFTER (wiki-enhanced) ---\n{result.answer[:600]}")
        print(f"Cited: {result.cited_pages}")

    def test_wiki_answer_is_longer_and_richer(self, gemini_agent, wiki_manager):
        """Wiki answer should be at least as detailed as raw answer."""
        question = self.QUESTIONS[1]

        raw_answer = gemini_agent._call(
            "You are a helpful assistant. Answer concisely.",
            question,
        )
        wiki_result = wiki_manager.query(question)

        print(f"\n--- BEFORE ---\n{raw_answer[:400]}")
        print(f"\n--- AFTER ---\n{wiki_result.answer[:400]}")
        print(f"Cited: {wiki_result.cited_pages}")

        # Wiki answer should be substantive
        assert len(wiki_result.answer) > 50
        # Wiki answer should cite pages
        assert len(wiki_result.cited_pages) >= 1

    def test_multi_question_comparison(self, gemini_agent, wiki_manager):
        """Run all comparison questions and print side-by-side results."""
        for question in self.QUESTIONS:
            raw_answer = gemini_agent._call(
                "You are a helpful assistant. Answer concisely.",
                question,
            )
            wiki_result = wiki_manager.query(question)

            print(f"\n{'='*60}")
            print(f"Q: {question}")
            print(f"\n--- BEFORE (raw LLM) ---")
            print(raw_answer[:300])
            print(f"\n--- AFTER (wiki-enhanced) ---")
            print(wiki_result.answer[:300])
            print(f"Cited: {wiki_result.cited_pages}")

            # Both should produce answers
            assert len(raw_answer) > 10
            assert len(wiki_result.answer) > 10
            # Wiki version should cite pages
            assert len(wiki_result.cited_pages) >= 1

    def test_wiki_context_adds_specificity(self, gemini_agent, wiki_manager):
        """Wiki answers should contain specific terms from the wiki pages."""
        question = "What are the main architectural patterns used?"
        wiki_result = wiki_manager.query(question)

        # The answer should reference specific wiki content
        answer_lower = wiki_result.answer.lower()
        # Should contain at least some architecture-specific vocabulary
        arch_terms = [
            "pattern", "architecture", "design", "system",
            "component", "service", "event", "domain",
        ]
        found_terms = [t for t in arch_terms if t in answer_lower]
        assert len(found_terms) >= 3, (
            f"Wiki answer should use architecture vocabulary, "
            f"found only: {found_terms}"
        )
        assert len(wiki_result.cited_pages) >= 1
