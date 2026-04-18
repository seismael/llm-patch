"""Tests for llm_patch.wiki.obsidian — Obsidian vault integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_patch.wiki.obsidian import (
    GraphData,
    GraphEdge,
    GraphNode,
    ObsidianConfig,
    ObsidianVault,
)
from llm_patch.wiki.page import WikiPage, WikiPageFrontmatter, PageType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def vault_root(tmp_path: Path) -> Path:
    """Create a temporary directory to use as a vault root."""
    root = tmp_path / "myvault"
    root.mkdir()
    return root


@pytest.fixture()
def vault(vault_root: Path) -> ObsidianVault:
    """Return an ObsidianVault with default config."""
    return ObsidianVault(vault_root)


@pytest.fixture()
def wiki_with_pages(vault_root: Path) -> Path:
    """Create a wiki/ directory with several interlinked pages."""
    wiki = vault_root / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "entities").mkdir(parents=True)
    (wiki / "summaries").mkdir(parents=True)

    (wiki / "concepts" / "attention.md").write_text(
        '---\ntitle: "Attention"\ntype: concept\ntags: [transformers]\n'
        "created: 2026-04-01\nupdated: 2026-04-01\nconfidence: high\n---\n\n"
        "Attention is a mechanism.\n\nSee [[Transformer]] and [[Self-Attention]].\n",
        encoding="utf-8",
    )
    (wiki / "entities" / "transformer.md").write_text(
        '---\ntitle: "Transformer"\ntype: entity\ntags: [architecture]\n'
        "created: 2026-04-01\nupdated: 2026-04-01\nconfidence: high\n---\n\n"
        "The Transformer model.\n\nRelated: [[Attention]].\n",
        encoding="utf-8",
    )
    (wiki / "entities" / "self-attention.md").write_text(
        '---\ntitle: "Self-Attention"\ntype: entity\ntags: [mechanism]\n'
        "created: 2026-04-01\nupdated: 2026-04-01\nconfidence: medium\n---\n\n"
        "Self-Attention is a core [[Attention]] variant.\n",
        encoding="utf-8",
    )
    (wiki / "summaries" / "paper-one.md").write_text(
        '---\ntitle: "Paper One"\ntype: summary\ntags: []\n'
        "created: 2026-04-01\nupdated: 2026-04-01\nconfidence: high\n---\n\n"
        "Summary with no outbound links.\n",
        encoding="utf-8",
    )
    # index.md and log.md should be excluded from graph
    (wiki / "index.md").write_text("# Index\n", encoding="utf-8")
    (wiki / "log.md").write_text("# Log\n", encoding="utf-8")

    return wiki


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetection:
    def test_detect_false_no_obsidian_dir(self, vault_root: Path) -> None:
        assert ObsidianVault.detect(vault_root) is False

    def test_detect_true_with_obsidian_dir(self, vault_root: Path) -> None:
        (vault_root / ".obsidian").mkdir()
        assert ObsidianVault.detect(vault_root) is True

    def test_is_vault_property(self, vault: ObsidianVault, vault_root: Path) -> None:
        assert vault.is_vault is False
        (vault_root / ".obsidian").mkdir()
        assert vault.is_vault is True


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_initialize_creates_obsidian_dir(self, vault: ObsidianVault) -> None:
        created = vault.initialize()
        assert vault.obsidian_dir.is_dir()
        assert ".obsidian/app.json" in created
        assert ".obsidian/appearance.json" in created
        assert ".obsidian/community-plugins.json" in created

    def test_initialize_writes_valid_json(self, vault: ObsidianVault) -> None:
        vault.initialize()
        app = json.loads((vault.obsidian_dir / "app.json").read_text(encoding="utf-8"))
        assert "userIgnoreFilters" in app
        assert "attachmentFolderPath" in app

    def test_initialize_sets_ignore_filters(self, vault_root: Path) -> None:
        cfg = ObsidianConfig(
            enabled=True,
            ignore_filters=[".claude", "raw", ".git", "reports"],
        )
        vault = ObsidianVault(vault_root, cfg)
        vault.initialize()
        app = json.loads((vault.obsidian_dir / "app.json").read_text(encoding="utf-8"))
        assert "reports" in app["userIgnoreFilters"]
        assert ".claude" in app["userIgnoreFilters"]

    def test_initialize_sets_attachment_folder(self, vault_root: Path) -> None:
        cfg = ObsidianConfig(enabled=True, attachment_folder="wiki/assets")
        vault = ObsidianVault(vault_root, cfg)
        vault.initialize()
        app = json.loads((vault.obsidian_dir / "app.json").read_text(encoding="utf-8"))
        assert app["attachmentFolderPath"] == "wiki/assets"

    def test_initialize_creates_assets_directory(self, vault: ObsidianVault) -> None:
        vault.initialize()
        assert (vault.root / "raw" / "assets").is_dir()

    def test_initialize_preserves_existing_config(self, vault: ObsidianVault) -> None:
        vault.initialize()
        # Write custom config
        app_json = vault.obsidian_dir / "app.json"
        custom = {"custom_key": True}
        app_json.write_text(json.dumps(custom), encoding="utf-8")
        # Re-initialize should not overwrite
        created = vault.initialize()
        assert ".obsidian/app.json" not in created
        reread = json.loads(app_json.read_text(encoding="utf-8"))
        assert reread.get("custom_key") is True

    def test_initialize_idempotent(self, vault: ObsidianVault) -> None:
        created1 = vault.initialize()
        created2 = vault.initialize()
        assert len(created1) > 0
        assert len(created2) == 0  # Nothing new created


# ---------------------------------------------------------------------------
# Configuration management
# ---------------------------------------------------------------------------


class TestConfigManagement:
    def test_read_app_config_missing(self, vault: ObsidianVault) -> None:
        assert vault.read_app_config() == {}

    def test_read_app_config_after_init(self, vault: ObsidianVault) -> None:
        vault.initialize()
        config = vault.read_app_config()
        assert config.get("showFrontmatter") is True

    def test_update_ignore_filters(self, vault: ObsidianVault) -> None:
        vault.initialize()
        vault.update_ignore_filters(["docs", "scripts"])
        config = vault.read_app_config()
        filters = config["userIgnoreFilters"]
        assert "docs" in filters
        assert "scripts" in filters
        # Originals preserved
        assert ".git" in filters

    def test_update_ignore_filters_deduplicates(self, vault: ObsidianVault) -> None:
        vault.initialize()
        vault.update_ignore_filters(["raw", "raw", ".git"])
        config = vault.read_app_config()
        filters = config["userIgnoreFilters"]
        assert filters.count("raw") == 1


# ---------------------------------------------------------------------------
# Embed extraction
# ---------------------------------------------------------------------------


class TestEmbeds:
    def test_extract_embeds(self) -> None:
        text = "Here is ![[image.png]] and ![[doc.pdf|my document]]."
        result = ObsidianVault.extract_embeds(text)
        assert "image.png" in result
        assert "doc.pdf" in result

    def test_extract_embeds_empty(self) -> None:
        assert ObsidianVault.extract_embeds("No embeds here.") == []

    def test_extract_embeds_vs_wikilinks(self) -> None:
        text = "Link [[page]] and embed ![[image.png]]."
        embeds = ObsidianVault.extract_embeds(text)
        assert embeds == ["image.png"]


# ---------------------------------------------------------------------------
# Graph building
# ---------------------------------------------------------------------------


class TestGraphBuilding:
    def test_build_graph_empty_dir(self, vault: ObsidianVault, vault_root: Path) -> None:
        wiki = vault_root / "wiki"
        wiki.mkdir()
        graph = vault.build_graph(wiki)
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_build_graph_nonexistent_dir(self, vault: ObsidianVault, vault_root: Path) -> None:
        graph = vault.build_graph(vault_root / "nope")
        assert graph.node_count == 0

    def test_build_graph_with_pages(
        self, vault: ObsidianVault, wiki_with_pages: Path
    ) -> None:
        graph = vault.build_graph(wiki_with_pages)
        # 4 content pages (index.md and log.md excluded)
        assert graph.node_count == 4
        # attention → transformer, attention → self-attention, transformer → attention,
        # self-attention → attention
        assert graph.edge_count >= 3

    def test_build_graph_excludes_index_and_log(
        self, vault: ObsidianVault, wiki_with_pages: Path
    ) -> None:
        graph = vault.build_graph(wiki_with_pages)
        ids = [n.id for n in graph.nodes]
        assert "index" not in ids
        assert "log" not in ids

    def test_graph_to_dict(self) -> None:
        graph = GraphData(
            nodes=[GraphNode(id="a", title="A", page_type="concept")],
            edges=[GraphEdge(source="a", target="b")],
        )
        d = graph.to_dict()
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["id"] == "a"
        assert len(d["edges"]) == 1

    def test_export_graph_json_to_file(
        self, vault: ObsidianVault, wiki_with_pages: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "graph.json"
        result = vault.export_graph_json(wiki_with_pages, out)
        assert out.exists()
        parsed = json.loads(result)
        assert "nodes" in parsed
        assert "edges" in parsed

    def test_export_graph_json_no_file(
        self, vault: ObsidianVault, wiki_with_pages: Path
    ) -> None:
        result = vault.export_graph_json(wiki_with_pages)
        parsed = json.loads(result)
        assert len(parsed["nodes"]) == 4


# ---------------------------------------------------------------------------
# ObsidianConfig model
# ---------------------------------------------------------------------------


class TestObsidianConfig:
    def test_defaults(self) -> None:
        cfg = ObsidianConfig()
        assert cfg.enabled is False
        assert cfg.attachment_folder == "raw/assets"
        assert ".git" in cfg.ignore_filters
        assert cfg.enable_dataview is True

    def test_custom_values(self) -> None:
        cfg = ObsidianConfig(
            enabled=True,
            attachment_folder="assets",
            ignore_filters=["build", "dist"],
            daily_notes=True,
        )
        assert cfg.enabled is True
        assert cfg.attachment_folder == "assets"
        assert cfg.ignore_filters == ["build", "dist"]
        assert cfg.daily_notes is True


# ---------------------------------------------------------------------------
# WikiManager integration (Obsidian helpers)
# ---------------------------------------------------------------------------


class TestWikiManagerObsidian:
    """Tests for Obsidian-related methods on WikiManager."""

    @pytest.fixture()
    def manager_with_pages(self, vault_root: Path, wiki_with_pages: Path) -> "WikiManager":
        from llm_patch.wiki.agents.mock import MockWikiAgent
        from llm_patch.wiki.manager import WikiManager

        mgr = WikiManager(agent=MockWikiAgent(), base_dir=vault_root)
        mgr.init()
        return mgr

    def test_init_with_obsidian_flag(self, vault_root: Path) -> None:
        from llm_patch.wiki.agents.mock import MockWikiAgent
        from llm_patch.wiki.manager import WikiManager

        mgr = WikiManager(agent=MockWikiAgent(), base_dir=vault_root)
        mgr.init(obsidian=True)
        assert (vault_root / ".obsidian").is_dir()
        assert (vault_root / ".obsidian" / "app.json").exists()

    def test_enable_obsidian(self, vault_root: Path) -> None:
        from llm_patch.wiki.agents.mock import MockWikiAgent
        from llm_patch.wiki.manager import WikiManager

        mgr = WikiManager(agent=MockWikiAgent(), base_dir=vault_root)
        mgr.init()
        vault = mgr.enable_obsidian()
        assert vault.is_vault
        assert mgr.obsidian is vault

    def test_graph_returns_data(self, manager_with_pages: "WikiManager") -> None:
        graph = manager_with_pages.graph()
        assert isinstance(graph, GraphData)
        assert graph.node_count >= 0

    def test_export_graph(
        self, manager_with_pages: "WikiManager", tmp_path: Path
    ) -> None:
        out = tmp_path / "out.json"
        json_str = manager_with_pages.export_graph(out)
        assert out.exists()
        assert "nodes" in json_str

    def test_status_includes_obsidian(self, vault_root: Path) -> None:
        from llm_patch.wiki.agents.mock import MockWikiAgent
        from llm_patch.wiki.manager import WikiManager

        mgr = WikiManager(agent=MockWikiAgent(), base_dir=vault_root)
        mgr.init(obsidian=True)
        st = mgr.status()
        assert st["obsidian_vault"] is True

    def test_status_obsidian_false_by_default(self, vault_root: Path) -> None:
        from llm_patch.wiki.agents.mock import MockWikiAgent
        from llm_patch.wiki.manager import WikiManager

        mgr = WikiManager(agent=MockWikiAgent(), base_dir=vault_root)
        mgr.init()
        st = mgr.status()
        assert st["obsidian_vault"] is False

    def test_schema_obsidian_enabled_auto_inits(self, vault_root: Path) -> None:
        from llm_patch.wiki.agents.mock import MockWikiAgent
        from llm_patch.wiki.manager import WikiManager
        from llm_patch.wiki.schema import WikiSchema

        schema = WikiSchema.default()
        schema.obsidian_enabled = True
        mgr = WikiManager(agent=MockWikiAgent(), base_dir=vault_root, schema=schema)
        assert mgr.obsidian is not None
        mgr.init()
        assert (vault_root / ".obsidian").is_dir()
