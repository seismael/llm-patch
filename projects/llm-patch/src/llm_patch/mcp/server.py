"""MCP (Model Context Protocol) server exposing wiki tools.

Provides a set of tools that any MCP-compatible client (Claude Desktop,
Cursor, etc.) can use to interact with the wiki: ingest, search, read,
write, query, lint, and status.

Requires the ``mcp`` package (``pip install mcp``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "MCP server requires the 'mcp' package. Install with: pip install 'llm-patch[mcp]'"
    )

from llm_patch.wiki.agents.mock import MockWikiAgent
from llm_patch.wiki.interfaces import IWikiAgent
from llm_patch.wiki.manager import WikiManager
from llm_patch.wiki.page import parse_wiki_page
from llm_patch.wiki.schema import WikiSchema

# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

mcp = FastMCP("llm-patch")

# Module-level WikiManager — initialized by ``configure()``.
_wiki: WikiManager | None = None

# Module-level Adapter Market handles — set via ``configure_hub()``.
_registry: object | None = None  # IAdapterRegistryClient
_controller: object | None = None  # IRuntimeAdapterController


def configure_hub(
    registry: object | None = None,
    controller: object | None = None,
) -> None:
    """Wire optional Adapter Market dependencies into MCP tools.

    Both arguments are :class:`~llm_patch.core.interfaces.IAdapterRegistryClient`
    and :class:`~llm_patch.core.interfaces.IRuntimeAdapterController`
    instances respectively. Each defaults to ``None``; tools that
    require an unset dependency raise :class:`RegistryUnavailableError`.
    """
    global _registry, _controller
    if registry is not None:
        _registry = registry
    if controller is not None:
        _controller = controller


def _require_registry() -> object:
    from llm_patch_utils import RegistryUnavailableError

    if _registry is None:
        raise RegistryUnavailableError(
            "No registry client configured. "
            "Call llm_patch.mcp.server.configure_hub(registry=...) at startup."
        )
    return _registry


def _require_controller() -> object:
    from llm_patch_utils import RegistryUnavailableError

    if _controller is None:
        raise RegistryUnavailableError(
            "No runtime adapter controller configured. "
            "Call llm_patch.mcp.server.configure_hub(controller=...) at startup."
        )
    return _controller


def configure(
    base_dir: str | Path,
    agent: IWikiAgent | None = None,
    schema: WikiSchema | None = None,
) -> WikiManager:
    """Initialize the global WikiManager used by MCP tools.

    Args:
        base_dir: Root directory containing raw/ and wiki/.
        agent: LLM agent to use.  Defaults to MockWikiAgent.
        schema: Wiki schema.  Defaults to WikiSchema.default().

    Returns:
        The configured WikiManager.
    """
    global _wiki
    _wiki = WikiManager(
        agent=agent or MockWikiAgent(),
        base_dir=Path(base_dir),
        schema=schema,
    )
    _wiki.init()
    return _wiki


def _get_wiki() -> WikiManager:
    if _wiki is None:
        raise RuntimeError(
            "Wiki not configured. Call configure(base_dir=...) first, "
            "or set LLM_PATCH_WIKI_DIR environment variable."
        )
    return _wiki


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def wiki_guide() -> str:
    """Return the wiki schema — directory layout, page types, and rules."""
    wiki = _get_wiki()
    schema = wiki.schema
    lines = [
        "# Wiki Schema",
        "",
        "## Directory Layout",
        f"- Raw sources: {schema.layout.raw_dir}/",
        f"- Wiki: {schema.layout.wiki_dir}/",
        f"- Summaries: {schema.layout.summaries_dir}/",
        f"- Concepts: {schema.layout.concepts_dir}/",
        f"- Entities: {schema.layout.entities_dir}/",
        f"- Syntheses: {schema.layout.syntheses_dir}/",
        f"- Journal: {schema.layout.journal_dir}/",
        "",
        "## Page Types",
    ]
    for page_type, config in schema.page_types.items():
        sections = ", ".join(config.required_sections) if config.required_sections else "none"
        directory = config.directory or schema.get_directory_for_type(page_type)
        lines.append(f"- **{page_type}**: directory={directory} (sections: {sections})")
    lines.append("")
    lines.append("## Rules")
    for i, rule in enumerate(schema.rules, 1):
        lines.append(f"{i}. {rule}")
    return "\n".join(lines)


@mcp.tool()
def wiki_ingest(source_path: str) -> str:
    """Ingest a raw source file into the wiki.

    Args:
        source_path: Path to the raw markdown file to ingest.
    """
    wiki = _get_wiki()
    path = Path(source_path).resolve()
    if not path.is_file():
        return f"Error: File not found: {source_path}"

    result = wiki.ingest(path)
    return json.dumps(asdict(result), indent=2, default=str)


@mcp.tool()
def wiki_search(keyword: str) -> str:
    """Search the wiki index for pages matching a keyword.

    Args:
        keyword: Search term to look for in titles, summaries, and tags.
    """
    wiki = _get_wiki()
    entries = wiki.index.search(keyword)
    if not entries:
        return f"No pages found for: {keyword}"

    lines = [f"Found {len(entries)} page(s):"]
    for entry in entries:
        lines.append(f"- [{entry.title}]({entry.path}) — {entry.summary}")
    return "\n".join(lines)


@mcp.tool()
def wiki_read(page_path: str) -> str:
    """Read a wiki page by its relative path.

    Args:
        page_path: Relative path within the wiki (e.g. 'summaries/attention.md').
    """
    wiki = _get_wiki()
    page = wiki.read_page(page_path)
    if page is None:
        return f"Error: Page not found: {page_path}"
    return page.to_markdown()


@mcp.tool()
def wiki_write(page_path: str, content: str) -> str:
    """Write or overwrite a wiki page.

    Args:
        page_path: Relative path within the wiki directory.
        content: Full markdown content of the page (with frontmatter).
    """
    wiki = _get_wiki()
    path = wiki.wiki_dir / page_path
    page = parse_wiki_page(content, page_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page.to_markdown(), encoding="utf-8")
    return f"Page written: {page_path}"


@mcp.tool()
def wiki_query(question: str, save: bool = False) -> str:
    """Query the wiki with a natural language question.

    Args:
        question: The question to answer.
        save: If true, save the answer as a synthesis page.
    """
    wiki = _get_wiki()
    result = wiki.query(question, save_as_synthesis=save)
    return json.dumps(asdict(result), indent=2, default=str)


@mcp.tool()
def wiki_lint() -> str:
    """Run a health check on the wiki and report issues."""
    wiki = _get_wiki()
    report = wiki.lint()
    return json.dumps(asdict(report), indent=2, default=str)


@mcp.tool()
def wiki_status() -> str:
    """Return wiki status: counts of raw sources, wiki pages, index entries."""
    wiki = _get_wiki()
    status = wiki.status()
    lines = ["# Wiki Status", ""]
    for key, value in status.items():
        lines.append(f"- {key.replace('_', ' ').title()}: {value}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Obsidian vault tools
# ---------------------------------------------------------------------------


@mcp.tool()
def obsidian_init(attachment_folder: str = "raw/assets") -> str:
    """Initialize the wiki directory as an Obsidian vault.

    Creates .obsidian/ with LLM-Wiki-friendly defaults: attachment
    folder, ignore filters for a clean graph view.

    Args:
        attachment_folder: Relative path for downloaded images/assets.
    """
    from llm_patch.wiki.obsidian import ObsidianConfig

    wiki = _get_wiki()
    cfg = ObsidianConfig(enabled=True, attachment_folder=attachment_folder)
    vault = wiki.enable_obsidian(cfg)
    return (
        f"Obsidian vault initialised at {vault.root}\n"
        f"Attachment folder: {attachment_folder}\n"
        f"Ignore filters: {', '.join(cfg.ignore_filters)}"
    )


@mcp.tool()
def obsidian_graph(summary_only: bool = False) -> str:
    """Export the wiki knowledge graph as JSON.

    Returns nodes (pages) and edges (wikilinks) suitable for
    Obsidian's graph view or external visualization.

    Args:
        summary_only: If true, return node/edge counts only.
    """
    wiki = _get_wiki()
    graph_data = wiki.graph()
    if summary_only:
        return f"Nodes: {graph_data.node_count}\nEdges: {graph_data.edge_count}"
    return wiki.export_graph()


@mcp.tool()
def obsidian_status() -> str:
    """Show Obsidian vault status and graph metrics."""
    wiki = _get_wiki()
    vault = wiki.obsidian
    if vault is None or not vault.is_vault:
        return "Obsidian vault: not configured. Use obsidian_init to set up."

    app_cfg = vault.read_app_config()
    graph_data = wiki.graph()
    lines = [
        "# Obsidian Status",
        "",
        f"- Root: {vault.root}",
        f"- Attachment folder: {app_cfg.get('attachmentFolderPath', '(default)')}",
        f"- Ignore filters: {', '.join(app_cfg.get('userIgnoreFilters', []))}",
        f"- Graph nodes: {graph_data.node_count}",
        f"- Graph edges: {graph_data.edge_count}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Adapter Market tools (Distributed Knowledge Registry)
# ---------------------------------------------------------------------------


@mcp.tool()
def search_knowledge_hub(query: str, limit: int = 10) -> list[dict]:  # type: ignore[type-arg]
    """Search the registry hub for adapters matching *query*.

    Args:
        query: Free-form text — frameworks, tags, descriptions.
        limit: Maximum number of results.

    Returns:
        A list of manifest dicts (one per hit).

    Raises:
        RegistryUnavailableError: If no registry client is configured.
    """
    from llm_patch.core.interfaces import IAdapterRegistryClient

    registry = _require_registry()
    assert isinstance(registry, IAdapterRegistryClient)
    return [m.model_dump(mode="json") for m in registry.search(query, limit=limit)]


@mcp.tool()
def pull_hub_adapter(ref: str) -> dict:  # type: ignore[type-arg]
    """Download an adapter from the hub (no attach).

    Args:
        ref: ``hub://owner/name[:version]`` reference.
    """
    from llm_patch.core.interfaces import IAdapterRegistryClient
    from llm_patch.core.models import AdapterRef

    registry = _require_registry()
    assert isinstance(registry, IAdapterRegistryClient)
    parsed = AdapterRef.parse(ref)
    return registry.pull(parsed).model_dump(mode="json")


@mcp.tool()
def load_hub_adapter(ref: str) -> dict:  # type: ignore[type-arg]
    """Pull *ref* from the hub and attach it to the running model.

    This is the canonical agent-facing hot-swap entry point.

    Args:
        ref: ``hub://owner/name[:version]`` reference.

    Returns:
        Manifest of the now-active adapter.
    """
    from llm_patch.core.interfaces import IRuntimeAdapterController
    from llm_patch.core.models import AdapterRef

    controller = _require_controller()
    assert isinstance(controller, IRuntimeAdapterController)
    parsed = AdapterRef.parse(ref)
    return controller.attach(parsed).model_dump(mode="json")


@mcp.tool()
def unload_hub_adapter(adapter_id: str) -> dict:  # type: ignore[type-arg]
    """Detach an active adapter by id."""
    from llm_patch.core.interfaces import IRuntimeAdapterController

    controller = _require_controller()
    assert isinstance(controller, IRuntimeAdapterController)
    controller.detach(adapter_id)
    return {"detached": adapter_id, "active": controller.active()}


@mcp.tool()
def list_active_adapters() -> list[str]:
    """Return the ids of all currently-attached adapters."""
    from llm_patch.core.interfaces import IRuntimeAdapterController

    controller = _require_controller()
    assert isinstance(controller, IRuntimeAdapterController)
    return controller.active()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(base_dir: str | Path | None = None) -> None:
    """Start the MCP server.

    If ``base_dir`` is not provided, reads from the ``LLM_PATCH_WIKI_DIR``
    environment variable.
    """
    import os

    if base_dir is None:
        base_dir = os.environ.get("LLM_PATCH_WIKI_DIR", ".")

    configure(base_dir)
    mcp.run()


if __name__ == "__main__":
    run()
