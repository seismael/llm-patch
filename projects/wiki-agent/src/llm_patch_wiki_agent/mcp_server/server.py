"""MCP server exposing wiki-agent tools to autonomous agents.

Tools:
* ``internalize_knowledge(path, context_id?, tags?, summary?)`` — compile a
  document into a fresh adapter and register routing metadata. The agent gets
  back an ``adapter_id`` it can immediately use via ``chat_with_adapter``.
* ``list_adapters()`` — enumerate available adapters with their metadata.
* ``chat_with_adapter(adapter_id, query)`` — bypass the router and run a
  single inference turn against a specific adapter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import llm_patch as engine
from llm_patch_utils import ConfigurationError, DependencyError

from llm_patch_wiki_agent.daemon.runner import _derive_metadata

if TYPE_CHECKING:
    from llm_patch_wiki_agent.gateway.deps import GatewayContext

logger = logging.getLogger(__name__)


def _build_default_compile_pipeline(context: "GatewayContext") -> Any:
    """Construct a :class:`engine.CompilePipeline` lazily from the gateway config."""
    cfg = context.config
    if cfg.checkpoint_dir is None:
        raise ConfigurationError(
            "WikiAgentConfig.checkpoint_dir is required for internalize_knowledge."
        )
    generator = engine.SakanaT2LGenerator(
        engine.GeneratorConfig(
            checkpoint_dir=cfg.checkpoint_dir,
            device=cfg.generator_device,
        )
    )

    # Adapt a single document into an IDataSource shim so CompilePipeline is happy.
    class _SingleDocSource(engine.IDataSource):
        def __init__(self) -> None:
            self._doc: engine.DocumentContext | None = None

        @property
        def name(self) -> str:
            return "mcp-internalize"

        def set(self, doc: engine.DocumentContext) -> None:
            self._doc = doc

        def fetch_all(self) -> list[engine.DocumentContext]:
            return [self._doc] if self._doc is not None else []

    source = _SingleDocSource()
    pipeline = engine.CompilePipeline(
        source=source, generator=generator, repository=context.repository
    )
    # Stash the source reference so callers can swap docs.
    pipeline.__dict__["_mcp_source"] = source
    return pipeline


def build_tools(context: "GatewayContext") -> dict[str, Any]:
    """Return the wiki-agent MCP tool callables bound to ``context``.

    Exposed as a separate function so unit tests can call them directly
    without instantiating a :class:`FastMCP` server.
    """

    def internalize_knowledge(
        path: str,
        context_id: str | None = None,
        tags: list[str] | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        source_path = Path(path)
        if not source_path.exists() or not source_path.is_file():
            raise ConfigurationError(f"Path does not exist or is not a file: {path}")

        document_id = source_path.stem
        document = engine.DocumentContext(
            document_id=document_id,
            content=source_path.read_text(encoding="utf-8"),
            metadata={
                "context_id": context_id or document_id,
                "tags": tags or [],
                "summary": summary,
                "source_path": str(source_path),
            },
        )

        pipeline = context.compile_pipeline or _build_default_compile_pipeline(context)
        sds = getattr(pipeline, "_mcp_source", None)
        if sds is not None:
            sds.set(document)
        manifest = pipeline.process_document(document)

        metadata = _derive_metadata(document, manifest)
        context.metadata_registry.save(metadata)
        context.refresh_router()
        context.invalidate_runtime(manifest.adapter_id)

        return {
            "adapter_id": manifest.adapter_id,
            "context_id": metadata.context_id,
            "storage_uri": manifest.storage_uri,
        }

    def list_adapters() -> list[dict[str, Any]]:
        rows = context.list_adapter_entries()
        normalized: list[dict[str, Any]] = []
        for row in rows:
            tags_value = row.get("tags") or ()
            tags_list = list(tags_value) if isinstance(tags_value, (list, tuple)) else []
            normalized.append({**row, "tags": tags_list})
        return normalized

    def chat_with_adapter(adapter_id: str, query: str) -> dict[str, Any]:
        runtime = context.runtime_for(adapter_id)
        response = runtime.chat(
            [engine.ChatMessage(role=engine.ChatRole.USER, content=query)]
        )
        return {"adapter_id": adapter_id, "answer": response.message.content}

    return {
        "internalize_knowledge": internalize_knowledge,
        "list_adapters": list_adapters,
        "chat_with_adapter": chat_with_adapter,
    }


def build_server(context: "GatewayContext") -> Any:
    """Construct a :class:`FastMCP` server bound to the shared :class:`GatewayContext`.

    Tools share state with any concurrently-running HTTP gateway: an adapter
    compiled here becomes routable on the next request after the registry
    refreshes (see ``MetadataExactMatchRouter.refresh``).
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as exc:
        raise DependencyError(
            "mcp is not installed. Install with: pip install 'llm-patch-wiki-agent[mcp]'"
        ) from exc

    server = FastMCP("llm-patch-wiki-agent")
    tools = build_tools(context)

    server.tool()(tools["internalize_knowledge"])
    server.tool()(tools["list_adapters"])
    server.tool()(tools["chat_with_adapter"])

    return server
