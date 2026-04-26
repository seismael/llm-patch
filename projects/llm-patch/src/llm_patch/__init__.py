"""llm_patch - Generic Ingest -> Compile -> Attach -> Use framework.

Convert text documents into LoRA adapter weights, attach them to any
HuggingFace model, and serve the patched model for inference.

Public symbols are loaded **lazily** via :pep:`562` so that importing
:mod:`llm_patch` (e.g. through the CLI) does not eagerly pull in
``torch``, ``peft`` or ``transformers``. Each name in :data:`__all__`
resolves on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any, Final

__version__ = "1.0.0rc1"


# Mapping of public symbol -> dotted module path that owns it.
_LAZY_EXPORTS: Final[dict[str, str]] = {
    # llm_patch.attach
    "HFModelProvider": "llm_patch.attach",
    "PeftAdapterLoader": "llm_patch.attach",
    # llm_patch.core.config
    "AgentConfig": "llm_patch.core.config",
    "AttachConfig": "llm_patch.core.config",
    "GeneratorConfig": "llm_patch.core.config",
    "ModelSpec": "llm_patch.core.config",
    "ServerConfig": "llm_patch.core.config",
    "StorageConfig": "llm_patch.core.config",
    "WatcherConfig": "llm_patch.core.config",
    "WikiConfig": "llm_patch.core.config",
    # llm_patch.core.interfaces
    "IAdapterCache": "llm_patch.core.interfaces",
    "IAdapterLoader": "llm_patch.core.interfaces",
    "IAdapterRegistryClient": "llm_patch.core.interfaces",
    "IAdapterRepository": "llm_patch.core.interfaces",
    "IAgentRuntime": "llm_patch.core.interfaces",
    "IDataSource": "llm_patch.core.interfaces",
    "IKnowledgeStream": "llm_patch.core.interfaces",
    "IModelProvider": "llm_patch.core.interfaces",
    "IRuntimeAdapterController": "llm_patch.core.interfaces",
    "IWeightGenerator": "llm_patch.core.interfaces",
    # llm_patch.core.models
    "AdapterManifest": "llm_patch.core.models",
    "AdapterRef": "llm_patch.core.models",
    "ChatMessage": "llm_patch.core.models",
    "ChatResponse": "llm_patch.core.models",
    "ChatRole": "llm_patch.core.models",
    "DataSourceDescriptor": "llm_patch.core.models",
    "DocumentContext": "llm_patch.core.models",
    "GenerationOptions": "llm_patch.core.models",
    "ModelHandle": "llm_patch.core.models",
    # llm_patch.core.plugins
    "PluginKind": "llm_patch.core.plugins",
    "PluginLoader": "llm_patch.core.plugins",
    "PluginSpec": "llm_patch.core.plugins",
    # llm_patch.generators (heavy: torch)
    "SakanaT2LGenerator": "llm_patch.generators.sakana_t2l",
    # llm_patch.orchestrator (legacy shim)
    "KnowledgeFusionOrchestrator": "llm_patch.orchestrator",
    # llm_patch.pipelines
    "CompilePipeline": "llm_patch.pipelines",
    "UsePipeline": "llm_patch.pipelines",
    "WikiPipeline": "llm_patch.pipelines",
    # llm_patch.runtime
    "ChatSession": "llm_patch.runtime",
    "PeftAgentRuntime": "llm_patch.runtime",
    "PeftRuntimeController": "llm_patch.runtime",
    # llm_patch.sources.markdown
    "MarkdownDataSource": "llm_patch.sources.markdown",
    "MarkdownDirectoryWatcher": "llm_patch.sources.markdown",
    # llm_patch.sources.wiki
    "WikiDataSource": "llm_patch.sources.wiki",
    "WikiDocumentAggregator": "llm_patch.sources.wiki",
    "WikiKnowledgeSource": "llm_patch.sources.wiki",
    # llm_patch.storage (heavy: torch via local_safetensors)
    "LRUAdapterCache": "llm_patch.storage.lru_cache",
    "LocalSafetensorsRepository": "llm_patch.storage.local_safetensors",
    # llm_patch.wiki
    "IWikiAgent": "llm_patch.wiki",
    "WikiManager": "llm_patch.wiki",
    "WikiSchema": "llm_patch.wiki",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy-attribute resolver.

    Resolves names declared in :data:`_LAZY_EXPORTS` on first access and
    caches them in module ``globals()``. Heavy dependencies (``torch``,
    ``peft``, ``transformers``) are only imported when the corresponding
    public symbol is actually requested.
    """
    target_module = _LAZY_EXPORTS.get(name)
    if target_module is None:
        raise AttributeError(f"module 'llm_patch' has no attribute {name!r}")
    module = importlib.import_module(target_module)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))


if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from llm_patch.attach import HFModelProvider, PeftAdapterLoader
    from llm_patch.core.config import (
        AgentConfig,
        AttachConfig,
        GeneratorConfig,
        ModelSpec,
        ServerConfig,
        StorageConfig,
        WatcherConfig,
        WikiConfig,
    )
    from llm_patch.core.interfaces import (
        IAdapterCache,
        IAdapterLoader,
        IAdapterRegistryClient,
        IAdapterRepository,
        IAgentRuntime,
        IDataSource,
        IKnowledgeStream,
        IModelProvider,
        IRuntimeAdapterController,
        IWeightGenerator,
    )
    from llm_patch.core.models import (
        AdapterManifest,
        AdapterRef,
        ChatMessage,
        ChatResponse,
        ChatRole,
        DataSourceDescriptor,
        DocumentContext,
        GenerationOptions,
        ModelHandle,
    )
    from llm_patch.core.plugins import PluginKind, PluginLoader, PluginSpec
    from llm_patch.generators.sakana_t2l import SakanaT2LGenerator
    from llm_patch.orchestrator import KnowledgeFusionOrchestrator
    from llm_patch.pipelines import CompilePipeline, UsePipeline, WikiPipeline
    from llm_patch.runtime import ChatSession, PeftAgentRuntime, PeftRuntimeController
    from llm_patch.sources.markdown import (
        MarkdownDataSource,
        MarkdownDirectoryWatcher,
    )
    from llm_patch.sources.wiki import (
        WikiDataSource,
        WikiDocumentAggregator,
        WikiKnowledgeSource,
    )
    from llm_patch.storage import LRUAdapterCache
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
    from llm_patch.wiki import IWikiAgent, WikiManager, WikiSchema


__all__ = [
    "AdapterManifest",
    "AdapterRef",
    "AgentConfig",
    "AttachConfig",
    "ChatMessage",
    "ChatResponse",
    "ChatRole",
    "ChatSession",
    "CompilePipeline",
    "DataSourceDescriptor",
    "DocumentContext",
    "GenerationOptions",
    "GeneratorConfig",
    "HFModelProvider",
    "IAdapterCache",
    "IAdapterLoader",
    "IAdapterRegistryClient",
    "IAdapterRepository",
    "IAgentRuntime",
    "IDataSource",
    "IKnowledgeStream",
    "IModelProvider",
    "IRuntimeAdapterController",
    "IWeightGenerator",
    "IWikiAgent",
    "KnowledgeFusionOrchestrator",
    "LRUAdapterCache",
    "LocalSafetensorsRepository",
    "MarkdownDataSource",
    "MarkdownDirectoryWatcher",
    "ModelHandle",
    "ModelSpec",
    "PeftAdapterLoader",
    "PeftAgentRuntime",
    "PeftRuntimeController",
    "PluginKind",
    "PluginLoader",
    "PluginSpec",
    "SakanaT2LGenerator",
    "ServerConfig",
    "StorageConfig",
    "UsePipeline",
    "WatcherConfig",
    "WikiConfig",
    "WikiDataSource",
    "WikiDocumentAggregator",
    "WikiKnowledgeSource",
    "WikiManager",
    "WikiPipeline",
    "WikiSchema",
]
