"""llm_patch — Generic Ingest → Compile → Attach → Use framework.

Convert text documents into LoRA adapter weights, attach them to any
HuggingFace model, and serve the patched model for inference.
"""

__version__ = "0.1.0"

# ── Core interfaces & models ──────────────────────────────────────────
from llm_patch.core.interfaces import (
    IAdapterLoader,
    IAdapterRepository,
    IAgentRuntime,
    IDataSource,
    IKnowledgeStream,
    IModelProvider,
    IWeightGenerator,
)
from llm_patch.core.models import (
    AdapterManifest,
    ChatMessage,
    ChatResponse,
    ChatRole,
    DataSourceDescriptor,
    DocumentContext,
    GenerationOptions,
    ModelHandle,
)

# ── Config ────────────────────────────────────────────────────────────
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

# ── Pipelines ─────────────────────────────────────────────────────────
from llm_patch.pipelines import CompilePipeline, UsePipeline, WikiPipeline

# ── Legacy shims (preserve existing user code) ────────────────────────
from llm_patch.orchestrator import KnowledgeFusionOrchestrator
from llm_patch.sources.markdown import MarkdownDataSource, MarkdownDirectoryWatcher
from llm_patch.sources.wiki import WikiDataSource, WikiDocumentAggregator, WikiKnowledgeSource
from llm_patch.wiki import IWikiAgent, WikiManager, WikiSchema

# Torch-dependent modules — imported lazily to allow use without torch.
try:
    from llm_patch.generators.sakana_t2l import SakanaT2LGenerator
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
except (ImportError, OSError):  # pragma: no cover
    pass

__all__ = [
    # Core interfaces
    "IAdapterLoader",
    "IAdapterRepository",
    "IAgentRuntime",
    "IDataSource",
    "IKnowledgeStream",
    "IModelProvider",
    "IWeightGenerator",
    # Models
    "AdapterManifest",
    "ChatMessage",
    "ChatResponse",
    "ChatRole",
    "DataSourceDescriptor",
    "DocumentContext",
    "GenerationOptions",
    "ModelHandle",
    # Config
    "AgentConfig",
    "AttachConfig",
    "GeneratorConfig",
    "ModelSpec",
    "ServerConfig",
    "StorageConfig",
    "WatcherConfig",
    "WikiConfig",
    # Pipelines
    "CompilePipeline",
    "UsePipeline",
    "WikiPipeline",
    # Sources
    "MarkdownDataSource",
    "MarkdownDirectoryWatcher",
    "WikiDataSource",
    "WikiDocumentAggregator",
    "WikiKnowledgeSource",
    # Legacy
    "KnowledgeFusionOrchestrator",
    # Torch-dependent
    "LocalSafetensorsRepository",
    "SakanaT2LGenerator",
    # Wiki
    "IWikiAgent",
    "WikiManager",
    "WikiSchema",
]
