"""llm_patch — Generic Ingest → Compile → Attach → Use framework.

Convert text documents into LoRA adapter weights, attach them to any
HuggingFace model, and serve the patched model for inference.
"""

__version__ = "0.1.0"

# ── Core interfaces & models ──────────────────────────────────────────
# ── Runtime / Attach ──────────────────────────────────────────────────
from llm_patch.attach import HFModelProvider, PeftAdapterLoader

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

# ── Legacy shims (preserve existing user code) ────────────────────────
from llm_patch.orchestrator import KnowledgeFusionOrchestrator

# ── Pipelines ─────────────────────────────────────────────────────────
from llm_patch.pipelines import CompilePipeline, UsePipeline, WikiPipeline
from llm_patch.runtime import ChatSession, PeftAgentRuntime
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
    "AdapterManifest",
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
    "IAdapterLoader",
    "IAdapterRepository",
    "IAgentRuntime",
    "IDataSource",
    "IKnowledgeStream",
    "IModelProvider",
    "IWeightGenerator",
    "IWikiAgent",
    "KnowledgeFusionOrchestrator",
    "LocalSafetensorsRepository",
    "MarkdownDataSource",
    "MarkdownDirectoryWatcher",
    "ModelHandle",
    "ModelSpec",
    "PeftAdapterLoader",
    "PeftAgentRuntime",
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
