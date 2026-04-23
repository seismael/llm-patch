"""Core domain models, interfaces, and configuration."""

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

__all__ = [
    "AdapterManifest",
    "AgentConfig",
    "AttachConfig",
    "ChatMessage",
    "ChatResponse",
    "ChatRole",
    "DataSourceDescriptor",
    "DocumentContext",
    "GenerationOptions",
    "GeneratorConfig",
    "IAdapterLoader",
    "IAdapterRepository",
    "IAgentRuntime",
    "IDataSource",
    "IKnowledgeStream",
    "IModelProvider",
    "IWeightGenerator",
    "ModelHandle",
    "ModelSpec",
    "ServerConfig",
    "StorageConfig",
    "WatcherConfig",
    "WikiConfig",
]
