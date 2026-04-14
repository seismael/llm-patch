"""llm_patch — Instant Knowledge Internalization (Doc-to-LoRA).

Convert text documents into LoRA adapter weights using hypernetworks.
"""

__version__ = "0.1.0"

from llm_patch.core.config import GeneratorConfig, StorageConfig, WatcherConfig
from llm_patch.core.interfaces import IAdapterRepository, IKnowledgeSource, IWeightGenerator
from llm_patch.core.models import AdapterManifest, DocumentContext
from llm_patch.orchestrator import KnowledgeFusionOrchestrator
from llm_patch.sources.markdown_watcher import MarkdownDirectoryWatcher
from llm_patch.sources.wiki_source import WikiDocumentAggregator, WikiKnowledgeSource

# Torch-dependent modules — imported lazily to allow use without torch installed.
try:
    from llm_patch.generators.sakana_t2l import SakanaT2LGenerator
    from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
except (ImportError, OSError):  # pragma: no cover
    pass

__all__ = [
    "AdapterManifest",
    "DocumentContext",
    "GeneratorConfig",
    "IAdapterRepository",
    "IKnowledgeSource",
    "IWeightGenerator",
    "KnowledgeFusionOrchestrator",
    "LocalSafetensorsRepository",
    "MarkdownDirectoryWatcher",
    "SakanaT2LGenerator",
    "StorageConfig",
    "WatcherConfig",
    "WikiDocumentAggregator",
    "WikiKnowledgeSource",
]
