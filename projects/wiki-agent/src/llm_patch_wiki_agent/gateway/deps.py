"""Composition root + per-adapter runtime cache for the wiki-agent gateway.

The :class:`GatewayContext` owns:
* a :class:`SidecarMetadataRegistry` (routing-side enrichment of manifests),
* an :class:`IAdapterRouter` (defaults to :class:`MetadataExactMatchRouter`),
* an :class:`engine.UsePipeline` (load-base-model + attach-adapter),
* a thread-safe runtime cache keyed by ``adapter_id`` so we never reload
  the base model twice for the same adapter.

The class accepts injected factory callables to keep the gateway testable
without torch / HuggingFace dependencies.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import llm_patch as engine
from llm_patch_shared import ConfigurationError, ResourceNotFoundError

from llm_patch_wiki_agent.agent import WikiAgentConfig
from llm_patch_wiki_agent.registry import SidecarMetadataRegistry
from llm_patch_wiki_agent.routing import (
    IAdapterRouter,
    MetadataExactMatchRouter,
)

logger = logging.getLogger(__name__)


class _UsePipelineRunner(Protocol):
    def build_agent(
        self,
        model_id: str,
        adapter_ids: list[str] | None = None,
        **model_kwargs: object,
    ) -> engine.IAgentRuntime: ...


class _CompilePipelineRunner(Protocol):
    def process_document(
        self, context: engine.DocumentContext
    ) -> engine.AdapterManifest: ...


@dataclass(frozen=True, slots=True)
class _RuntimeKwargs:
    model_id: str
    device_map: str
    dtype: str


class GatewayContext:
    """Dependency-injection container for the FastAPI gateway and MCP server.

    The two phases (HTTP + MCP) share the same context so an adapter compiled
    via MCP's ``internalize_knowledge`` becomes immediately routable via HTTP.
    """

    def __init__(
        self,
        *,
        config: WikiAgentConfig,
        repository: engine.IAdapterRepository,
        metadata_registry: SidecarMetadataRegistry,
        router: IAdapterRouter,
        use_pipeline: _UsePipelineRunner,
        compile_pipeline: _CompilePipelineRunner | None = None,
    ) -> None:
        self._config = config
        self._repository = repository
        self._metadata_registry = metadata_registry
        self._router = router
        self._use_pipeline = use_pipeline
        self._compile_pipeline = compile_pipeline
        self._runtime_cache: dict[str, engine.IAgentRuntime] = {}
        self._cache_lock = threading.Lock()

    # ── Construction ──────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: WikiAgentConfig,
        *,
        router_factory: Callable[[SidecarMetadataRegistry], IAdapterRouter] | None = None,
        repository_factory: Callable[
            [engine.StorageConfig], engine.IAdapterRepository
        ]
        | None = None,
        model_provider_factory: Callable[[], engine.IModelProvider] | None = None,
        adapter_loader_factory: Callable[[], engine.IAdapterLoader] | None = None,
        use_pipeline_factory: Callable[
            [engine.IModelProvider, engine.IAdapterLoader, engine.IAdapterRepository],
            _UsePipelineRunner,
        ]
        | None = None,
    ) -> GatewayContext:
        """Build a context using the engine's public defaults."""
        repo_factory = repository_factory or engine.LocalSafetensorsRepository
        repository = repo_factory(engine.StorageConfig(output_dir=config.adapter_dir))

        metadata_registry = SidecarMetadataRegistry(config.adapter_dir)
        router_ctor = router_factory or MetadataExactMatchRouter
        router = router_ctor(metadata_registry)

        provider = (model_provider_factory or engine.HFModelProvider)()
        loader = (adapter_loader_factory or engine.PeftAdapterLoader)()
        use_factory = use_pipeline_factory or (
            lambda mp, al, repo: engine.UsePipeline(
                model_provider=mp, adapter_loader=al, repository=repo
            )
        )
        use_pipeline = use_factory(provider, loader, repository)

        return cls(
            config=config,
            repository=repository,
            metadata_registry=metadata_registry,
            router=router,
            use_pipeline=use_pipeline,
        )

    # ── Read-only accessors ───────────────────────────────────────────

    @property
    def config(self) -> WikiAgentConfig:
        return self._config

    @property
    def repository(self) -> engine.IAdapterRepository:
        return self._repository

    @property
    def metadata_registry(self) -> SidecarMetadataRegistry:
        return self._metadata_registry

    @property
    def router(self) -> IAdapterRouter:
        return self._router

    @property
    def compile_pipeline(self) -> _CompilePipelineRunner | None:
        return self._compile_pipeline

    # ── Behaviour ─────────────────────────────────────────────────────

    def list_adapter_entries(self) -> list[dict[str, object]]:
        """Join engine manifests with sidecar metadata for the gateway list view."""
        manifests = {m.adapter_id: m for m in self._repository.list_adapters()}
        records = self._metadata_registry.list_all()
        seen: set[str] = set()
        rows: list[dict[str, object]] = []

        for record in records:
            seen.add(record.adapter_id)
            manifest = manifests.get(record.adapter_id)
            rows.append(
                {
                    "adapter_id": record.adapter_id,
                    "context_id": record.context_id,
                    "tags": record.tags,
                    "summary": record.summary,
                    "storage_uri": manifest.storage_uri if manifest else None,
                }
            )
        # Surface engine-side adapters that have no sidecar yet.
        for adapter_id, manifest in manifests.items():
            if adapter_id in seen:
                continue
            rows.append(
                {
                    "adapter_id": adapter_id,
                    "context_id": None,
                    "tags": (),
                    "summary": None,
                    "storage_uri": manifest.storage_uri,
                }
            )
        rows.sort(key=lambda row: str(row["adapter_id"]))
        return rows

    def runtime_for(self, adapter_id: str) -> engine.IAgentRuntime:
        """Return (and cache) an inference runtime with ``adapter_id`` attached."""
        if not self._config.model_id:
            raise ConfigurationError(
                "WikiAgentConfig.model_id is required to serve inference."
            )
        if not self._repository.exists(adapter_id):
            raise ResourceNotFoundError(f"Adapter not found: {adapter_id!r}")

        with self._cache_lock:
            cached = self._runtime_cache.get(adapter_id)
            if cached is not None:
                return cached
            runtime = self._use_pipeline.build_agent(
                self._config.model_id,
                adapter_ids=[adapter_id],
                dtype=self._config.model_dtype,
                device_map=self._config.model_device_map,
            )
            self._runtime_cache[adapter_id] = runtime
            return runtime

    def invalidate_runtime(self, adapter_id: str) -> None:
        with self._cache_lock:
            self._runtime_cache.pop(adapter_id, None)

    def refresh_router(self) -> None:
        self._router.refresh()

    # Allow callers (MCP) to attach a compile pipeline for live ingestion.
    def attach_compile_pipeline(self, pipeline: _CompilePipelineRunner) -> None:
        self._compile_pipeline = pipeline
