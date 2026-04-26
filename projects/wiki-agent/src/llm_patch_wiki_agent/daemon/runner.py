"""WikiCompileDaemon — wraps :class:`engine.CompilePipeline` to also persist
wiki-agent-side :class:`AdapterMetadata` sidecars.

The engine ``CompilePipeline`` is generic and intentionally has no knowledge of
``context_id`` / ``tags`` / ``summary``. This wrapper sits *above* the pipeline:

    Source → CompilePipeline.process_document → AdapterManifest
                                                ↓
                                  SidecarMetadataRegistry.save(AdapterMetadata)

It exposes ``run_once`` (batch) and ``start``/``stop`` (live, when the source
also implements ``IKnowledgeStream``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import llm_patch as engine
from llm_patch_shared import ConfigurationError, IntegrationError

from llm_patch_wiki_agent.agent import WikiAgentConfig
from llm_patch_wiki_agent.registry import AdapterMetadata, SidecarMetadataRegistry

logger = logging.getLogger(__name__)


class _CompileRunner(Protocol):
    def process_document(self, context: engine.DocumentContext) -> engine.AdapterManifest: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DaemonResult:
    """Outcome of a daemon batch run.

    Attributes:
        manifests: Engine manifests produced by the compile pipeline.
        metadata: Sidecar metadata saved by the daemon (one per manifest).
    """

    manifests: tuple[engine.AdapterManifest, ...]
    metadata: tuple[AdapterMetadata, ...]


def _derive_metadata(
    document: engine.DocumentContext,
    manifest: engine.AdapterManifest,
) -> AdapterMetadata:
    """Build :class:`AdapterMetadata` from a compiled document + manifest."""
    raw: dict[str, Any] = dict(document.metadata or {})

    context_id_value = raw.get("context_id") or raw.get("id") or document.document_id
    context_id = str(context_id_value) if context_id_value is not None else None

    tags_value = raw.get("tags", ())
    if isinstance(tags_value, (list, tuple)):
        tags = tuple(str(tag) for tag in tags_value)
    elif isinstance(tags_value, str):
        tags = (tags_value,)
    else:
        tags = ()

    summary_value = raw.get("summary") or raw.get("description")
    summary = str(summary_value) if summary_value else None

    source_path_value = raw.get("source_path") or raw.get("path")
    source_path = str(source_path_value) if source_path_value else None

    return AdapterMetadata(
        adapter_id=manifest.adapter_id,
        context_id=context_id,
        tags=tags,
        summary=summary,
        source_path=source_path,
    )


class WikiCompileDaemon:
    """Compile wiki pages to LoRA adapters and persist routing metadata.

    The daemon owns the wiring between an :class:`engine.IDataSource`, an
    :class:`engine.CompilePipeline`, and a :class:`SidecarMetadataRegistry`.

    It supports two modes:

    * :meth:`run_once` — batch compile every document in the source and write
      sidecars. Suitable for CI/CD pipelines.
    * :meth:`start` / :meth:`stop` — register a live callback on the source
      (when it also implements :class:`engine.IKnowledgeStream`) so changes
      are compiled incrementally.
    """

    def __init__(
        self,
        *,
        source: engine.IDataSource,
        pipeline: _CompileRunner,
        metadata_registry: SidecarMetadataRegistry,
        stream: engine.IKnowledgeStream | None = None,
    ) -> None:
        self._source = source
        self._pipeline = pipeline
        self._registry = metadata_registry
        self._stream = stream

        if self._stream is not None:
            self._stream.subscribe(self._on_document_changed)

    # ── Construction helpers ──────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: WikiAgentConfig,
        metadata_registry: SidecarMetadataRegistry | None = None,
        *,
        knowledge_source_factory: Callable[
            [engine.WatcherConfig, bool], engine.IDataSource
        ]
        | None = None,
        generator_factory: Callable[[engine.GeneratorConfig], engine.IWeightGenerator]
        | None = None,
        repository_factory: Callable[[engine.StorageConfig], engine.IAdapterRepository]
        | None = None,
    ) -> WikiCompileDaemon:
        """Build a daemon from a :class:`WikiAgentConfig`.

        Mirrors the factory pattern used by :class:`WikiAgent` so the same
        engine pieces are constructed consistently.
        """
        if config.wiki_dir is None:
            raise ConfigurationError(
                "WikiAgentConfig.wiki_dir is required for the compile daemon."
            )
        if config.checkpoint_dir is None:
            raise ConfigurationError(
                "WikiAgentConfig.checkpoint_dir is required for the compile daemon."
            )

        source_cfg = engine.WatcherConfig(
            directory=config.wiki_dir,
            patterns=list(config.source_patterns),
            recursive=config.recursive,
        )
        generator_cfg = engine.GeneratorConfig(
            checkpoint_dir=config.checkpoint_dir,
            device=config.generator_device,
        )

        ks_factory = knowledge_source_factory or (
            lambda cfg, agg: engine.WikiKnowledgeSource(cfg, aggregate=agg)
        )
        gen_factory = generator_factory or engine.SakanaT2LGenerator
        repo_factory = repository_factory or engine.LocalSafetensorsRepository

        source = ks_factory(source_cfg, config.aggregate_links)
        generator = gen_factory(generator_cfg)
        repository = repo_factory(engine.StorageConfig(output_dir=config.adapter_dir))
        registry = metadata_registry or SidecarMetadataRegistry(config.adapter_dir)

        # If the source is also a stream, wire it for live mode.
        stream = source if isinstance(source, engine.IKnowledgeStream) else None
        pipeline = engine.CompilePipeline(
            source=source,
            generator=generator,
            repository=repository,
        )
        return cls(
            source=source,
            pipeline=pipeline,
            metadata_registry=registry,
            stream=stream,
        )

    # ── Public API ────────────────────────────────────────────────────

    def run_once(self) -> DaemonResult:
        """Compile every document the source yields and write sidecars."""
        manifests: list[engine.AdapterManifest] = []
        metadata: list[AdapterMetadata] = []
        documents: Iterable[engine.DocumentContext] = self._source.fetch_all()

        for document in documents:
            try:
                manifest = self._pipeline.process_document(document)
            except (RuntimeError, ValueError, OSError) as exc:
                msg = f"Compilation failed for document_id={document.document_id!r}."
                raise IntegrationError(msg) from exc
            sidecar_metadata = _derive_metadata(document, manifest)
            self._registry.save(sidecar_metadata)
            manifests.append(manifest)
            metadata.append(sidecar_metadata)
            logger.info(
                "Compiled %s → %s (context_id=%s)",
                document.document_id,
                manifest.storage_uri,
                sidecar_metadata.context_id,
            )

        return DaemonResult(manifests=tuple(manifests), metadata=tuple(metadata))

    def start(self) -> None:
        """Begin live compilation if the source supports streaming."""
        if self._stream is None:
            raise ConfigurationError(
                "This source does not implement IKnowledgeStream; live mode is unavailable."
            )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    # ── Internals ─────────────────────────────────────────────────────

    def _on_document_changed(self, document: engine.DocumentContext) -> None:
        try:
            manifest = self._pipeline.process_document(document)
        except (RuntimeError, ValueError, OSError):
            logger.exception(
                "Live compile failed for document_id=%s", document.document_id
            )
            return
        sidecar_metadata = _derive_metadata(document, manifest)
        self._registry.save(sidecar_metadata)
        logger.info(
            "Live compile: %s → %s (context_id=%s)",
            document.document_id,
            manifest.storage_uri,
            sidecar_metadata.context_id,
        )

    @property
    def adapter_dir(self) -> Path:
        return self._registry.directory
