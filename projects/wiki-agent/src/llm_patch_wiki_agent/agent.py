"""Wiki-specialized orchestration built on the public ``llm_patch`` API."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import llm_patch as engine
from llm_patch_utils import ConfigurationError, IntegrationError


class _UsePipelineRunner(Protocol):
    def build_agent(
        self,
        model_id: str,
        adapter_ids: list[str] | None = None,
        **model_kwargs: object,
    ) -> engine.IAgentRuntime:
        """Build an agent runtime for inference."""


def _default_knowledge_source_factory(
    config: engine.WatcherConfig,
    aggregate_links: bool,
) -> engine.IDataSource:
    return engine.WikiKnowledgeSource(config, aggregate=aggregate_links)


def _default_generator_factory(config: engine.GeneratorConfig) -> engine.IWeightGenerator:
    try:
        generator_cls = engine.SakanaT2LGenerator
    except AttributeError as exc:
        msg = (
            "The llm_patch public API does not currently expose SakanaT2LGenerator. "
            "Install the full generator runtime and retry."
        )
        raise IntegrationError(msg) from exc
    return generator_cls(config)


def _default_repository_factory(config: engine.StorageConfig) -> engine.IAdapterRepository:
    try:
        repository_cls = engine.LocalSafetensorsRepository
    except AttributeError as exc:
        msg = "The llm_patch public API does not currently expose LocalSafetensorsRepository."
        raise IntegrationError(msg) from exc
    return repository_cls(config)


def _default_model_provider_factory() -> engine.IModelProvider:
    return engine.HFModelProvider()


def _default_adapter_loader_factory() -> engine.IAdapterLoader:
    return engine.PeftAdapterLoader()


def _default_use_pipeline_factory(
    model_provider: engine.IModelProvider,
    adapter_loader: engine.IAdapterLoader,
    repository: engine.IAdapterRepository,
) -> _UsePipelineRunner:
    return engine.UsePipeline(
        model_provider=model_provider,
        adapter_loader=adapter_loader,
        repository=repository,
    )


@dataclass(frozen=True, slots=True)
class WikiAgentInfo:
    """Runtime metadata describing a wiki-agent configuration."""

    adapter_dir: Path
    adapter_count: int
    adapter_ids: tuple[str, ...]
    wiki_dir: Path | None = None
    model_id: str | None = None
    checkpoint_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class WikiAgentConfig:
    """Configuration for :class:`WikiAgent`.

    Attributes:
        adapter_dir: Directory where compiled LoRA adapters are stored.
        wiki_dir: Directory containing the wiki vault (markdown + frontmatter).
        model_id: HuggingFace model identifier or local path of the base LLM.
        checkpoint_dir: Sakana Text-to-LoRA checkpoint directory.
        source_patterns: Wiki page glob patterns included in compilation.
        recursive: Whether wiki compilation recurses into subdirectories.
        aggregate_links: Whether to enrich pages by following wiki links.
        generator_device: PyTorch device used by the T2L generator.
        model_device_map: Device-map hint for HuggingFace model loading.
        model_dtype: Torch dtype string used for the base model.
        system_prompt: Optional default system prompt for chat.
        max_history: Maximum number of chat messages retained in-session.
    """

    adapter_dir: Path
    wiki_dir: Path | None = None
    model_id: str | None = None
    checkpoint_dir: Path | None = None
    source_patterns: tuple[str, ...] = ("*.md",)
    recursive: bool = True
    aggregate_links: bool = True
    generator_device: str = "cuda"
    model_device_map: str = "auto"
    model_dtype: str = "float16"
    system_prompt: str | None = None
    max_history: int = 0


class WikiAgent:
    """Wiki-specialized composition over the engine's public API."""

    def __init__(
        self,
        config: WikiAgentConfig,
        *,
        knowledge_source_factory: Callable[[engine.WatcherConfig, bool], engine.IDataSource]
        | None = None,
        generator_factory: Callable[[engine.GeneratorConfig], engine.IWeightGenerator]
        | None = None,
        repository_factory: Callable[[engine.StorageConfig], engine.IAdapterRepository]
        | None = None,
        model_provider_factory: Callable[[], engine.IModelProvider] | None = None,
        adapter_loader_factory: Callable[[], engine.IAdapterLoader] | None = None,
        use_pipeline_factory: Callable[
            [engine.IModelProvider, engine.IAdapterLoader, engine.IAdapterRepository],
            _UsePipelineRunner,
        ]
        | None = None,
    ) -> None:
        self._config = config
        self._knowledge_source_factory = (
            knowledge_source_factory or _default_knowledge_source_factory
        )
        self._generator_factory = generator_factory or _default_generator_factory
        self._repository_factory = repository_factory or _default_repository_factory
        self._model_provider_factory = model_provider_factory or _default_model_provider_factory
        self._adapter_loader_factory = adapter_loader_factory or _default_adapter_loader_factory
        self._use_pipeline_factory = use_pipeline_factory or _default_use_pipeline_factory

    @property
    def config(self) -> WikiAgentConfig:
        return self._config

    def describe(self) -> WikiAgentInfo:
        adapter_ids = self._scan_adapter_ids()
        return WikiAgentInfo(
            adapter_dir=self._config.adapter_dir,
            adapter_count=len(adapter_ids),
            adapter_ids=adapter_ids,
            wiki_dir=self._config.wiki_dir,
            model_id=self._config.model_id,
            checkpoint_dir=self._config.checkpoint_dir,
        )

    def compile(
        self,
        *,
        checkpoint_dir: Path | None = None,
        generator_device: str | None = None,
    ) -> list[engine.AdapterManifest]:
        wiki_dir = self._require_wiki_dir()
        resolved_checkpoint_dir = checkpoint_dir or self._config.checkpoint_dir
        if resolved_checkpoint_dir is None:
            msg = (
                "Missing checkpoint directory. Provide checkpoint_dir in WikiAgentConfig or "
                "pass --checkpoint-dir in the CLI."
            )
            raise ConfigurationError(msg)
        if not resolved_checkpoint_dir.exists() or not resolved_checkpoint_dir.is_dir():
            msg = f"Checkpoint directory does not exist: {resolved_checkpoint_dir}"
            raise ConfigurationError(msg)

        self._config.adapter_dir.mkdir(parents=True, exist_ok=True)

        source_cfg = engine.WatcherConfig(
            directory=wiki_dir,
            patterns=list(self._config.source_patterns),
            recursive=self._config.recursive,
        )
        source = self._knowledge_source_factory(source_cfg, self._config.aggregate_links)

        generator_cfg = engine.GeneratorConfig(
            checkpoint_dir=resolved_checkpoint_dir,
            device=generator_device or self._config.generator_device,
        )
        generator = self._generator_factory(generator_cfg)
        repository = self._build_repository()

        pipeline = engine.CompilePipeline(source=source, generator=generator, repository=repository)
        return pipeline.compile_all()

    def chat(
        self,
        prompt: str,
        *,
        adapter_ids: Sequence[str] | None = None,
        model_id: str | None = None,
        system_prompt: str | None = None,
        model_device_map: str | None = None,
        model_dtype: str | None = None,
    ) -> str:
        if not prompt.strip():
            raise ConfigurationError("Prompt must not be empty.")

        resolved_model_id = model_id or self._config.model_id
        if not resolved_model_id:
            msg = "Missing model_id. Configure it on the agent or pass it explicitly to chat()."
            raise ConfigurationError(msg)

        repository = self._build_repository()
        available_manifests = repository.list_adapters()
        if not available_manifests:
            msg = (
                f"No compiled adapters were found in {self._config.adapter_dir}. "
                "Run the compile command before starting a chat session."
            )
            raise ConfigurationError(msg)

        available_ids = {manifest.adapter_id for manifest in available_manifests}
        if adapter_ids is None:
            resolved_adapter_ids = sorted(available_ids)
        else:
            requested_ids = list(adapter_ids)
            missing_ids = sorted(set(requested_ids) - available_ids)
            if missing_ids:
                msg = f"Requested adapter IDs were not found: {', '.join(missing_ids)}"
                raise ConfigurationError(msg)
            resolved_adapter_ids = requested_ids

        model_provider = self._model_provider_factory()
        adapter_loader = self._adapter_loader_factory()
        use_pipeline = self._use_pipeline_factory(model_provider, adapter_loader, repository)

        try:
            runtime = use_pipeline.build_agent(
                resolved_model_id,
                adapter_ids=resolved_adapter_ids,
                dtype=model_dtype or self._config.model_dtype,
                device_map=model_device_map or self._config.model_device_map,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            msg = f"Failed to initialize the wiki-agent runtime for model '{resolved_model_id}'."
            raise IntegrationError(msg) from exc

        session = engine.ChatSession(
            runtime,
            system_prompt=system_prompt or self._config.system_prompt,
            max_history=self._config.max_history,
        )
        return session.say(prompt)

    def _build_repository(self) -> engine.IAdapterRepository:
        repository_cfg = engine.StorageConfig(output_dir=self._config.adapter_dir)
        return self._repository_factory(repository_cfg)

    def _require_wiki_dir(self) -> Path:
        if self._config.wiki_dir is None:
            msg = "Missing wiki_dir. Configure it on the agent or pass --wiki-dir in the CLI."
            raise ConfigurationError(msg)
        if not self._config.wiki_dir.exists() or not self._config.wiki_dir.is_dir():
            msg = f"Wiki directory does not exist: {self._config.wiki_dir}"
            raise ConfigurationError(msg)
        return self._config.wiki_dir

    def _scan_adapter_ids(self) -> tuple[str, ...]:
        if not self._config.adapter_dir.exists():
            return ()

        adapter_ids: list[str] = []
        for manifest_path in sorted(self._config.adapter_dir.glob("*/manifest.json")):
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            adapter_id = payload.get("adapter_id")
            if isinstance(adapter_id, str):
                adapter_ids.append(adapter_id)

        return tuple(adapter_ids)
