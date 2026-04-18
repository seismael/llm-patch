"""Use pipeline — load model, attach adapters, build agent runtime."""

from __future__ import annotations

import logging
from pathlib import Path

from llm_patch.core.interfaces import IAdapterLoader, IAdapterRepository, IModelProvider
from llm_patch.core.models import ModelHandle

logger = logging.getLogger(__name__)


class UsePipeline:
    """High-level pipeline: load a base model, attach compiled adapters, serve.

    Args:
        model_provider: Loads the base model.
        adapter_loader: Attaches LoRA weights onto the model.
        repository: Retrieves stored adapter manifests.
    """

    def __init__(
        self,
        model_provider: IModelProvider,
        adapter_loader: IAdapterLoader,
        repository: IAdapterRepository,
    ) -> None:
        self._model_provider = model_provider
        self._adapter_loader = adapter_loader
        self._repository = repository

    def load_and_attach(
        self,
        model_id: str,
        adapter_ids: list[str] | None = None,
        **model_kwargs: object,
    ) -> ModelHandle:
        """Load a base model and optionally attach adapters.

        Args:
            model_id: HuggingFace model ID or local path.
            adapter_ids: List of adapter IDs to attach.  If ``None``,
                attaches all adapters in the repository.
            **model_kwargs: Forwarded to the model provider.

        Returns:
            A ``ModelHandle`` with all requested adapters active.
        """
        handle = self._model_provider.load(model_id, **model_kwargs)
        logger.info("Loaded base model: %s", model_id)

        if adapter_ids is None:
            manifests = self._repository.list_adapters()
        else:
            manifests = []
            for aid in adapter_ids:
                for m in self._repository.list_adapters():
                    if m.adapter_id == aid:
                        manifests.append(m)
                        break
                else:
                    logger.warning("Adapter not found: %s", aid)

        for manifest in manifests:
            handle = self._adapter_loader.attach(handle, manifest)
            logger.info("Attached adapter: %s", manifest.adapter_id)

        return handle

    def build_agent(
        self,
        model_id: str,
        adapter_ids: list[str] | None = None,
        **model_kwargs: object,
    ) -> "PeftAgentRuntime":  # noqa: F821
        """Convenience: load, attach, and wrap in an agent runtime."""
        from llm_patch.runtime.agent import PeftAgentRuntime

        handle = self.load_and_attach(model_id, adapter_ids, **model_kwargs)
        return PeftAgentRuntime(handle)
