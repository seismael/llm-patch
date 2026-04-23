"""PEFT adapter loader — attaches LoRA adapters onto a loaded base model."""

from __future__ import annotations

import logging

from llm_patch_shared import DependencyError, IntegrationError

from llm_patch.core.interfaces import IAdapterLoader
from llm_patch.core.models import AdapterManifest, ModelHandle

logger = logging.getLogger(__name__)


class PeftAdapterLoader(IAdapterLoader):
    """Attaches LoRA adapter weights via the PEFT library.

    Supports stacking multiple adapters onto the same model and
    switching the active adapter.
    """

    def attach(
        self,
        handle: ModelHandle,
        manifest: AdapterManifest,
    ) -> ModelHandle:
        """Load adapter from ``manifest.storage_uri`` onto *handle*.

        Returns a new ``ModelHandle`` with the adapter recorded in
        ``attached_adapters``.
        """
        try:
            from peft import PeftModel
        except ImportError as exc:
            msg = "PEFT is unavailable. Install the llm-patch adapter runtime dependencies."
            raise DependencyError(msg) from exc

        adapter_name = manifest.adapter_id
        model = handle.model

        try:
            if handle.attached_adapters:
                # Already a PeftModel — load an additional adapter
                logger.info("Stacking adapter %s onto existing PeftModel", adapter_name)
                model.load_adapter(manifest.storage_uri, adapter_name=adapter_name)
                model.set_adapter(adapter_name)
            else:
                logger.info("Attaching first adapter %s", adapter_name)
                model = PeftModel.from_pretrained(
                    model,
                    manifest.storage_uri,
                    adapter_name=adapter_name,
                )
        except (OSError, ValueError) as exc:
            msg = f"Failed to attach adapter '{adapter_name}' from '{manifest.storage_uri}'."
            raise IntegrationError(msg) from exc

        return ModelHandle(
            model=model,
            tokenizer=handle.tokenizer,
            base_model_id=handle.base_model_id,
            attached_adapters=(*handle.attached_adapters, adapter_name),
            device=handle.device,
        )
