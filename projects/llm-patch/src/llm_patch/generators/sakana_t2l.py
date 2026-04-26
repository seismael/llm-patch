"""Sakana AI Text-to-LoRA generator (Strategy Pattern).

Wraps the hyper_llm_modulator library from SakanaAI/text-to-lora to generate
LoRA adapter weights from text document content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from llm_patch_utils import ConfigurationError, DependencyError, IntegrationError

from llm_patch.core.config import GeneratorConfig
from llm_patch.core.interfaces import IWeightGenerator
from llm_patch.core.models import DocumentContext

logger = logging.getLogger(__name__)

_REQUIRED_FILES = ("hypermod.pt", "args.yaml", "adapter_config.json")


class SakanaT2LGenerator(IWeightGenerator):
    """Generates LoRA adapter weights using Sakana AI's Text-to-LoRA hypernetwork.

    This generator loads a pre-trained T2L checkpoint and uses its hypernetwork
    to convert arbitrary text into LoRA weight matrices in a single forward pass.

    Requires the ``hyper_llm_modulator`` package from
    https://github.com/SakanaAI/text-to-lora

    Args:
        config: Generator configuration specifying checkpoint path and device.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        self._config = config
        self._device = torch.device(config.device)
        checkpoint_dir = Path(config.checkpoint_dir)

        # Validate checkpoint directory
        for required in _REQUIRED_FILES:
            path = checkpoint_dir / required
            if not path.exists():
                msg = f"Required file not found: {path}"
                raise ConfigurationError(msg)

        # Import Sakana modules (deferred to avoid hard import at module level)
        try:
            from hyper_llm_modulator.hyper_modulator import load_hypermod_checkpoint
            from hyper_llm_modulator.utils import get_layers
        except ImportError as exc:
            msg = (
                "hyper_llm_modulator is unavailable. Install the Sakana Text-to-LoRA runtime "
                "dependencies before compiling wiki adapters."
            )
            raise DependencyError(msg) from exc

        checkpoint_path = str(checkpoint_dir / "hypermod.pt")
        logger.info("Loading T2L checkpoint from %s", checkpoint_path)

        try:
            (
                self._args,
                self._hypermod,
                self._model,
                self._tokenizer,
                self._emb_model,
                self._emb_tokenizer,
                self._task_desc_format_fn,
                self._pooling_fn,
            ) = load_hypermod_checkpoint(checkpoint_path, self._device)
        except (OSError, RuntimeError, ValueError) as exc:
            msg = f"Failed to load Text-to-LoRA checkpoint from '{checkpoint_path}'."
            raise IntegrationError(msg) from exc

        # Compute layer indices for all transformer layers
        layers = get_layers(self._model)
        self._layer_indices = torch.tensor(
            range(len(layers)), dtype=torch.long, device=self._device
        )

        # Store PEFT config from the hypermodulator
        self._peft_config = self._hypermod.peft_config

        logger.info(
            "T2L generator ready — %d layers, rank=%d, device=%s",
            len(layers),
            self._peft_config.r,
            self._device,
        )

    @torch.inference_mode()  # type: ignore[untyped-decorator]
    def generate(self, context: DocumentContext) -> dict[str, torch.Tensor]:
        """Generate LoRA weights from document content.

        Embeds the document text using the T2L embedding model, encodes it
        through the task encoder, and generates LoRA A/B matrices for all
        transformer layers via the hypernetwork.

        Args:
            context: Document whose content will be converted to weights.

        Returns:
            State dict with PEFT-format keys mapping to LoRA weight tensors.
        """
        from hyper_llm_modulator.utils import embed_texts

        logger.info("Generating weights for document: %s", context.document_id)

        # Step 1: Embed the document text
        task_emb = embed_texts(
            [context.content],
            self._emb_model,
            self._emb_tokenizer,
            self._task_desc_format_fn,
            self._pooling_fn,
            self._device,
        )

        # Step 2: Encode through the task encoder
        encoder_out = self._hypermod.task_encoder(task_emb)
        encoded_task_emb = encoder_out["encoded_task_emb"].detach()

        # Step 3: Generate LoRA state dict via hypernetwork forward pass
        lora_state_dict: dict[str, torch.Tensor] = self._hypermod.gen_lora(
            self._layer_indices, encoded_task_emb
        )

        logger.info(
            "Generated %d weight tensors for %s",
            len(lora_state_dict),
            context.document_id,
        )
        return lora_state_dict

    def get_peft_config(self) -> Any:
        """Return the PEFT LoraConfig from the loaded hypernetwork."""
        return self._peft_config
