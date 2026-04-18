"""Adapter merger — merge adapters into the base model or blend weights."""

from __future__ import annotations

import logging
from pathlib import Path

from llm_patch.core.models import ModelHandle

logger = logging.getLogger(__name__)


def merge_into_base(handle: ModelHandle, output_dir: Path) -> Path:
    """Merge the active adapter into base weights and save a standalone model.

    Args:
        handle: A ``ModelHandle`` with at least one adapter attached.
        output_dir: Directory to write the merged HF model.

    Returns:
        The output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = handle.model
    logger.info("Merging adapter(s) into base model and saving to %s", output_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(str(output_dir))
    handle.tokenizer.save_pretrained(str(output_dir))

    return output_dir


def weighted_blend(
    handle: ModelHandle,
    weights: dict[str, float],
    combined_name: str = "blended",
) -> ModelHandle:
    """Blend multiple loaded adapters using PEFT weighted combination.

    All adapters specified in *weights* must already be loaded on the
    model (via ``PeftAdapterLoader.attach``).

    Args:
        handle: Model handle with multiple adapters loaded.
        weights: Mapping of ``adapter_id → weight``.
        combined_name: Name for the blended adapter.

    Returns:
        An updated ``ModelHandle`` with the blended adapter active.
    """
    adapter_names = list(weights.keys())
    adapter_weights = [weights[n] for n in adapter_names]

    logger.info("Blending adapters %s with weights %s", adapter_names, adapter_weights)
    handle.model.add_weighted_adapter(
        adapter_names, adapter_weights, combination_type="linear", adapter_name=combined_name,
    )
    handle.model.set_adapter(combined_name)

    return ModelHandle(
        model=handle.model,
        tokenizer=handle.tokenizer,
        base_model_id=handle.base_model_id,
        attached_adapters=(*handle.attached_adapters, combined_name),
        device=handle.device,
    )
