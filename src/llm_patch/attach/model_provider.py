"""HuggingFace model provider — loads a base model + tokenizer into memory."""

from __future__ import annotations

import logging
from typing import Any

from llm_patch.core.interfaces import IModelProvider
from llm_patch.core.models import ModelHandle

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "float16": "torch.float16",
    "bfloat16": "torch.bfloat16",
    "float32": "torch.float32",
}


class HFModelProvider(IModelProvider):
    """Loads a base model and tokenizer via HuggingFace ``transformers``.

    All heavy imports (``torch``, ``transformers``) are deferred to the
    ``load`` call so that the class can be instantiated cheaply without
    requiring a GPU-capable environment.
    """

    def load(self, model_id: str, **kwargs: Any) -> ModelHandle:
        """Load a base model.

        Accepted kwargs (passed through to ``from_pretrained``):
            dtype, device_map, trust_remote_code, low_cpu_mem_usage.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_str = kwargs.pop("dtype", "float16")
        torch_dtype = getattr(torch, dtype_str, torch.float16)
        device_map = kwargs.pop("device_map", "auto")
        trust_remote_code = kwargs.pop("trust_remote_code", False)

        logger.info("Loading base model %s (dtype=%s, device_map=%s)", model_id, dtype_str, device_map)

        tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=kwargs.pop("low_cpu_mem_usage", True),
            **kwargs,
        )

        device = str(model.device) if hasattr(model, "device") else "cpu"

        return ModelHandle(
            model=model,
            tokenizer=tokenizer,
            base_model_id=model_id,
            device=device,
        )
