"""Adapter attachment layer — load base models and attach LoRA adapters."""

from llm_patch.attach.model_provider import HFModelProvider
from llm_patch.attach.peft_loader import PeftAdapterLoader

__all__ = ["HFModelProvider", "PeftAdapterLoader"]
