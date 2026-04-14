#!/usr/bin/env python
"""Validate a generated LoRA adapter by loading it onto a base model (GPU required).

This script requires:
  - torch, transformers, peft installed with working CUDA
  - A base model accessible (e.g. google/gemma-2-2b-it)
  - A previously generated adapter directory with adapter_model.safetensors

Usage:
    python validate_adapter.py --adapter-dir adapters/attention-is-all-you-need \\
                               --base-model google/gemma-2-2b-it

If torch is unavailable, the script prints a placeholder message and exits.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import torch
    from peft import PeftModel  # type: ignore[import-untyped]
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-untyped]

    HAS_DEPS = True
except (ImportError, OSError):
    HAS_DEPS = False


def validate(adapter_dir: Path, base_model_name: str) -> None:
    """Load base model + adapter and run a quick inference comparison."""
    if not HAS_DEPS:
        logger.warning(
            "torch / transformers / peft not available. "
            "Skipping real validation — install dependencies and retry."
        )
        _print_placeholder(adapter_dir)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    # Load base model
    logger.info("Loading base model: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
    )

    # Inference WITHOUT adapter
    prompt = "Explain the Transformer architecture in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    logger.info("Generating WITHOUT adapter...")
    with torch.no_grad():
        base_output = base_model.generate(**inputs, max_new_tokens=64)
    base_text = tokenizer.decode(base_output[0], skip_special_tokens=True)

    # Load adapter
    logger.info("Loading adapter from: %s", adapter_dir)
    adapted_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    logger.info("Generating WITH adapter...")
    with torch.no_grad():
        adapted_output = adapted_model.generate(**inputs, max_new_tokens=64)
    adapted_text = tokenizer.decode(adapted_output[0], skip_special_tokens=True)

    # Report
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    print(f"Adapter:    {adapter_dir}")
    print(f"Base model: {base_model_name}")
    print(f"Device:     {device}")
    print("-" * 60)
    print(f"Prompt:  {prompt}")
    print(f"\n[Base]    {base_text}")
    print(f"\n[Adapted] {adapted_text}")
    print("=" * 60)


def _print_placeholder(adapter_dir: Path) -> None:
    """Show what validation *would* do when deps are missing."""
    print("\n" + "=" * 60)
    print("VALIDATION REPORT (placeholder — torch unavailable)")
    print("=" * 60)
    print(f"Adapter directory: {adapter_dir}")
    print()
    print("To run real validation, install torch + transformers + peft and")
    print("ensure a generated adapter exists at the given path.")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate a LoRA adapter via inference comparison.")
    p.add_argument(
        "--adapter-dir",
        type=Path,
        required=True,
        help="Path to the generated adapter directory.",
    )
    p.add_argument(
        "--base-model",
        type=str,
        default="google/gemma-2-2b-it",
        help="HuggingFace model ID for the base LLM.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
    args = build_parser().parse_args(argv)
    validate(args.adapter_dir.resolve(), args.base_model)


if __name__ == "__main__":
    main()
