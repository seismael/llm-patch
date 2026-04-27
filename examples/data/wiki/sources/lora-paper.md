---
title: "LoRA: Low-Rank Adaptation"
authors: Hu et al.
year: 2021
tags: lora, fine-tuning, parameter-efficient
---

# LoRA: Low-Rank Adaptation

LoRA freezes pre-trained model weights and injects trainable low-rank decomposition
matrices into each [[Transformer]] layer, greatly reducing trainable parameters.

## Method

For a weight matrix W0, LoRA constrains the update: W0 + delta_W = W0 + BA,
where B and A are low-rank matrices with rank r << min(d, k). During training,
W0 is frozen. A is initialized with random Gaussian, B with zeros.

## Key Benefits

- **10,000x fewer parameters**: For GPT-3 175B, LoRA reduces trainable parameters
  from 175 billion to ~18 million.
- **3x less GPU memory**: No need to store optimizer states for the full model.
- **No inference latency**: The low-rank matrices can be merged with frozen weights.
- **Switchable**: Multiple LoRA adapters can be swapped at inference time.

## Hyperparameters

- Rank (r): Typically 4-64. r=8 is a good default.
- Alpha: Scaling factor applied as alpha/r.
- Target modules: Usually attention projections (q_proj, v_proj).
