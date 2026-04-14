---
title: "LoRA: Low-Rank Adaptation of Large Language Models"
authors: Hu et al.
year: 2021
arxiv: "2106.09685"
tags: lora, fine-tuning, parameter-efficient, adaptation
---

# LoRA: Low-Rank Adaptation of Large Language Models

## Abstract

An important paradigm of natural language processing consists of large-scale
pre-training on general domain data and adaptation to particular tasks or
domains. As we pre-train larger models, full fine-tuning becomes less
feasible. LoRA freezes the pre-trained model weights and injects trainable
rank decomposition matrices into each layer of the Transformer architecture,
greatly reducing the number of trainable parameters for downstream tasks.

## Key Contributions

- **Low-Rank Decomposition**: LoRA decomposes the weight update matrix
  $\Delta W$ into two low-rank matrices: $\Delta W = BA$ where
  $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$
  with rank $r \ll \min(d, k)$.

- **No Additional Inference Latency**: The trained low-rank matrices can
  be merged with the frozen weights during inference, introducing zero
  additional latency compared to a fully fine-tuned model.

- **Drastically Reduced Parameters**: For GPT-3 175B, LoRA reduces the
  number of trainable parameters by 10,000x and the GPU memory requirement
  by 3x compared to full fine-tuning.

## Method

For a pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA
constrains the update by representing it with a low-rank decomposition:

$$W_0 + \Delta W = W_0 + BA$$

During training, $W_0$ is frozen and does not receive gradient updates, while
$A$ and $B$ contain trainable parameters. Both $W_0$ and $\Delta W = BA$ are
multiplied with the same input. For $h = W_0 x$, the modified forward pass
yields:

$$h = W_0 x + \Delta W x = W_0 x + BAx$$

Matrix $A$ is initialized with a random Gaussian, and $B$ is initialized to
zero, so $\Delta W = BA$ is zero at the beginning of training.

## Key Hyperparameters

- **Rank ($r$)**: Controls the expressiveness of the adaptation. Typical
  values range from 4 to 64. Lower ranks use fewer parameters; $r = 8$
  often provides a good trade-off.

- **Alpha ($\alpha$)**: A scaling factor applied as $\alpha / r$ to the
  low-rank update, controlling the magnitude of adaptation.

- **Target Modules**: LoRA is typically applied to attention projection
  matrices ($W_q$, $W_v$) but can also target feed-forward layers.

## Results

LoRA matches or exceeds fine-tuning quality on RoBERTa, DeBERTa, GPT-2,
and GPT-3, despite having far fewer trainable parameters. On GPT-3 175B,
LoRA achieves comparable or better performance than full fine-tuning on
multiple benchmarks while training with 10,000x fewer parameters.

## Impact

LoRA became the de facto standard for parameter-efficient fine-tuning of
large language models. The PEFT library (Hugging Face) implements LoRA
alongside QLoRA and other variants, enabling adaptation of billion-parameter
models on consumer hardware. This directly enables the llm-patch approach
of using hypernetworks to generate LoRA weights from text.
