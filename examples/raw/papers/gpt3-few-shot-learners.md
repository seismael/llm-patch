---
title: "Language Models are Few-Shot Learners"
authors: Brown et al.
year: 2020
arxiv: "2005.14165"
tags: gpt-3, few-shot, in-context-learning, scaling
---

# Language Models are Few-Shot Learners (GPT-3)

## Abstract

Recent work has demonstrated substantial gains on many NLP tasks and
benchmarks by pre-training on a large corpus of text followed by fine-tuning
on a specific task. While typically task-agnostic in architecture, this method
still requires task-specific fine-tuning datasets. We show that scaling up
language models greatly improves task-agnostic, few-shot performance,
sometimes even reaching competitiveness with prior state-of-the-art
fine-tuning approaches.

## Key Contributions

- **Scaling Laws**: GPT-3 demonstrates that language model performance
  scales predictably with model size, dataset size, and compute budget.
  The 175B parameter model shows dramatic gains over the 1.5B GPT-2.

- **In-Context Learning**: GPT-3 can perform tasks by conditioning on a
  few examples provided in the prompt (few-shot), a single example
  (one-shot), or just a task description (zero-shot), without any gradient
  updates or fine-tuning.

- **Broad Task Coverage**: The model achieves strong performance across
  translation, question answering, cloze tasks, and on-the-fly reasoning
  tasks, demonstrating general linguistic competence.

## Architecture

GPT-3 uses the same Transformer decoder architecture as GPT-2 with the
following scale:

| Model | Parameters | Layers | $d_{\text{model}}$ | Heads |
|-------|-----------|--------|---------------------|-------|
| GPT-3 Small | 125M | 12 | 768 | 12 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 |
| GPT-3 Large | 760M | 24 | 1536 | 16 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 |
| GPT-3 175B | 175B | 96 | 12288 | 96 |

The model uses alternating dense and locally banded sparse attention
patterns in the layers of the transformer, following the Sparse Transformer.

## Training Data

GPT-3 was trained on a filtered version of Common Crawl (410B tokens), an
expanded version of WebText (19B tokens), two internet-based books corpora
(Books1 at 12B tokens, Books2 at 55B tokens), and English-language Wikipedia
(3B tokens), totaling roughly 499B tokens with weighted sampling.

## Few-Shot Learning

The key finding is that model scale enables in-context learning without
parameter updates:

- **Zero-shot**: Model receives only a natural language description of the
  task. Example: "Translate English to French: cheese =>"
- **One-shot**: Model receives one example of the task alongside the
  description.
- **Few-shot**: Model receives a small number (typically 10-100) of examples
  in the prompt.

Performance improves log-linearly with model scale across all three settings,
with few-shot often approaching fine-tuned baselines.

## Limitations

- Text generation can produce plausible-sounding but factually incorrect
  content.
- The model has difficulty with tasks requiring multi-step reasoning.
- Training data biases are reflected in model outputs.
- The model has a fixed context window limiting in-context examples.

## Impact

GPT-3 established the paradigm of foundation models — large pre-trained
language models that can be adapted to downstream tasks through prompting
rather than fine-tuning. This motivated subsequent work on instruction
tuning, RLHF, and parameter-efficient methods like LoRA that enable
cost-effective adaptation of these massive models.
