---
title: GPT-3 Few-Shot Learning
authors: Brown et al.
year: 2020
tags: gpt-3, few-shot, scaling
---

# GPT-3: Language Models are Few-Shot Learners

GPT-3 demonstrates that scaling up language models greatly improves few-shot
performance. The 175B parameter model can perform tasks by conditioning on
examples in the prompt, without any gradient updates.

## In-Context Learning

- Zero-shot: Only a task description.
- One-shot: One example alongside the description.
- Few-shot: 10-100 examples in the prompt.

Performance improves log-linearly with model scale across all settings.
