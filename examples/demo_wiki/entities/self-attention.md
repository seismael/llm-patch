---
title: Self-Attention
type: entity
---

# Self-Attention

Self-attention (also called intra-attention) relates different positions of a
single sequence to compute a representation. Each position attends to all positions
in the previous layer. The computation uses three projections: queries (Q),
keys (K), and values (V), all derived from the input.

The output is a weighted sum of values, where weights are determined by the
compatibility between queries and keys via scaled dot-product.
