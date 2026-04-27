---
title: Attention Is All You Need
authors: Vaswani et al.
year: 2017
tags: transformer, attention
---

# Attention Is All You Need

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks. We propose the [[Transformer]], a new architecture
based solely on [[Self-Attention]] mechanisms.

## Key Ideas

- **Self-Attention Mechanism**: Scaled dot-product attention computes compatibility
  between queries and keys, weighting values accordingly.
- **Multi-Head Attention**: Allows the model to attend to information from different
  representation sub-spaces at different positions.
- **Positional Encoding**: Added to give the model position information since there
  is no recurrence or convolution.

## Architecture Details

The Transformer uses an encoder-decoder structure. The encoder maps input symbols
to continuous representations using 6 identical layers, each with multi-head
self-attention and a feed-forward network. The decoder also has 6 layers, adding
a third sub-layer for cross-attention over the encoder output. Residual connections
and layer normalization are applied around each sub-layer.

## Attention Function

The attention function maps queries Q, keys K, and values V:
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

Multi-head attention runs h parallel attention heads and concatenates the results.

## Results

The Transformer achieves 28.4 BLEU on WMT 2014 English-to-German translation,
improving over existing best results by over 2 BLEU. On English-to-French, it
achieves 41.0 BLEU, outperforming all previously published single models.
