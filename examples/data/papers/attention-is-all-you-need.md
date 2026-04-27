---
title: Attention Is All You Need
authors: Vaswani et al.
year: 2017
arxiv: "1706.03762"
tags: transformer, attention, sequence-to-sequence
---

# Attention Is All You Need

## Abstract

The dominant sequence transduction models are based on complex recurrent or
convolutional neural networks that include an encoder and a decoder. The best
performing models also connect the encoder and decoder through an attention
mechanism. We propose a new simple network architecture, the Transformer,
based solely on attention mechanisms, dispensing with recurrence and
convolutions entirely.

## Key Contributions

- **Self-Attention Mechanism**: Scaled dot-product attention computes
  compatibility between queries and keys, weighting values accordingly.
  Multi-head attention allows the model to attend to information from
  different representation sub-spaces at different positions.

- **Positional Encoding**: Since the model contains no recurrence and no
  convolution, positional encodings are added to give the model information
  about the relative or absolute position of tokens in the sequence.

- **Encoder-Decoder Architecture**: The encoder maps an input sequence of
  symbol representations to a sequence of continuous representations. The
  decoder then generates an output sequence of symbols one element at a time.

## Architecture Details

The Transformer follows an encoder-decoder structure using stacked
self-attention and point-wise fully connected layers for both the encoder
and decoder.

**Encoder**: Each layer has two sub-layers — a multi-head self-attention
mechanism and a position-wise fully connected feed-forward network. Residual
connections and layer normalization are employed around each sub-layer.

**Decoder**: In addition to the two sub-layers in each encoder layer, the
decoder inserts a third sub-layer which performs multi-head attention over
the output of the encoder stack.

## Attention Function

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Multi-head attention concatenates $h$ parallel attention heads:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

## Results

The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German
translation task, improving over existing best results by over 2 BLEU. On
the English-to-French translation task, the model achieves 41.0 BLEU,
outperforming all previously published single models at a fraction of the
training cost.

## Impact

The Transformer architecture became the foundation for virtually all modern
large language models including BERT, GPT, T5, and their successors. The
self-attention mechanism proved to be more parallelizable and required
significantly less time to train than recurrent architectures.
