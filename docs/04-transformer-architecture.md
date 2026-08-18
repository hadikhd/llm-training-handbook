---
id: Transformer Architecture
title: Transformer Architecture
sidebar_label: Transformer Architecture
sidebar_position: 5
description: The Transformer components and architectural trade-offs behind modern decoder-only LLMs.
---
<div className="chapter-hero">

![Chapter 4 — Transformer Architecture](/static/img/chapters/transformer-architecture.png)

</div>

[Previous: Tokenization](./03-tokenization.md) |
[Contents](./index.md) |
[Next: Pretraining](./05-pretraining.md)

---



## Learning Objectives

By the end of this chapter, you should be able to:

- Explain why the Transformer replaced recurrent architectures for large-scale language modeling.
- Describe the main components of a Transformer block.
- Understand token embeddings, positional information, self-attention, feed-forward networks, normalization, and residual connections.
- Derive the core equations of scaled dot-product attention.
- Explain multi-head attention and why multiple attention heads are useful.
- Distinguish between encoder-only, decoder-only, and encoder-decoder Transformer architectures.
- Explain why modern LLMs usually use decoder-only causal Transformers.
- Understand causal masking and how it enables autoregressive generation.
- Describe common architectural variants used in modern LLMs, including Pre-LN, RMSNorm, SwiGLU, RoPE, grouped-query attention, and multi-query attention.
- Analyze how Transformer architecture affects compute, memory, context length, and inference efficiency.
- Identify common implementation pitfalls in Transformer models.

---

## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**

## 4.1 Why Transformers Matter

The Transformer is the dominant architecture behind modern large language models.

Before Transformers, many sequence models used recurrent neural networks such as RNNs, LSTMs, or GRUs. These models processed text token by token in sequence. This made long-range dependencies difficult to learn and limited parallelism during training.

The Transformer changed this by using **attention** as the main mechanism for mixing information across tokens, while allowing the sequence positions to be processed in parallel during training.

Instead of processing tokens strictly one at a time, a Transformer can process all tokens in a sequence in parallel during training, while allowing each token representation to attend to other tokens.

This made it possible to scale language models to:

- Larger datasets
- Longer sequences
- More parameters
- More parallel training hardware
- Better long-context reasoning
- Better transfer learning
- More general-purpose behavior

The original Transformer was introduced in:

> Vaswani et al., *Attention Is All You Need*, 2017.

Modern LLMs are not identical to the original Transformer, but they are direct descendants of it.

---


## 4.2 The High-Level Transformer View

A Transformer language model takes a sequence of token IDs and produces a probability distribution over the next token.

A simplified decoder-only Transformer looks like this:
```text
Token IDs
   |
   v
Token Embeddings + Positional Information
   |
   v
Transformer Block 1
   |
   v
Transformer Block 2
   |
   v
...
   |
   v
Transformer Block N
   |
   v
Final Normalization
   |
   v
Output Projection
   |
   v
Logits over Vocabulary
   |
   v
Softmax
   |
   v
Next-token probabilities
```
Each Transformer block typically contains:

```text
Input
  |
  +--> Normalization
  |       |
  |       v
  |   Self-Attention
  |       |
  +<------+
  |
  +--> Normalization
  |       |
  |       v
  |   Feed-Forward Network
  |       |
  +<------+
  |
Output
```

The two core sublayers are:

- **Self-attention**
- **Feed-forward network**, also called MLP or FFN

These are connected by:

- Residual connections
- Normalization layers
- Positional information
- Dropout in some training setups

---


## 4.3 From Tokens to Vectors

As discussed in Chapter 3, text is first converted into token IDs.

Given a sequence:


$$
x_1, x_2, \ldots, x_T
$$


where each $x_t$ is a token ID, the model maps each token ID into a dense vector using an embedding matrix:


$$
E \in \mathbb{R}^{|V| \times d_{\text{model}}}
$$


The embedding for token $x_t$ is:


$$
h_t^{(0)} = E[x_t]
$$


where:

- $|V|$ is the vocabulary size.
- $d_{\text{model}}$ is the hidden dimension.
- $h_t^{(0)}$ is the initial representation of token $t$.

For the full sequence:


$$
H^{(0)} \in \mathbb{R}^{T \times d_{\text{model}}}
$$


where:

- $T$ is sequence length.
- $d_{\text{model}}$ is model width.

The Transformer repeatedly updates this matrix through multiple layers.

---


## 4.4 Why Position Matters

Self-attention by itself is permutation-equivariant: without positional information, it does not distinguish token order.

If we give the model these two sequences:

```text
The dog chased the cat.
The cat chased the dog.
```
the meaning is different. But without positional information, attention sees a set of token vectors rather than an ordered sequence.

Therefore, Transformers need a way to represent position.

Common approaches include:

- Learned absolute positional embeddings
- Sinusoidal positional embeddings
- Relative positional bias
- Rotary positional embeddings, or RoPE
- ALiBi
- Other long-context positional methods

Modern decoder-only LLMs often use RoPE or related variants.

---


## 4.5 Absolute Positional Embeddings

The original Transformer used sinusoidal positional encodings. Some later models used learned absolute positional embeddings.

With learned absolute positions, the model has a position embedding matrix:


$$
P \in \mathbb{R}^{T_{\max} \times d_{\text{model}}}
$$


The input representation becomes:


$$
h_t^{(0)} = E[x_t] + P[t]
$$


where $P[t]$ is the position embedding for position $t$.


### Advantages

- Simple to implement.
- Works well within the trained context length.
- Easy to understand.


### Disadvantages

- Does not naturally extrapolate to longer contexts.
- Requires a fixed maximum position table.
- Position IDs must be handled carefully during packing and generation.

Many older Transformer models used absolute positional embeddings, but modern LLMs often use alternatives.

---


## 4.6 Sinusoidal Positional Encoding

The original Transformer used fixed sinusoidal functions:


$$
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$


$$
PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$


where:

- $pos$ is the token position.
- $i$ indexes the embedding dimension.

Sinusoidal encodings allow the model to represent relative distances through combinations of periodic functions.

They are no longer the default choice for many modern LLMs, but they remain important historically.

---


## 4.7 Rotary Positional Embeddings

Rotary Positional Embeddings, or RoPE, inject position information by rotating query and key vectors in attention.

Instead of adding a position vector to token embeddings, RoPE applies a position-dependent rotation to the query and key vectors.

At a high level:

```text
Token hidden state
   |
   v
Query and key projections
   |
   v
Apply position-dependent rotation
   |
   v
Attention score computation

RoPE is popular because it:

- Encodes relative position information naturally.
- Works well in decoder-only LLMs.
- Is compatible with efficient attention implementations.
- Supports long-context extension techniques.
- Avoids a learned absolute position table.
```

Many modern open-source LLM families use RoPE.

---


## 4.8 The Transformer Block

A standard decoder-only Transformer block contains:

```text
Input hidden states
|
v
Normalization
|
v
Causal self-attention
|
v
Residual addition
|
v
Normalization
|
v
Feed-forward network
|
v
Residual addition
|
v
Output hidden states
```

In equation form, a common Pre-LN block is:


$$
\tilde{H}^{(l)} = H^{(l)} + \text{Attention}(\text{Norm}(H^{(l)}))
$$


$$
H^{(l+1)} = \tilde{H}^{(l)} + \text{FFN}(\text{Norm}(\tilde{H}^{(l)}))
$$


where:

- $H^{(l)}$ is the input to layer $l$.
- $H^{(l+1)}$ is the output of layer $l$.
- $\text{Norm}$ is usually LayerNorm or RMSNorm.
- $\text{FFN}$ is the feed-forward network.

---


## 4.9 Self-Attention Intuition

Self-attention lets each token look at other tokens in the same sequence.

Consider:

```text
The animal did not cross the street because it was tired.
```

The word `it` may need to refer to `animal`.

Self-attention allows the representation of `it` to use information from earlier tokens.

In a causal language model, token $t$ may attend only to tokens at positions:


$$
1, 2, \ldots, t
$$


It cannot attend to future tokens during training or generation.

This restriction preserves the autoregressive objective.

---


## 4.10 Queries, Keys, and Values

Self-attention uses three learned projections:

- Query
- Key
- Value

Given hidden states:


$$
H \in \mathbb{R}^{T \times d_{\text{model}}}
$$


the model computes:


$$
Q = HW_Q
$$


$$
K = HW_K
$$


$$
V = HW_V
$$


where:


$$
W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}
$$


Each token produces:

- A query vector: what this token is looking for.
- A key vector: what this token offers for matching.
- A value vector: the information this token contributes if attended to.

Attention scores are computed by comparing queries with keys.

---


## 4.11 Scaled Dot-Product Attention

The core attention operation is:


$$
\text{Attention}(Q, K, V) = \text{softmax} \left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$


where:

- $Q \in \mathbb{R}^{T \times d_k}$
- $K \in \mathbb{R}^{T \times d_k}$
- $V \in \mathbb{R}^{T \times d_v}$
- $d_k$ is the key/query dimension

The matrix:


$$
QK^\top \in \mathbb{R}^{T \times T}
$$


contains pairwise attention scores between tokens.

The scale factor:


$$
\frac{1}{\sqrt{d_k}}
$$


prevents dot products from becoming too large as the dimension grows.

Without scaling, softmax can saturate, causing poor gradients.

---


## 4.12 Attention Matrix

For a sequence of length $T$, attention produces a $T \times T$ score matrix.

Example for $T = 4$:

```text
key1   key2   key3   key4
query1     .      .      .      .
query2     .      .      .      .
query3     .      .      .      .
query4     .      .      .      .

Each row corresponds to one query token. Each column corresponds to a key token.
```

After softmax, each row becomes a probability distribution over tokens.

The output for token $i$ is a weighted sum of value vectors:


$$
o_i = \sum_{j=1}^{T} a_{ij} v_j
$$


where:

- $a_{ij}$ is the attention weight from token $i$ to token $j$.
- $v_j$ is the value vector for token $j$.

---


## 4.13 Causal Masking

In autoregressive language modeling, token $t$ must not attend to future tokens.

The model predicts:


$$
P(x_t \mid x_{1:t-1})
$$


Therefore, when computing attention for position $t$, positions greater than $t$ must be masked.

For a sequence of length 4, the causal mask allows:

```text
key1   key2   key3   key4
query1     ✓      x      x      x
query2     ✓      ✓      x      x
query3     ✓      ✓      ✓      x
query4     ✓      ✓      ✓      ✓
```

The masked attention scores are set to a very negative value before softmax:


$$
-\infty
$$


so their softmax probability becomes approximately zero.

This prevents information leakage from future tokens.

---


## 4.14 Why Causal Masking Matters

Suppose the training sequence is:

```text
The capital of France is Paris.
```

When predicting `Paris`, the model can attend to:

```text
The capital of France is

But it must not attend to `Paris` itself or any later token.
```

If causal masking is implemented incorrectly, the model may see future tokens during training. This creates artificially low training loss but fails during generation.

This is one of the most serious implementation bugs in language model training.

---


## 4.15 Multi-Head Attention

Instead of performing attention once, Transformers use multiple attention heads.

Each head has its own query, key, and value projections.

For head $h$:


$$
Q_h = HW_Q^{(h)}
$$


$$
K_h = HW_K^{(h)}
$$


$$
V_h = HW_V^{(h)}
$$


$$
O_h = \text{Attention}(Q_h, K_h, V_h)
$$


The outputs from all heads are concatenated:


$$
O = \text{Concat}(O_1, O_2, \ldots, O_n)
$$


Then projected back to the model dimension:


$$
\text{MHA}(H) = OW_O
$$


where:


$$
W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$


Multi-head attention allows the model to attend to different relationships in parallel.

Different heads may specialize in patterns such as:

- Nearby tokens
- Long-range dependencies
- Syntax
- Repeated names
- Lists
- Code indentation
- Punctuation
- Document structure
- Retrieval-like copying behavior

---


## 4.16 Attention Head Dimensions

Usually:


$$
d_{\text{model}} = n_{\text{heads}} \times d_{\text{head}}
$$


For example:

```text
d_model = 4096
n_heads = 32
d_head = 128
```

The attention projections commonly map from:


$$
d_{\text{model}} \rightarrow n_{\text{heads}} \times d_{\text{head}}
$$


The choice of head dimension affects:

- Attention quality
- Memory layout
- Kernel efficiency
- KV-cache size
- RoPE behavior
- Hardware utilization

Many modern LLMs use head dimensions such as 64 or 128.

---


## 4.17 Feed-Forward Network

Each Transformer block also contains a position-wise feed-forward network.

The FFN is applied independently to each token position.

A classic FFN is:


$$
\text{FFN}(x) = W_2 \sigma(W_1 x + b_1) + b_2
$$


where:

- $W_1$ projects from $d_{\text{model}}$ to $d_{\text{ff}}$.
- $W_2$ projects from $d_{\text{ff}}$ back to $d_{\text{model}}$.
- $\sigma$ is a nonlinearity such as ReLU, GELU, or SiLU.

Usually:


$$
d_{\text{ff}} \approx 4 \times d_{\text{model}}
$$


though modern gated FFNs often use different ratios.

The FFN is responsible for much of the model’s parameter count and compute.

---


## 4.18 Activation Functions

Early Transformers used ReLU. Later models often use GELU.

Modern LLMs commonly use gated activations such as SwiGLU.


### ReLU


$$
\text{ReLU}(x) = \max(0, x)
$$


### GELU

GELU is smoother than ReLU and was used in models such as BERT and GPT-2.


### SiLU


$$
\text{SiLU}(x) = x \cdot \sigma(x)
$$


where $\sigma$ is the sigmoid function.


### SwiGLU

SwiGLU is a gated activation:


$$
\text{SwiGLU}(x) = \text{SiLU}(xW_g) \odot (xW_u)
$$


followed by a down projection.

A simplified form is:


$$
\text{FFN}(x) = (\text{SiLU}(xW_g) \odot xW_u)W_d
$$


where:

- $W_g$ is the gate projection.
- $W_u$ is the up projection.
- $W_d$ is the down projection.
- $\odot$ is elementwise multiplication.

SwiGLU is widely used in modern decoder-only LLMs.

---


## 4.19 Residual Connections

Residual connections add the input of a sublayer to its output:


$$
y = x + f(x)
$$


In a Transformer block:

```text
x -> attention -> add back to x
x -> FFN       -> add back to x

Residual connections help:

- Improve gradient flow
- Stabilize deep networks
- Preserve information across layers
- Make optimization easier
```

Without residual connections, very deep Transformers would be much harder to train.

---


## 4.20 Normalization Layers

Normalization stabilizes training.

The original Transformer used LayerNorm after sublayers. Modern LLMs often use Pre-LN or RMSNorm.


### LayerNorm

LayerNorm normalizes across hidden dimensions for each token:


$$
\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$


where:

- $\mu$ is the mean of hidden dimensions.
- $\sigma^2$ is the variance.
- $\gamma$ and $\beta$ are learned parameters.


### RMSNorm

RMSNorm removes mean subtraction and normalizes by root mean square:


$$
\text{RMSNorm}(x) = \gamma \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}}
$$


RMSNorm is simpler and often faster than LayerNorm.

Many modern LLMs use RMSNorm.

---


## 4.21 Pre-LN vs Post-LN

There are two common normalization placements.


### Post-LN

Used in the original Transformer:


$$
H' = \text{Norm}(H + \text{Sublayer}(H))
$$


### Pre-LN

Common in modern LLMs:


$$
H' = H + \text{Sublayer}(\text{Norm}(H))
$$


Pre-LN improves training stability for deep models.


### Practical Difference

Post-LN can work, but deep Post-LN Transformers are often harder to train without careful initialization or learning-rate schedules.

Pre-LN is usually preferred for large decoder-only LLMs.

---


## 4.22 Decoder-Only Transformer

Most modern generative LLMs use a decoder-only Transformer.

Examples include GPT-style models and many open-source LLM families.

A decoder-only model uses:

- Token embeddings
- Positional information
- Stacked causal self-attention blocks
- Feed-forward networks
- Final normalization
- Output projection to vocabulary logits

The key property is causal attention:

```text
Each token can attend only to itself and previous tokens.
```

This directly matches next-token prediction.

---


## 4.23 Encoder-Only Transformer

Encoder-only Transformers, such as BERT-style models, use bidirectional attention.

Each token can attend to all tokens:

```text
key1   key2   key3   key4
query1     ✓      ✓      ✓      ✓
query2     ✓      ✓      ✓      ✓
query3     ✓      ✓      ✓      ✓
query4     ✓      ✓      ✓      ✓

Encoder-only models are useful for:

- Classification
- Embeddings
- Retrieval
- Reranking
- Named entity recognition
- Token classification
- Semantic similarity
```

They are usually trained with objectives such as masked language modeling.

They are not the standard architecture for autoregressive text generation.

---


## 4.24 Encoder-Decoder Transformer

Encoder-decoder Transformers were used in the original Transformer and are common in sequence-to-sequence models such as T5.

They contain:

- An encoder that reads the input sequence bidirectionally.
- A decoder that generates output autoregressively.
- Cross-attention from decoder tokens to encoder outputs.

This architecture is useful for:

- Translation
- Summarization
- Text-to-text tasks
- Conditional generation

The decoder has two attention mechanisms:

- Causal self-attention over generated tokens
- Cross-attention over encoder states

Many modern general-purpose chat LLMs use decoder-only architectures instead, because decoder-only models scale simply and work well for broad generative tasks.

---


## 4.25 Why Decoder-Only Models Dominate LLMs

Decoder-only Transformers are popular for LLMs because they are:

- Simple
- Scalable
- Compatible with next-token prediction
- Easy to train on raw text
- Easy to adapt to chat formatting
- Efficient for autoregressive generation
- Flexible across tasks through prompting

The training format is straightforward:

```text
Document tokens:
x1 x2 x3 ... xT

Training:
predict x2 from x1
predict x3 from x1 x2
predict x4 from x1 x2 x3
...
```

The same objective works for:

- Web documents
- Books
- Code
- Conversations
- Tool traces
- Mathematical text
- Instruction data

This simplicity is one reason decoder-only Transformers became the default for large generative models.

---


## 4.26 Output Projection and Logits

After the final Transformer layer, each position has a hidden state:


$$
h_t \in \mathbb{R}^{d_{\text{model}}}
$$


To predict the next token, the model projects this vector to vocabulary size:


$$
z_t = h_t W_{\text{out}} + b
$$


where:


$$
W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}
$$


The output $z_t$ is called the logits vector:


$$
z_t \in \mathbb{R}^{|V|}
$$


The probability distribution is:


$$
P(x_{t+1} \mid x_{\le t}) = \text{softmax}(z_t)
$$


The training loss is cross-entropy against the true next token.

---


## 4.27 Weight Tying

Some language models share the input embedding matrix and output projection matrix.

This is called weight tying.

Input embedding:


$$
E \in \mathbb{R}^{|V| \times d_{\text{model}}}
$$


Output projection can use:


$$
W_{\text{out}} = E^\top
$$


Weight tying reduces parameters and can improve efficiency.

However, not all architectures tie weights. The decision depends on model design, training setup, and implementation.

---


## 4.28 Attention Compute Complexity

The attention score matrix has shape:


$$
T \times T
$$


For each layer and head, attention requires comparing every token with every allowed token.

The approximate attention complexity is:


$$
O(T^2 d)
$$


where:

- $T$ is sequence length.
- $d$ is hidden size or head dimension depending on context.

This quadratic dependence on sequence length is one of the main limitations of Transformers.

If sequence length doubles, attention score computation and memory can grow roughly four times.

This is why long-context training and inference require careful engineering.

---


## 4.29 FFN Compute Complexity

The feed-forward network is applied independently at each token position.

Its approximate complexity is:


$$
O(T d_{\text{model}} d_{\text{ff}})
$$


Since $d_{\text{ff}}$ is often several times larger than $d_{\text{model}}$, the FFN can dominate compute, especially at moderate sequence lengths.

In many LLMs, a large fraction of parameters are in the FFN layers.

This is why FFN design matters for both model quality and training efficiency.

---


## 4.30 Parameter Count of a Transformer Block

A simplified decoder-only Transformer block contains:

- Query projection
- Key projection
- Value projection
- Output attention projection
- FFN up projection
- FFN down projection
- Normalization parameters

Ignoring biases and normalization, standard multi-head attention has approximately:


$$
4 d_{\text{model}}^2
$$


parameters:

- $W_Q$
- $W_K$
- $W_V$
- $W_O$

A standard FFN has approximately:


$$
2 d_{\text{model}} d_{\text{ff}}
$$


parameters.

If:


$$
d_{\text{ff}} = 4d_{\text{model}}
$$


then FFN parameters are:


$$
8d_{\text{model}}^2
$$


So a classic Transformer block has roughly:


$$
12d_{\text{model}}^2
$$


parameters, excluding embeddings and small terms.

For gated FFNs such as SwiGLU, parameter counting differs because there are usually three FFN matrices (gate, up, and down projections).

---


## 4.31 KV Cache

During autoregressive inference, the model generates one token at a time.

For each generated token, each layer computes key and value vectors.

Instead of recomputing keys and values for all previous tokens at every step, inference systems store them in a **KV cache**.

At generation step $t$, the model stores:

```text
Keys:   K_1, K_2, ..., K_t
Values: V_1, V_2, ..., V_t
```

For the next token, the model computes only the new query, key, and value, then attends to cached previous keys and values.

This makes decoding much faster.

---


## 4.32 KV Cache Memory Cost

KV cache memory grows with:

- Batch size
- Sequence length
- Number of layers
- Number of KV heads
- Head dimension
- Precision

A simplified formula is:


$$
\text{KV cache elements} = 2 \times B \times T \times L \times n_{\text{kv heads}} \times d_{\text{head}}
$$


where:

- $2$ accounts for keys and values.
- $B$ is batch size.
- $T$ is sequence length.
- $L$ is number of layers.
- $n_{\text{kv heads}}$ is number of key-value heads.
- $d_{\text{head}}$ is head dimension.

Memory in bytes:


$$
\text{bytes} = \text{elements} \times \text{bytes per element}
$$


For long-context serving, KV-cache memory is often a major bottleneck because it grows linearly with sequence length, batch size, layer count, KV-head count, and head dimension.

---


## 4.33 Multi-Query Attention

Multi-Query Attention (MQA) uses many query heads but shares a single key head and a single value head across those query heads.

Standard multi-head attention:

```text
Q heads: many
K heads: many
V heads: many
```
MQA:

```text
Q heads: many
K heads: one
V heads: one
```

This greatly reduces KV cache size during inference.


### Advantages

- Lower KV-cache memory
- Faster decoding
- Better serving efficiency


### Disadvantages

- May reduce model quality if too aggressive
- Less expressive than full multi-head KV representations

MQA is mainly an inference-efficiency optimization.

---


## 4.34 Grouped-Query Attention

Grouped-Query Attention, or GQA, is a compromise between standard multi-head attention and MQA.

It uses many query heads but fewer key-value heads.

Example:

```text
Query heads: 32
KV heads: 8

Each group of query heads shares one key-value head.

GQA reduces KV cache memory while preserving more capacity than MQA.
```

Many modern LLMs use GQA because it provides a good balance between quality and inference efficiency.

---


## 4.35 Sliding Window Attention

Some models use sliding window attention to reduce attention cost for long sequences.

Instead of allowing each token to attend to all previous tokens, each token attends only to a recent window.

Example:

```text
Window size = 4

Token 10 attends to tokens:
7, 8, 9, 10
```

This reduces compute from full quadratic attention over the entire sequence.


### Advantages

- Lower memory use
- Better long-context scalability
- Faster training and inference for long sequences


### Disadvantages

- Limits direct access to distant tokens
- Requires architectural or training adjustments
- May hurt tasks requiring long-range dependencies

Some architectures combine local attention with occasional global attention or other mechanisms.

---


## 4.36 Attention Implementation Details

Naive attention materializes the full attention matrix:


$$
T \times T
$$


This can be memory-expensive.

Modern implementations use optimized attention kernels such as FlashAttention.

These kernels reduce memory overhead by computing attention in blocks and avoiding full materialization of the attention matrix.

Benefits include:

- Lower memory use
- Faster training
- Better GPU utilization
- Support for longer sequences

FlashAttention-style kernels compute the same attention operation as standard scaled dot-product attention, but use IO-aware tiling to reduce high-cost memory traffic and avoid materializing the full attention matrix.

---


## 4.37 Dropout

Dropout randomly zeroes activations during training.

The original Transformer used dropout in several places:

- Attention weights
- Residual connections
- Feed-forward activations
- Embeddings

Many modern large-scale LLMs use little or no dropout during pretraining, especially when trained on very large datasets.

However, dropout can still be useful for:

- Smaller models
- Smaller datasets
- Fine-tuning
- Preventing overfitting

Dropout is disabled during inference.

---


## 4.38 Initialization

Weight initialization affects training stability.

A poor initialization can cause:

- Exploding activations
- Vanishing gradients
- Slow convergence
- Instability in deep networks

Large Transformers often use carefully chosen initialization schemes, sometimes with depth scaling.

The exact scheme depends on:

- Normalization placement
- Activation function
- Model depth
- Residual path design
- Optimizer
- Precision format

Initialization is easy to overlook, but it matters significantly at scale.

---


## 4.39 Residual Stream

Modern interpretations of Transformers often describe the hidden state as a **residual stream**.

Each layer reads from the stream, computes an update, and writes back into it.

Simplified:

```text
Residual stream
   |
   +-- attention reads and writes update
   |
   +-- FFN reads and writes update
   |
   +-- next layer continues
```

This view is useful because information can flow through the network across many layers, while each block contributes transformations to the shared representation.

---


## 4.40 Depth and Width

Transformer capacity depends strongly on:

- Number of layers
- Hidden dimension
- Number of attention heads
- FFN dimension
- Vocabulary size
- Context length

Depth means more Transformer blocks.

Width means larger hidden dimension.

Increasing depth and width both increase parameter count and compute, but they affect learning differently.


### Deeper Models

Advantages:

- More sequential transformations
- Potentially better abstraction
- More compositional processing

Disadvantages:

- Harder optimization
- More latency
- More activation memory
- Greater need for stable normalization and initialization


### Wider Models

Advantages:

- More capacity per layer
- Better parallelism in some hardware regimes
- Larger representation space

Disadvantages:

- More matrix multiplication cost
- Larger FFN and attention projections
- More memory bandwidth pressure

Scaling laws help guide these choices, but architecture and hardware constraints also matter.

---


## 4.41 Context Length

Context length is the maximum number of tokens the model can process at once.

For a decoder-only LLM, context length affects:

- Maximum prompt size
- Training sequence length
- Attention memory
- KV-cache size
- Long-document ability
- RAG document packing
- Chat history length

A context length of 8192 tokens does not mean 8192 words. As discussed in Chapter 3, tokenization determines how much text fits.

Longer context is useful, but it is expensive.

---


## 4.42 Long-Context Challenges

Long-context models face several challenges:

- Quadratic attention cost
- Large KV-cache memory
- Positional extrapolation
- Data scarcity for long sequences
- Difficulty using distant information
- Evaluation complexity
- Retrieval-like failure modes
- Increased serving cost

Simply increasing the configured context window does not guarantee that the model will reliably use information from all positions.

Long-context capability depends on:

- Architecture
- Positional encoding
- Training data
- sequence-length curriculum
- attention implementation
- evaluation design

---


## 4.43 Positional Extrapolation

A model trained on 4096-token sequences may not work well at 32768 tokens.

Reasons include:

- Positional embeddings may not support unseen positions.
- Attention patterns may not generalize.
- Training data may not contain long-range dependencies.
- RoPE frequencies may behave poorly beyond the trained range.

Long-context extension methods often modify RoPE scaling or fine-tune on longer sequences.

However, context extension should be evaluated carefully. A model may accept a long prompt but fail to use far-away information reliably.

---


## 4.44 Model Configuration Example

A Transformer configuration may look like:

```json
{
  "vocab_size": 50000,
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 11008,
  "hidden_act": "silu",
  "max_position_embeddings": 8192,
  "rope_theta": 10000,
  "rms_norm_eps": 1e-6,
  "tie_word_embeddings": false
}
```

Important fields include:

- `vocab_size`: number of tokenizer tokens.
- `hidden_size`: model width.
- `num_hidden_layers`: depth.
- `num_attention_heads`: number of query heads.
- `num_key_value_heads`: number o

## Common Failure Modes

- **Unexpected memory growth:** inspect attention tensors, KV-cache assumptions, sequence length, and activation storage.
- **Poor training stability:** compare normalization placement, initialization, residual scaling, and optimization settings.
- **Long-context degradation:** evaluate attention behavior and positional representation beyond the training regime.
---
[Previous: Tokenization](./03-tokenization.md) |
[Contents](./index.md) |
[Next: Pretraining](./05-pretraining.md)
---