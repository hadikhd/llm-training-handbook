---
id: Pretraining
title: Pretraining
sidebar_label: Pretraining
sidebar_position: 6
description: The optimization process that turns large token streams into a pretrained language model.
---
<div className="chapter-hero">

![Chapter 5 — Pretraining](/img/chapters/pretraining.png)

</div>

[Previous: Transformer Architecture](./04-transformer-architecture.md) |
[Contents](./index.md) |
[Next: scaling-laws](./06-scaling-laws.md)
---
## Learning Objectives

By the end of this chapter, you should be able to:

- Understand next-token prediction as the core pretraining objective.
- Connect token streams, sequence packing, batch size, and tokens per update.
- Explain optimization choices such as AdamW, learning-rate schedules, clipping, and mixed precision.
- Diagnose common pretraining failures using loss, throughput, stability, and checkpoint signals.


---
## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**


## 1. What Pretraining Means

Pretraining is the stage where a language model learns general linguistic, factual, and reasoning patterns from large-scale text before it is adapted to specific tasks or aligned with user preferences.

For decoder-only LLMs, pretraining is typically self-supervised rather than supervised in the traditional labeled-data sense. The model is not trained on manually labeled examples such as:
```text
input -> class label
```
Instead, it learns from raw token sequences by predicting the next token.

```text
The capital of France is -> Paris
```

This simple self-supervised objective scales remarkably well. With enough data, model capacity, and compute, next-token prediction produces models that acquire syntax, semantics, world knowledge, translation ability, coding patterns, and basic reasoning behavior.

Pretraining is therefore the foundation of an LLM. Post-training can shape behavior, but pretraining determines much of what the model knows and what it is capable of learning later.

---

## 2. The Next-Token Prediction Objective

For decoder-only language models, the dominant pretraining objective is causal language modeling.

Given a sequence of tokens:

$$
x = [x_1, x_2, x_3, \ldots, x_T]
$$

the model is trained to predict each token from the tokens before it:


$$
P(x_t \mid x_1, x_2, \ldots, x_{t-1})
$$


The training target is simply the same sequence shifted by one position.

```text
input:  [The, cat, sat, on, the]
target: [cat, sat, on, the, mat]
```

This is why causal masking is essential. At position `t`, the model must not see tokens after `t`.

Without causal masking, the model could cheat by directly attending to future tokens, producing a low training loss but failing as an autoregressive generator.

---

## 3. From Documents to Token Streams

Raw training data usually begins as documents: books, articles, code files, web pages, papers, conversations, or domain-specific corpora.

Before pretraining, these documents are transformed into token streams:

```text
documents -> cleaning -> filtering -> deduplication -> tokenization -> packed sequences
```

The model does not train on documents as variable-length objects. The training system presents token sequences, typically arranged into fixed-length windows such as:

```text
sequence length = 2048, 4096, 8192, ...
```

A key engineering question is how document boundaries are handled. Some pipelines insert an end-of-document token:

```text
<eos>
```

This teaches the model where one document ends and another begins. Without careful boundary handling, unrelated documents may appear artificially connected.

---

## 4. Sequence Packing

Pretraining is expensive, so unused tokens are wasteful.

If every short document is padded to the full context length, a large fraction of compute may be spent predicting padding tokens. Sequence packing solves this by concatenating multiple documents into one training sequence.

Example:

```text
doc1 <eos> doc2 <eos> doc3 <eos>

Then the packed stream is split into fixed-length chunks.
```

This improves hardware utilization, but it must be done carefully. If document boundaries are not represented, the model may learn unnatural transitions between unrelated texts.

For chat, instruction, or code data, boundaries are even more important because the structure of the sample carries meaning.

---

## 5. Loss Function: Cross-Entropy

The standard pretraining loss is token-level cross-entropy.

At each position, the model outputs a probability distribution over the vocabulary:

```text
vocab_size = 32k, 50k, 100k, ...
```

If the correct next token is `y`, the loss is:


$$
\mathcal{L}_t = -\log P(y_t \mid x_{<t})
$$


The final loss is averaged over many tokens.

Lower loss means the model assigns higher probability to the correct next tokens. However, low loss does not automatically mean the model is helpful, truthful, safe, or aligned. It only means it predicts the training distribution well.

---

## 6. Perplexity

Perplexity is a common metric derived from cross-entropy:


$$
\mathrm{PPL} = \exp(\mathcal{L})
$$

Intuitively, perplexity measures how uncertain the model is when predicting the next token.


A perplexity of 20 corresponds to an average exponential loss of 20; it is useful as an intuition for predictive uncertainty, but it should not be interpreted literally as the model choosing uniformly among 20 tokens.

Perplexity is useful for comparing models on the same tokenizer and evaluation corpus. It is less reliable when comparing models with different tokenizers, different data mixtures, or different evaluation preprocessing.

---

## 7. Batch Size, Sequence Length, and Tokens per Update

In LLM training, the true unit of scale is usually tokens, not examples.

Ignoring padding and masked tokens, the nominal number of tokens presented per optimizer update is:


$$
N_{\text{tokens/update}} = B_{\text{global}} \times T
$$


For example:

```text
global_batch_size = 1024
sequence_length = 4096
```

$$
N_{\text{tokens/update}} = 1024 \times 4096 = 4{,}194{,}304
$$


Large token batches can stabilize training, but they also require more memory and more distributed coordination.

Important quantities include:

- `micro_batch_size`: number of sequences processed per device before gradient accumulation
- `gradient_accumulation_steps`: number of micro-batches accumulated before an optimizer step
- `global_batch_size`: total batch size across all devices
- `sequence_length`: number of tokens per sequence
- `tokens_per_update`: total training tokens per optimizer step

A practical training run is often constrained by GPU memory, interconnect bandwidth, and checkpointing strategy.

---

## 8. Optimizer: AdamW

AdamW is a widely used optimizer for LLM pretraining.

Adam estimates first and second moments of gradients:

```text
m_t: running average of gradients
v_t: running average of squared gradients

AdamW modifies Adam by decoupling weight decay from the gradient update. This usually improves generalization and makes regularization easier to tune.
```

Common hyperparameters include:

```text
learning_rate
betas
weight_decay
epsilon
```

A typical configuration may look like:

```text
optimizer: AdamW
betas: (0.9, 0.95)
weight_decay: 0.1
epsilon: 1e-8

Exact values depend on model size, data quality, batch size, and training duration.
```

---

## 9. Learning-Rate Schedule

The learning rate is one of the most important training controls.

LLM pretraining usually uses:

```text
warmup -> decay

During warmup, the learning rate gradually increases from a small value to the peak value. This reduces instability early in training, when weights are still poorly calibrated.
```

After warmup, the learning rate decays. Common schedules include:

- cosine decay
- linear decay
- constant with cooldown
- inverse square-root decay

A simple pattern is:

```text
small LR -> peak LR -> slowly decreasing LR

Too high a learning rate can cause divergence. Too low a learning rate wastes compute and may undertrain the model.
```

---

## 10. Gradient Clipping

Gradient clipping prevents unusually large gradients from destabilizing training.

A common strategy is global norm clipping:


$$
g \leftarrow
g \cdot
\frac{\tau}{\lVert g\rVert}
\qquad
\text{if } \lVert g\rVert > \tau
$$


Typical clipping values are often around:

```text
1.0
```

Gradient clipping is not a substitute for a good learning rate, clean data, or stable architecture. It is a safety mechanism that helps prevent rare spikes from damaging the run.

---

## 11. Mixed Precision Training

LLMs are rarely trained in full FP32 precision because it is too expensive.

Common precision formats include:

- FP16
- BF16
- FP32 for selected optimizer states or sensitive operations

BF16 is widely used because it has the same exponent width as FP32 and therefore a much larger dynamic range than FP16, which generally improves numerical robustness for large-scale training.

Mixed precision reduces memory usage and improves throughput, but numerical stability must be monitored carefully.

Typical symptoms of precision issues include:

- sudden loss spikes
- NaNs
- exploding gradients
- unstable attention scores
- optimizer state corruption

---

## 12. Distributed Pretraining

Large LLMs cannot usually be trained on a single GPU. Distributed training splits computation and memory across many devices.

Common forms of parallelism include:

| Method | Purpose |
|---|---|
| Data Parallelism | Replicate model, split batches |
| Tensor Parallelism | Split large tensors across devices |
| Pipeline Parallelism | Split model layers across devices |
| Sequence Parallelism | Split sequence-related activations |
| ZeRO / FSDP | Shard optimizer states, gradients, and parameters |

Data parallelism is conceptually simple: each GPU processes different data, then gradients are synchronized.

Tensor and pipeline parallelism become important when the model itself does not fit on one device.

FSDP and ZeRO-style methods reduce memory duplication by sharding model states across devices.

---

## 13. Checkpointing

Checkpointing is essential because pretraining runs may last days, weeks, or months.

A robust training checkpoint usually contains:

- model weights
- optimizer state
- learning-rate scheduler state
- random number generator state
- data loader position
- tokenizer version or fingerprint
- training configuration

Without the relevant checkpoint state, resuming training may not reproduce the same optimization trajectory or data order.

There are usually two checkpoint types:

| Type | Purpose |
|---|---|
| Training checkpoint | Resume training exactly |
| Release checkpoint | Use model for inference or post-training |

Training checkpoints are often much larger because they include optimizer states.

---

## 14. Monitoring Training

Pretraining should be monitored continuously.

Important signals include:

- training loss
- validation loss
- learning rate
- gradient norm
- throughput
- GPU memory usage
- token/sec
- data mixture proportions
- loss by data domain
- NaN or Inf detection
- checkpoint health

A smooth loss curve is not enough. A model can train stably while learning from poor-quality, duplicated, contaminated, or badly mixed data.

Domain-level monitoring is especially important. If code, math, Persian, English, or long-form text are mixed into the same run, each domain should be tracked separately.

---

## 15. Common Pretraining Failures

#
## Data Problems

Poor data often causes poor models, even when the architecture and optimizer are correct.

Common issues include:

- excessive duplication
- benchmark contamination
- broken Unicode normalization
- bad language identification
- low-quality OCR
- toxic or spam-heavy data
- malformed code
- document boundary corruption

#
## Optimization Problems

Training may fail due to unstable optimization.

Common symptoms include:

- loss divergence
- repeated loss spikes
- gradient explosions
- NaNs
- sensitivity to restart
- unstable mixed precision

#
## Scaling Problems

Distributed systems introduce their own failure modes.

Examples include:

- slow interconnect communication
- inefficient batch construction
- checkpoint bottlenecks
- GPU underutilization
- dataloader starvation
- inconsistent random seeds across workers

---

#
# 16. Practical Pretraining Checklist

Before starting a serious pretraining run, verify:

- The tokenizer is finalized and versioned.
- Special tokens are fixed.
- Data cleaning and deduplication are complete.
- Document boundaries are represented correctly.
- Sequence packing is tested.
- Loss masking is correct for every token type that should or should not contribute to the objective.
- Causal masking is correct.
- Training and validation sets are separated.
- Benchmark contamination checks are performed.
- Optimizer and scheduler settings are logged.
- Mixed precision is tested at small scale.
- Checkpoint resume is tested before the full run.
- Throughput is measured.
- Monitoring dashboards are ready.
- A small overfitting test succeeds.
- A short pilot run produces a healthy loss curve.

---

## 17. Key Takeaways

Pretraining is the stage where an LLM learns its broad capabilities from raw token streams.

The dominant objective is next-token prediction with causal masking.

The main engineering challenge is not only model design, but the coordination of data quality, tokenizer stability, sequence packing, optimization, distributed systems, and monitoring.

A successful pretraining run is reproducible, observable, stable, and data-aware.

Post-training can improve behavior, but it cannot fully compensate for a weak pretraining foundation.

---

## Common Failure Modes

- **Loss instability:** inspect learning rate, numerical precision, data integrity, and gradient statistics.
- **Throughput regression:** profile the input pipeline, accelerator utilization, communication, and checkpoint overhead.
- **Validation plateau:** inspect the data mixture, optimization schedule, and the relationship between training and held-out loss.

## Review Questions

1. How do sequence length and global batch size determine tokens per update?
2. Why is learning-rate scheduling important during large-scale training?
3. What signals would you inspect when a training run becomes unstable?
---
[Previous: Transformer Architecture](./04-transformer-architecture.md) |
[Contents](./index.md) |
[Next: scaling-laws](./06-scaling-laws.md)
---