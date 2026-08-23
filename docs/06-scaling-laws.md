---
id: Scaling Laws
title: Scaling Laws
sidebar_label: Scaling Laws
sidebar_position: 7
description: How model size, data, compute, context, and infrastructure interact when scaling LLM training.
---

<div className="chapter-hero">

![Chapter 6 — Scaling Laws](/img/chapters/scaling-laws.png)

</div>

[Previous: Pretraining](./05-pretraining.md) |
[Contents](./index.md) |
[Next: Post-training](./07-post-training.md)

---
## Learning Objectives

By the end of this chapter, you should be able to:

- Understand how parameters, training tokens, compute, and data quality interact.
- Use scaling laws as empirical engineering tools rather than as exact predictions.
- Distinguish undertraining, overtraining, and inefficient allocation of compute.
- Plan small experiments before committing to expensive large-scale training runs.


---


## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**


## 1. Why Scaling Laws Matter

Scaling laws describe how model performance changes as we increase the main resources used during training:

- model parameters
- training tokens
- compute
- data quality
- context length
- optimization budget

For large language models, scaling is not just about making the model bigger. A larger model trained on too few tokens can be worse than a smaller model trained well. A huge dataset with poor filtering can be worse than a smaller, cleaner dataset. More GPUs can reduce training time, but they do not automatically improve the model unless the training recipe is balanced.

Scaling laws help answer practical questions:

- How large should the model be?
- How many tokens should it see?
- How much compute is needed?
- Is the model undertrained?
- Is adding more data better than adding more parameters?
- When does scaling stop being efficient?

In production LLM development, scaling laws are used as engineering tools. They do not give perfect predictions, but they provide a disciplined way to reason about budget, model size, data volume, and expected quality.

---

## 2. The Main Scaling Variables

The most important variables are:

| Variable | Meaning |
|---|---|
| Parameters | Number of trainable weights in the model |
| Tokens | Number of training tokens processed |
| Compute | Total training FLOPs |
| Dataset quality | Usefulness, cleanliness, and diversity of data |
| Context length | Maximum number of tokens processed per sequence |
| Optimization | Optimizer, schedule, precision, stability, batch size |

A simplified view is:

$$
Q = f(N, D, C, Q_{\text{data}}, O)
$$

where $N$ is parameter count, $D$ is training-token count, $C$ is compute, $Q_{\text{data}}$ represents data quality, and $O$ represents the optimization/training recipe.


This is only a simplification. Real training runs depend on many details, including tokenizer behavior, data mixture, deduplication, architecture, and distributed systems efficiency.

Still, the central idea is useful: model quality improves predictably when resources are scaled in a balanced way.

---

## 3. Parameters

Parameters represent the capacity of the model.

More parameters generally allow the model to store and compute more complex patterns. Larger models often show better reasoning, stronger few-shot learning, better multilingual transfer, and more robust instruction following after post-training.

However, parameters are expensive.

Increasing parameters affects:

- training memory
- inference memory
- training FLOPs
- inference latency
- serving cost
- checkpoint size
- communication overhead in distributed training

A bigger model is not automatically better. If the model is too large for the available token budget, it may be undertrained. It will have enough capacity but insufficient data exposure to use that capacity effectively.

---

## 4. Training Tokens

Training tokens are the amount of textual evidence the model sees.

A token is not the same as a word. Token count depends on the tokenizer, language, normalization rules, and vocabulary design.

For pretraining, token count is one of the most important scale variables.

Examples:

```text
10B tokens
100B tokens
1T tokens
10T tokens
```

More tokens usually improve performance, but only if the tokens are useful. Low-quality, duplicated, corrupted, or irrelevant tokens can waste compute and damage model behavior.

A smaller amount of high-quality data may outperform a much larger noisy corpus.

This is especially important for multilingual training. If Persian is only a small fraction of the training mixture, the model may tokenize Persian inefficiently, learn weaker representations, and perform poorly on Persian tasks even if the total token count is large.

---

## 5. Compute

Compute is the amount of numerical work performed during training, commonly approximated in floating-point operations (FLOPs).

For a dense decoder-only Transformer, a common first-order estimate of training compute is:


$$
C \approx 6ND
$$

where $N$ is the number of model parameters and $D$ is the number of training tokens.


This is a simplified rule of thumb. Actual compute depends on architecture, sequence length, attention implementation, activation checkpointing, parallelism strategy, and hardware efficiency.

Compute is usually measured in FLOPs:

```text
floating-point operations
```

In practice, engineers also care about:

- GPU hours
- wall-clock time
- cluster utilization
- tokens per second
- cost per training run
- cost per useful checkpoint

The best model is not always the largest model. It is often the model that uses the available compute most efficiently.

---

## 6. Empirical Scaling Laws

Early scaling-law studies observed that language model loss often improves smoothly as model size, data size, and compute increase.

A simplified statement is:

```text
loss decreases predictably as scale increases
```

This matters because it allows teams to run smaller experiments and estimate the behavior of larger runs.

For example, instead of immediately training a 70B-parameter model, a team may train models at smaller scales:

```text
125M -> 350M -> 1B -> 3B -> 7B

Then they analyze loss trends and estimate what might happen at larger scale.
```

Scaling laws are empirical, not theoretical guarantees. They depend on:

- data distribution
- tokenizer
- architecture
- optimizer
- training duration
- evaluation set
- loss measurement
- contamination control

They are useful because LLM training is too expensive to tune entirely by trial and error at full scale.

---

## 7. Compute-Optimal Training

A model is compute-optimal, in the scaling-law sense, when model size and training-token count are chosen to make effective use of a fixed compute budget.

If the compute budget is fixed, there is a trade-off:

```text
larger model + fewer tokens
smaller model + more tokens
```
Older training recipes often favored very large models trained on relatively fewer tokens. Later results showed that many models were undertrained: they had many parameters but had not seen enough data.


The compute-optimal perspective says:

```text
for a fixed compute budget, train a smaller model on more tokens if the larger model would be data-starved
```

This changed how many teams think about LLM training.

The practical message is simple: do not only scale parameters. Scale tokens too.

---

## 8. Chinchilla-Style Perspective

The Chinchilla study popularized the idea that many earlier large language models were undertrained relative to their parameter counts, motivating a stronger emphasis on jointly scaling parameters and training tokens.

The main lesson was not that every model must follow one exact ratio. The important lesson was that token budget and parameter count must be planned together.

A simplified comparison:

| Model Type | Problem |
|---|---|
| Too many parameters, too few tokens | Undertrained |
| Too few parameters, too many tokens | Capacity-limited |
| Balanced parameters and tokens | More compute-efficient |

This perspective encouraged training smaller models for longer.

For example, a well-trained 7B model may outperform a poorly trained 13B model. A well-trained 13B model may be more useful than a larger model that consumed more compute but did not receive enough data.

---

## 9. Undertraining and Overtraining

### Undertraining

A model is undertrained when it has not seen enough useful tokens for its size.

Symptoms may include:

- validation loss still decreasing rapidly at the end of training
- weak downstream performance for the model size
- poor factual coverage
- unstable post-training behavior
- worse performance than smaller, better-trained models

Undertraining wastes parameter capacity.

### Overtraining

The relationship between additional training and generalization is more subtle.

In LLM pretraining, continuing to train on high-quality diverse data can still improve the model. Problems can appear when additional training repeatedly exposes the model to limited, duplicated, or narrow data, especially when evaluation performance begins to diverge from training behavior.

Symptoms may include:

- memorization
- benchmark contamination
- reduced diversity
- degradation on held-out domains
- overfitting to a narrow style
- weaker generalization

Overtraining is not only about the number of tokens. It is about the relationship between token count, diversity, duplication, and evaluation behavior.

---

## 10. Model Size vs Token Budget

A practical design question:

```text
Should we train a larger model, or train a smaller model on more tokens?
```

The answer depends on the deployment goal.

A larger model may be preferred when:

- maximum quality matters more than serving cost
- reasoning performance is critical
- inference infrastructure is available
- latency constraints are relaxed
- the model will be distilled later

A smaller, longer-trained model may be preferred when:

- inference cost matters
- latency matters
- deployment is on limited hardware
- many users must be served concurrently
- the dataset is large and high quality
- the model must be easy to fine-tune or adapt

For many production systems, the best model is not the largest possible model. It is the best model that can be served reliably and economically.

---

## 11. Scaling Data Quality

Scaling data volume is easier than scaling data quality.

Adding more raw web text may increase token count, but it can also introduce:

- duplication
- spam
- boilerplate
- broken encoding
- low-information text
- synthetic artifacts
- benchmark leakage
- toxic or unsafe content
- inconsistent formatting

High-quality data improves sample efficiency. This means the model learns more per token.

A clean 500B-token dataset can be more valuable than a noisy 2T-token dataset.

Data quality includes:

- correctness
- diversity
- coverage
- freshness
- formatting consistency
- language balance
- deduplication
- domain relevance
- document integrity

Scaling laws based only on token count can be misleading when data quality changes.

---

## 12. Multilingual Scaling

Multilingual models introduce additional scaling challenges.

A model trained mostly on English may perform strongly in English while remaining weak in lower-resource languages. This is not only because of token count. It also depends on tokenizer efficiency, script normalization, data quality, and domain coverage.

For Persian, important issues include:

- Arabic vs Persian character normalization
- ZWNJ handling
- inconsistent spacing
- mixed Persian-English text
- low-quality OCR
- duplicated news or web boilerplate
- informal writing variants
- code-switching
- underrepresentation in technical corpora

A multilingual model needs enough high-quality tokens per language. It also needs a tokenizer that does not fragment some languages excessively.

If Persian text has high token fertility, the same semantic content requires more tokens, consuming more context budget and increasing both training and inference cost. This affects both training efficiency and inference cost.

---

## 13. Context Length Scaling

Context length is another dimension of scale.

Increasing context length allows the model to process longer documents, conversations, code files, and retrieval-augmented inputs.

But long context is expensive. Standard attention has quadratic complexity with respect to sequence length:


$$
C_{\text{attention}} \propto T^2
$$

where $T$ is the sequence length.


Modern implementations such as FlashAttention improve IO efficiency and memory usage, but the underlying full-attention operation still has quadratic dependence on sequence length.

Long-context training requires more than changing a configuration value. It may require:

- positional embedding strategy
- RoPE scaling or interpolation
- long-document data
- long-context evaluation
- memory-efficient attention
- careful packing
- training stability checks

A model trained mostly on short sequences may not use long context effectively, even if its architecture technically supports a large context window.

---

## 14. Scaling Batch Size

Batch size affects optimization and throughput.

The relevant quantity is usually tokens per update:


$$
N_{\text{tokens/update}} = B_{\text{global}} \times T
$$

   

Larger batches can improve hardware utilization and reduce gradient noise. But very large batches may require learning-rate adjustments and can reduce the number of optimizer updates for a fixed token budget.


Important concepts:

| Term | Meaning |
|---|---|
| Micro-batch | Batch processed per device before accumulation |
| Global batch | Total batch across all devices |
| Gradient accumulation | Accumulating gradients across micro-batches |
| Tokens per update | Total tokens used per optimizer step |

Scaling batch size is not only a memory question. It changes training dynamics.

---

## 15. Scaling Infrastructure

At small scale, training is mostly a modeling problem. At large scale, it becomes a systems problem.

Scaling infrastructure involves:

- GPU memory
- GPU interconnect bandwidth
- storage throughput
- dataloader efficiency
- checkpoint writing speed
- distributed communication
- fault tolerance
- experiment tracking
- reproducibility
- cluster scheduling

A model can be theoretically compute-optimal but practically inefficient if the training system cannot feed GPUs fast enough.

Common scaling bottlenecks include:

- GPUs waiting for data
- slow checkpoint saves
- communication overhead from tensor parallelism
- imbalanced pipeline stages
- inefficient sequence packing
- unstable multi-node runs
- poor recovery after preemption or failure

Systems efficiency directly changes the cost of scaling.

---

## 16. Budget-Aware Model Design

A useful model plan starts with constraints.

Key questions:

- What is the compute budget?
- How many tokens are available?
- What is the target latency?
- What hardware will serve the model?
- Is the model for research, product, or internal use?
- Is multilingual performance required?
- Is domain specialization required?
- Will the model be post-trained?
- Will the model be quantized?
- How often will it be retrained?

A research model may optimize for benchmark performance. A product model may optimize for cost, latency, reliability, and update frequency.

A budget-aware design does not ask:

```text
What is the largest model we can train?
```
It asks:
```text
What is the best model we can train, evaluate, serve, and maintain?
```

---

## 17. Small-Scale Experiments

Before launching a large run, teams often run smaller experiments to estimate scaling behavior.

Examples:

```text
same data, different model sizes
same model, different token budgets
same compute, different parameter-token trade-offs
same model, different data mixtures
same model, different learning-rate schedules
```

Small runs help detect:

- bad data mixtures
- unstable learning rates
- tokenizer problems
- loss masking bugs
- poor sequence packing
- domain imbalance
- contamination issues
- weak validation design

A small model cannot reveal every behavior of a large model, but it can catch many expensive mistakes early.

---

## 18. Reading Scaling Curves

A scaling curve usually plots loss or evaluation score against model size, tokens, or compute.

Useful patterns:

| Pattern | Possible Interpretation |
|---|---|
| Loss decreases smoothly | Training recipe is likely stable |
| Loss plateaus early | Data, capacity, or optimization bottleneck |
| Larger model worse than smaller model | Undertraining or instability |
| Validation loss diverges from training loss | Overfitting or data mismatch |
| Domain loss improves unevenly | Data mixture imbalance |
| Loss spikes repeatedly | Optimization or systems instability |

Scaling curves should be read with domain-level metrics. A model can improve on average while getting worse on a specific language, domain, or task type.

---

## 19. Common Misconceptions

### Misconception 1: Bigger is always better

Bigger models can be stronger, but only when trained and served properly.

### Misconception 2: More data is always better

More data helps when it is useful, diverse, and clean. Bad data can waste compute or degrade behavior.

### Misconception 3: Scaling laws predict everything

Scaling laws describe broad trends. They do not replace evaluation, data audits, or engineering judgment.

### Misconception 4: Context length can be increased for free

Long context increases compute, memory use, and evaluation complexity.

### Misconception 5: Training cost is the only cost

Serving cost may dominate over time. A model that is cheap to train but expensive to serve may be a poor production choice.

---

## 20. Practical Scaling Checklist

Before scaling a run, verify:

- The tokenizer is final and versioned.
- The dataset is deduplicated and filtered.
- Data mixture weights are documented.
- Validation sets are clean and separated.
- Benchmark contamination checks are performed.
- Small-scale runs show stable loss curves.
- Training tokens and model size are planned together.
- The learning-rate schedule is tested.
- Batch-size scaling is tested.
- Checkpoint resume works.
- Throughput is measured in tokens/sec.
- GPU utilization is monitored.
- Dataloader performance is sufficient.
- Loss is tracked by domain and language.
- Persian and other target languages have separate validation sets.
- Inference constraints are considered before choosing model size.
- Total cost includes training and serving.

---

## 21. Key Takeaways

Scaling laws help engineers reason about how model quality changes with parameters, tokens, compute, and data quality.

The central lesson is balance. A model should not be scaled only by increasing parameter count. Token budget, dataset quality, optimizer stability, infrastructure, and serving constraints must scale with it.

Compute-optimal training often favors models that are smaller than expected but trained on more tokens.

Data quality changes scaling behavior. Token count alone is not enough.

For production systems, the best model is the one that satisfies the full lifecycle constraints: training, evaluation, post-training, inference, monitoring, cost, and maintenance.

---
## Common Failure Modes

- **Larger model without better quality:** verify that model size, data, and compute are scaled in a balanced regime.
- **Small-scale trend fails at scale:** verify that the data distribution, optimization regime, and infrastructure remain comparable.
- **Cost grows faster than quality:** measure marginal quality improvement against compute and infrastructure cost.

## Review Questions

1. Why can adding parameters be less useful than adding training tokens?
2. What does compute-optimal training mean operationally?
3. Why should small-scale scaling experiments precede a large training run?

---
[Previous: Pretraining](./05-pretraining.md) |
[Contents](./index.md) |
[Next: Post-training](./07-post-training.md)
---