---
id: Introduction
title: Introduction
sidebar_label: Introduction
sidebar_position: 2
description: The foundations, terminology, lifecycle, and engineering mental models needed to understand LLM systems.
---
<div className="chapter-hero">

![Introduction](/img/chapters/introduction.png)

</div>

[Previous:Preface](./00-preface.md) |
[Contents](./index.md) |
[Next: Tokenization](./02-data.md)


## Learning Objectives

By the end of this chapter, you should be able to:

- Explain the core probability model behind autoregressive language modeling.
- Distinguish tokens, parameters, activations, hyperparameters, training, inference, and serving.
- Understand the major stages of an LLM development lifecycle.
- Recognize common LLM limitations and why fluent output is not equivalent to correctness.

## 1.1 What Is a Language Model?



---

A language model assigns probabilities to sequences of tokens.

Let a token sequence be:

$$
x_{1:T} = (x_1, x_2, \ldots, x_T)
$$

A language model estimates the joint probability:

$$
P(x_{1:T})
$$

Using the probability chain rule, this joint distribution can be factorized as:

$$
P(x_{1:T}) = \prod_{t=1}^{T} P(x_t \mid x_{\lt t})
$$

where:

$$
x_{\lt t} = (x_1, x_2, \ldots, x_{t-1})
$$


The model therefore learns to predict each token from the tokens that precede it.

For example:

```text
Context: The capital of France is
Target:  Paris

The model computes a probability distribution over its vocabulary:

Paris       0.81
Lyon        0.04
France      0.03
London      0.01
...
```

During generation, one token is selected from this distribution and appended to the context. The process is repeated until the model produces a stopping token or reaches a generation limit.

---

## 1.2 Text, Tokens, and Probability Distributions

Language models do not directly operate on words or characters. A tokenizer converts text into a sequence of discrete token identifiers.

For example:

```text
Transformers process sequences efficiently.

Possible tokens:
["Transform", "ers", " process", " sequences", " efficiently", "."]

Possible token IDs:
[41762, 388, 1920, 16374, 17226, 13]
```

The exact segmentation depends on the tokenizer and its vocabulary.

If the vocabulary contains $V$ tokens, the model produces $V$ logits at each sequence position:

$$
z_t \in \mathbb{R}^{V}
$$

A softmax function converts those logits into probabilities:

$$
P(x_t = i \mid x_{\lt t}) = \frac{\exp(z_{t,i})}{\sum_{j=1}^{V}\exp(z_{t,j})}
$$


The resulting vector defines a categorical probability distribution over the next token.

---

## 1.3 Autoregressive Language Modeling

Most text-generation LLMs use an autoregressive objective.

Autoregressive means that the model generates a sequence one element at a time, conditioning each new prediction on previously available elements.

Given:

```text
Large language models
```

the generation process may look like:

```text
Step 1: Large language models are
Step 2: Large language models are trained
Step 3: Large language models are trained on
Step 4: Large language models are trained on large
Step 5: Large language models are trained on large datasets
```

During training, however, predictions for all positions can usually be computed in parallel because the complete target sequence is already available.

A causal attention mask ensures that position $t$ cannot access future positions:

$$
x_t \text{ may attend only to } x_1, \ldots, x_t
$$

This preserves the autoregressive objective while allowing efficient parallel training.

---

## 1.4 From N-Grams to Neural Language Models

### 1.4.1 N-Gram Models

Traditional statistical language models approximate the next-token probability using only a fixed number of previous tokens.

A trigram model approximates:

$$
P(x_t \mid x_{\lt t})
\approx
P(x_t \mid x_{t-2}, x_{t-1})
$$

Its probabilities can be estimated using corpus counts:

$$
P(x_t \mid x_{t-2}, x_{t-1})
=
\frac{C(x_{t-2}, x_{t-1}, x_t)}{C(x_{t-2}, x_{t-1})}
$$


N-gram models are conceptually simple, but they have important limitations:

* They use a short, fixed context.
* Their count tables grow rapidly with vocabulary size.
* Many valid sequences never occur in the training corpus.
* They generalize poorly to unseen combinations.
* Similar words do not automatically share statistical strength.

Smoothing methods reduce zero-probability problems but do not remove the fundamental limitations.

### 1.4.2 Neural Language Models

Neural language models replace sparse count tables with learned continuous representations.

Tokens are mapped to embeddings:

$$
e_t = E[x_t]
$$

where $E$ is an embedding matrix.

Tokens with similar functions or meanings can develop related representations. The model can therefore generalize across expressions that did not appear in exactly the same form during training.

Early neural language models used feed-forward networks. Recurrent neural networks, LSTMs, and GRUs later improved sequence modeling by maintaining a hidden state.

However, recurrent models process tokens sequentially, making large-scale parallel training difficult. They can also struggle to preserve information over very long distances.

---

## 1.5 Why Transformers Changed Language Modeling

The Transformer replaced recurrence with attention-based sequence processing.

Its central operation is self-attention:

$$
\operatorname{Attention}(Q,K,V)
=

\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}} + M
\right)V
$$

where:

* $Q$ contains query vectors.
* $K$ contains key vectors.
* $V$ contains value vectors.
* $d_k$ is the key dimension.
* $M$ is an optional mask, such as a causal mask.

Transformers became the dominant LLM architecture because they offer:

* Efficient parallel training across sequence positions
* Flexible interaction between distant tokens
* Strong scaling behavior
* Compatibility with highly optimized matrix multiplication
* A modular architecture that can be distributed across many accelerators

The standard Transformer still has limitations. In particular, ordinary self-attention has quadratic complexity with respect to sequence length:

$$
O(T^2)
$$

This affects both computation and memory as the context length increases.

---

## 1.6 What Makes a Language Model "Large"?

There is no universal parameter threshold that defines an LLM.

The term generally refers to a neural language model that combines:

* A large parameter count
* A large and diverse training corpus
* Significant training compute
* Broad capabilities across multiple tasks

Parameter count alone is not sufficient. A large model trained on insufficient or low-quality data may underperform a smaller, well-trained model.

Model quality depends on the interaction of:

$$
\text{Capability}
=

f(
\text{architecture},
\text{parameters},
\text{data},
\text{compute},
\text{optimization},
\text{post-training}
)
$$

Two models with the same number of parameters may behave very differently because of differences in data quality, tokenization, context length, optimization, or post-training.

---

## 1.7 Parameters, Activations, and Hyperparameters

These concepts should not be confused.

### Parameters

Parameters are learned values, including:

* Embedding matrices
* Attention projection matrices
* Feed-forward network weights
* Normalization parameters
* Output projection weights

They are updated during training.

### Activations

Activations are intermediate values produced during a forward pass.

Their memory usage depends strongly on:

* Batch size
* Sequence length
* Hidden dimension
* Number of layers
* Attention implementation
* Activation checkpointing strategy

### Hyperparameters

Hyperparameters are selected by the training team rather than learned directly.

Examples include:

* Learning rate
* Batch size
* Number of layers
* Hidden dimension
* Number of attention heads
* Vocabulary size
* Sequence length
* Weight decay
* Warmup duration

Hyperparameter choices affect stability, efficiency, and final model quality.

---

## 1.8 The LLM Development Lifecycle

A practical LLM development pipeline contains several stages.

### Stage 1: Data Acquisition

Possible sources include:

* Web documents
* Books
* Scientific literature
* Source code
* Reference material
* Conversations
* Licensed private datasets
* Synthetic examples

### Stage 2: Data Processing

Typical operations include:

* Text extraction
* Language identification
* Quality filtering
* Safety filtering
* Exact deduplication
* Near-duplicate removal
* Personally identifiable information handling
* Benchmark decontamination
* Dataset mixing

### Stage 3: Tokenizer Training

The tokenizer defines:

* Vocabulary units
* Token-to-ID mapping
* Special tokens
* Text normalization rules
* Multilingual segmentation behavior

### Stage 4: Model Design

The architecture determines:

* Parameter count
* Number of layers
* Hidden size
* Attention configuration
* Feed-forward dimensions
* Positional representation
* Context length
* Normalization and activation functions

### Stage 5: Pretraining

The model learns general statistical structure through objectives such as next-token prediction.

### Stage 6: Post-training

The pretrained model is adapted using techniques such as:

* Supervised fine-tuning
* Instruction tuning
* Preference optimization
* Reinforcement learning
* Safety tuning
* Tool-use training

### Stage 7: Evaluation

Evaluation covers areas such as:

* Language modeling quality
* Knowledge
* Reasoning
* Coding
* Multilingual performance
* Instruction following
* Safety
* Long-context behavior
* Domain-specific performance

### Stage 8: Inference and Serving

Deployment introduces additional requirements:

* Low latency
* High throughput
* Memory efficiency
* Request scheduling
* Quantization
* KV-cache management
* Monitoring
* Cost control

---

## 1.9 Pretraining and Post-training

### Pretraining

Pretraining usually optimizes a broad language-modeling objective over a large corpus.

Its purpose is to learn:

* Linguistic patterns
* Factual associations
* Common reasoning structures
* Domain knowledge
* Representational features
* General continuation ability

A pretrained model is commonly called a base model.

### Post-training

Post-training changes model behavior after pretraining.

Its purpose may include:

* Improving instruction following
* Teaching conversation formats
* Increasing response usefulness
* Applying behavioral constraints
* Improving tool use
* Adapting to a specific domain

A post-trained model does not necessarily contain dramatically more factual knowledge than its base model. Much of the improvement may come from better elicitation and response selection.

---

## 1.10 Training, Inference, and Serving

### Training

Training includes:

1. A forward pass
2. Loss calculation
3. Backpropagation
4. Gradient synchronization when distributed
5. Optimizer updates

Training stores or manages:

* Model parameters
* Activations
* Gradients
* Optimizer states
* Temporary buffers

### Inference

Inference computes model outputs without gradient-based weight updates.

For autoregressive generation, inference usually has two phases:

* **Prefill:** Process all prompt tokens, usually in parallel.
* **Decode:** Generate additional tokens one at a time.

### Serving

Serving is the production system around inference. It includes:

* Request admission
* Dynamic or continuous batching
* Scheduling
* KV-cache allocation
* Model parallel execution
* Streaming responses
* Rate limiting
* Failure handling
* Observability

A model can be fast in an isolated benchmark but inefficient in production if the serving system handles batching or memory poorly.

---

## 1.11 Foundation, Base, Instruction, and Chat Models

### Foundation Model

A broad model that can support multiple downstream applications. The term may cover both base and adapted models.

### Base Model

A model trained primarily with a general pretraining objective.

It is usually best understood as a continuation model rather than a conversational assistant.

### Instruction-Tuned Model

A model fine-tuned on instruction-response examples.

It learns to interpret requests and produce task-oriented responses.

### Chat Model

An instruction-tuned model trained for multi-turn interaction using a defined conversation template.

The template may include special roles such as:

```text
system
user
assistant
tool
```

Using the wrong chat template can significantly reduce quality, even when the model weights are correct.

---

## 1.12 Emergent and General Capabilities

LLMs can perform tasks that are not represented as separate modules:

* Summarization
* Translation
* Classification
* Question answering
* Code generation
* Information extraction
* In-context learning
* Structured output generation

These behaviors arise from shared representations and the next-token prediction objective.

However, apparent task generality should not be interpreted as universal reliability. Performance varies with:

* Prompt format
* Language
* Domain
* Context length
* Required precision
* Availability of relevant training patterns
* Evaluation method

---

## 1.13 In-Context Learning

In-context learning occurs when a model adapts its behavior based on examples placed in the prompt, without updating its parameters.

### Zero-shot

```text
Classify the sentiment as positive or negative:

The documentation was clear and useful.
```

### One-shot

```text
Text: The service was terrible.
Label: negative

Text: The documentation was clear and useful.
Label: positive
```

### Few-shot

Several demonstrations are supplied before the target example.

In-context learning is useful, but it is sensitive to:

* Example selection
* Example order
* Label wording
* Prompt formatting
* Context length
* Distribution mismatch

It should not be confused with fine-tuning, because no parameter update occurs.

---

## 1.14 Core Limitations

### Hallucination

A model may generate plausible but unsupported claims because its objective rewards likely continuations, not guaranteed truth.

### Finite Context

The model can directly condition only on information available within its context window.

### Knowledge Staleness

Parameters represent information learned during training. They do not automatically update when the world changes.

### Prompt Sensitivity

Small changes in wording or structure may affect output quality.

### Weak Calibration

A confident linguistic style does not necessarily indicate a high probability of correctness.

### Bias and Coverage Gaps

Models inherit biases and imbalances from data, annotations, evaluation criteria, and post-training procedures.

### Computational Cost

Training and serving large models require substantial compute, memory, communication bandwidth, and operational engineering.

### Limited Interpretability

It remains difficult to provide complete causal explanations for many model outputs.

---

## 1.15 LLMs and Retrieval-Augmented Generation

Retrieval-Augmented Generation, or RAG, combines a language model with an external retrieval system.

A simplified pipeline is:

```text
User query
   |
   v
Query processing
   |
   v
Retriever
   |
   v
Relevant documents
   |
   v
Prompt construction
   |
   v
Language model
   |
   v
Grounded response
```

RAG can improve:

* Access to current information
* Domain-specific accuracy
* Source attribution
* Auditability
* Knowledge coverage

RAG does not automatically eliminate hallucinations. It can fail because of:

* Poor retrieval
* Missing evidence
* Irrelevant context
* Context truncation
* Weak prompt construction
* Failure to follow the evidence
* Incorrect citation mapping

The retriever and generator must therefore be evaluated both separately and end to end.

---

## 1.16 Practical Example: A Minimal Causal LM Objective

The following pseudocode illustrates next-token training:

```python
input_ids = batch[:, :-1]
target_ids = batch[:, 1:]

logits = model(input_ids)

loss = cross_entropy(
    logits.reshape(-1, vocabulary_size),
    target_ids.reshape(-1),
)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

The input and target sequences are shifted by one token:

```text
Input:   [BOS, Large, language, models, are]
Target:  [Large, language, models, are, useful]
```

In production training code, additional concerns include:

* Padding masks
* Sequence packing
* Distributed synchronization
* Mixed precision
* Gradient accumulation
* Gradient clipping
* Learning-rate scheduling
* Checkpointing
* Fault recovery

---

## 1.17 Practical Engineering Notes

* Treat tokenizer design as part of the model, not as a replaceable preprocessing detail.
* Track tokens rather than only documents or words when measuring training volume.
* Separate base-model evaluation from instruction-following evaluation.
* Evaluate prefill and decode performance independently during inference.
* Measure data quality and duplication before increasing model size.
* Validate chat templates and special tokens before diagnosing model quality.
* Record exact dataset, code, tokenizer, and checkpoint versions.
* Use held-out evaluations throughout training, not only after completion.
* Distinguish model limitations from retrieval, prompting, and serving failures.

---

## 1.18 Common Pitfalls

### Pitfall 1: Equating Parameter Count with Quality

A larger parameter count does not guarantee a better model. Training data, token budget, architecture, and post-training quality matter.

### Pitfall 2: Treating an LLM as a Database

An LLM is a probabilistic generator. It does not provide transactional storage, guaranteed lookup, or automatic provenance.

### Pitfall 3: Treating Fluent Output as Correct Output

Fluency and factual correctness are different properties.

### Pitfall 4: Comparing Models with Different Prompt Formats

A model may appear weak when evaluated with an incorrect chat template or unsuitable prompting strategy.

### Pitfall 5: Using Perplexity as the Only Metric

Perplexity measures next-token prediction, but does not fully measure reasoning, instruction following, safety, or factual grounding.

### Pitfall 6: Ignoring Data Contamination

Benchmark overlap may inflate measured performance without improving general capability.

### Pitfall 7: Ignoring Systems Constraints

A theoretically suitable architecture may be impractical if it cannot be trained or served efficiently on available hardware.

---

## 1.19 Summary

* A language model estimates probabilities over token sequences.
* Autoregressive LLMs factorize sequence probability into next-token predictions.
* Tokenization converts text into discrete units processed by the model.
* Transformers use attention to build contextual token representations.
* LLM capability depends on architecture, data, compute, optimization, and post-training.
* Pretraining develops broad continuation ability; post-training shapes useful behavior.
* Training, inference, and production serving are distinct engineering workloads.
* LLMs are powerful probabilistic models, but they are not guaranteed factual databases.
* RAG and tool use can extend model capabilities, but introduce additional failure modes.
* Reliable LLM development requires coordinated work across data, modeling, evaluation, and systems.

---

## Review Questions

1. How does the chain rule support autoregressive language modeling?
2. Why do n-gram models generalize poorly to unseen sequences?
3. What advantages did Transformers introduce over recurrent models?
4. How do parameters, activations, and hyperparameters differ?
5. Why is a base model not automatically a useful conversational assistant?
6. What is the difference between prefill and decode during inference?
7. Why can low perplexity fail to predict instruction-following quality?
8. How does in-context learning differ from fine-tuning?
9. Why does RAG not completely eliminate hallucination?
10. Which parts of the LLM lifecycle are primarily systems problems?

---

## Further Reading

1. Bengio, Y., Ducharme, R., Vincent, P., and Jauvin, C.
   *A Neural Probabilistic Language Model*. Journal of Machine Learning Research, 2003.

2. Vaswani, A. et al.
   *Attention Is All You Need*. NeurIPS, 2017.

3. Radford, A. et al.
   *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019.

4. Brown, T. et al.
   *Language Models are Few-Shot Learners*. NeurIPS, 2020.

5. Bommasani, R. et al.
   *On the Opportunities and Risks of Foundation Models*. 2021.

6. Zhao, W. X. et al.
   *A Survey of Large Language Models*. 2023.

---

[Previous: Preface](./00-preface.md) | [Next: Data for LLM Training](./02-data.md)
