---
id: Evaluation
title: Evaluation
sidebar_label: Evaluation
sidebar_position: 9
description: How to measure quality, safety, robustness, multilingual behavior, and production performance across the LLM lifecycle.
---

<div className="chapter-hero">

![Chapter 8 — Evaluation](/static/static/img/chapters/evaluation.png)

</div>
[Previous: Post-Training](./07-post-training.md) |
[Contents](./index.md) |
[Next: Systems](./09-systems.md)

---

## Learning Objectives

By the end of this chapter, you should be able to:

- Design evaluation as a continuous engineering loop rather than a final benchmark step.
- Choose metrics appropriate to quality, factuality, safety, multilingual behavior, tools, and retrieval.
- Combine automated metrics, human evaluation, model-based judges, and error analysis.
- Build regression and online evaluation practices that protect production behavior.


---

## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**


## 1. Why Evaluation Matters

Evaluation measures whether a language model actually behaves as intended. Training loss tells us whether the model is learning to predict tokens, but it does not fully tell us whether the model is useful, factual, safe, multilingual, robust, or reliable in deployment.

Evaluation is necessary at every stage:
```text
data preparation -> pretraining -> post-training -> safety tuning -> deployment -> monitoring
```
A model can improve on one benchmark while becoming worse in real user workflows. It can become more fluent while becoming less factual. It can follow instructions better in English while regressing in Persian. It can produce valid-looking answers while failing silently on reasoning, citations, retrieval grounding, or tool calls.

Evaluation turns model development from subjective inspection into an engineering process.

---

## 2. What Evaluation Measures

LLM evaluation is multi-dimensional. No single metric captures model quality.

Common evaluation dimensions include:

| Dimension | What It Measures |
|---|---|
| Language modeling loss | How well the model predicts held-out text |
| Instruction following | Whether the model follows user requests correctly |
| Factuality | Whether claims are correct |
| Reasoning | Whether the model can solve multi-step problems |
| Safety | Whether the model handles harmful or sensitive requests appropriately |
| Robustness | Whether behavior remains stable under variation |
| Multilingual ability | Whether performance holds across languages |
| Tool use | Whether the model calls tools correctly |
| Retrieval grounding | Whether answers are supported by retrieved context |
| Calibration | Whether confidence matches correctness |
| Efficiency | Latency, throughput, memory, and serving cost |

A serious evaluation pipeline uses several complementary tests rather than relying on one leaderboard score.

---

## 3. Evaluation Across the Model Lifecycle

Evaluation changes depending on the training stage.

During pretraining, evaluation usually focuses on:

- validation loss
- perplexity
- domain-level loss
- language-level loss
- contamination checks
- downstream zero-shot or few-shot tasks

During post-training, evaluation focuses more on behavior:

- instruction following
- response quality
- refusal behavior
- safety boundaries
- preference win rate
- multilingual chat behavior
- tool-use correctness
- structured-output validity

During deployment, evaluation focuses on real-world reliability:

- latency
- user satisfaction
- failure rate
- hallucination rate
- retrieval grounding
- cost per request
- monitoring drift
- regressions after updates

Each stage requires different metrics and test sets.

---

## 4. Validation Loss and Perplexity

Validation loss is the average loss on held-out data. It is one of the most important signals during pretraining.

For next-token prediction, the loss is usually cross-entropy:

$$
\mathcal{L} = -\log P(\text{correct next token})
$$

Perplexity is derived from cross-entropy:

$$
\operatorname{PPL} = \exp(\mathcal{L})
$$

Lower perplexity means the model assigns higher probability to the correct next tokens.

However, perplexity has limitations:

- It depends on the tokenizer.
- It may not correlate perfectly with instruction-following quality.
- It can improve while factual behavior remains weak.
- It is sensitive to the validation data distribution.
- It does not directly measure safety or usefulness.

Perplexity is useful, but it is not enough.

---

## 5. Held-Out Data

Evaluation data must be separate from training data. If the model has seen evaluation examples during training, the score may reflect memorization rather than generalization.

A proper evaluation set should be:

- held out from training
- deduplicated against training data
- versioned
- representative of target use cases
- stable across model comparisons
- documented with known limitations

For large-scale pretraining, contamination is a serious issue. Public benchmarks often appear in web data, code repositories, tutorials, forums, and synthetic datasets. Without contamination checks, benchmark scores can be misleading.

---

## 6. Benchmark Evaluation

Benchmarks provide standardized tests for comparing models.

Examples of common benchmark categories:

| Category | Example Tasks |
|---|---|
| Knowledge | question answering, factual recall |
| Reasoning | math, logic, multi-step tasks |
| Coding | code generation, debugging, unit tests |
| Reading comprehension | answering from passages |
| Multilingual | translation, QA, natural language understanding |
| Safety | refusal, harmful request handling |
| Tool use | API calls, function calling |
| Long context | retrieval from long documents |

Benchmarks are useful because they are repeatable. But they are incomplete. A model can perform well on benchmarks and still fail in production workflows.

Benchmark results should be treated as indicators, not final proof of model quality.

---

## 7. Task-Specific Evaluation

Production models should be evaluated on tasks that match their intended use.

For example, a model used in a RAG system should be tested on:

- answer correctness
- citation accuracy
- context faithfulness
- refusal when context is insufficient
- handling of conflicting retrieved passages
- Persian and English queries
- long-document synthesis
- source attribution
- robustness to noisy retrieval

A model used for coding should be tested on:

- unit-test pass rate
- code compilation
- bug-fixing accuracy
- dependency awareness
- security issues
- style consistency
- multi-file edits

General benchmarks cannot replace task-specific evaluation.

---

## 8. Human Evaluation

Human evaluation remains important because many qualities are hard to measure automatically.

Human raters can judge:

- helpfulness
- clarity
- correctness
- tone
- completeness
- safety
- instruction compliance
- citation usefulness
- comparative response quality

A common setup is pairwise comparison:

```text
prompt + response A + response B -> choose better response

```
Pairwise evaluation is often more reliable than asking raters to assign absolute scores.

However, human evaluation must be carefully designed. Raters need clear guidelines, representative prompts, blind model labels, quality control, and agreement checks. Without this structure, human evaluation becomes noisy and subjective.

---

## 9. LLM-as-a-Judge

LLM-as-a-judge uses another language model to evaluate responses. It can scale evaluation faster than human review.

A judge model may score responses for:

- relevance
- correctness
- coherence
- instruction following
- safety
- groundedness
- style
- completeness

Example judge prompt structure:

```text
You are evaluating an assistant response.
Given the user request and the candidate answer, score the answer from 1 to 5
for correctness, completeness, and instruction following.
Return JSON only.

```
LLM judges are useful, but they have risks:

- bias toward verbose answers
- bias toward familiar phrasing
- weak detection of subtle factual errors
- preference for responses from similar model families
- inconsistent scores across prompt wording
- vulnerability to prompt injection inside evaluated content

LLM-as-a-judge should be calibrated against human judgments.

---

## 10. Reference-Based and Reference-Free Metrics

Some evaluations compare the model output against a reference answer.

Reference-based metrics include:

- exact match
- F1 score
- BLEU
- ROUGE
- chrF
- semantic similarity

These are useful when the task has a clear expected answer.

But many LLM tasks have multiple valid answers. In these cases, reference-free evaluation may be needed. A judge evaluates the response directly against criteria, context, or rubric.

Examples:


```text
Does the answer follow the instruction?
Is the answer grounded in the provided context?
Does the answer include unsupported claims?
Is the JSON valid?
Is the refusal appropriate?

```
The best evaluation design depends on the task.

---

## 11. Factuality and Hallucination Evaluation

A hallucination occurs when a model produces unsupported or false information while presenting it as true.

Factuality evaluation asks:

- Are the claims correct?
- Are the claims supported by evidence?
- Are citations accurate?
- Does the model distinguish uncertainty from fact?
- Does it refuse when information is missing?
- Does it invent names, dates, papers, or sources?

For RAG systems, factuality should be separated into two questions:


```text
Is the answer correct?
Is the answer supported by the retrieved context?

```
A correct answer may still be ungrounded if it is not supported by the provided sources. A grounded answer may still be incomplete if the retrieval system missed key evidence.

---

## 12. Evaluation for RAG Systems

Retrieval-augmented generation requires evaluation of both retrieval and generation.

Retriever metrics include:

| Metric | Meaning |
|---|---|
| Recall@k | Whether relevant documents appear in top-k results |
| Precision@k | How many retrieved documents are relevant |
| MRR | Rank of the first relevant result |
| nDCG | Ranking quality with graded relevance |
| Hit rate | Whether at least one relevant result is retrieved |

Generator metrics include:

| Metric | Meaning |
|---|---|
| Answer correctness | Whether the final answer is correct |
| Faithfulness | Whether the answer is supported by context |
| Citation accuracy | Whether cited sources support the claims |
| Completeness | Whether the answer covers the required information |
| Abstention quality | Whether the model refuses when context is insufficient |

RAG evaluation should also test failure cases:

- no relevant context
- partially relevant context
- conflicting documents
- outdated documents
- long retrieved passages
- multilingual queries
- ambiguous user questions

---

## 13. Safety Evaluation

Safety evaluation tests whether the model handles risky or disallowed requests appropriately.

A safety suite should include:

- clearly harmful requests
- benign educational requests
- ambiguous requests
- adversarial rephrasing
- multilingual safety prompts
- role-play attempts
- tool-use safety cases
- requests involving private or sensitive data
- over-refusal tests

Safety evaluation must measure both sides:


```text
unsafe compliance rate
over-refusal rate

```
A model that refuses everything is not useful. A model that complies with harmful instructions is not acceptable. Good safety behavior requires boundary precision.

---

## 14. Multilingual Evaluation

Multilingual evaluation should not be hidden inside one aggregate score. Each target language should be evaluated separately.

For Persian, evaluation should include:

- formal Persian
- informal Persian
- Persian-English code switching
- Arabic/Persian character variants
- ZWNJ usage
- domain-specific terminology
- translation quality
- summarization
- question answering
- instruction following
- safety behavior
- RAG groundedness

Important Persian-specific normalization issues include:

```text
ي -> ی
ك -> ک
ZWNJ handling
Arabic diacritics
half-space variants
mixed punctuation
```
A model may appear strong in multilingual benchmarks while still failing on real Persian user queries.

---

## 15. Long-Context Evaluation

Long-context evaluation tests whether a model can use information from long inputs.

It should test:

- retrieval from the beginning, middle, and end of context
- multi-document synthesis
- contradiction detection
- long conversation memory
- irrelevant-context resistance
- citation to exact passages
- robustness to distractors

A model with a large context window does not necessarily use the full context effectively. Some models degrade when relevant information appears in the middle of a long prompt. This is sometimes called the lost-in-the-middle problem.

Long-context evaluation should measure both capacity and reliability.

---

## 16. Tool-Use Evaluation

For tool-using models, evaluation must check behavior before, during, and after tool calls.

Important questions:

- Did the model decide correctly whether a tool was needed?
- Did it choose the correct tool?
- Did it generate valid arguments?
- Did it handle tool errors?
- Did it use the tool result correctly?
- Did it avoid inventing tool outputs?
- Did it produce a final answer consistent with the tool result?

Tool-use evaluation should use schema validation. Textual inspection alone is not enough.

Example:

```text
tool_call_valid = JSON schema validation passes
tool_result_used = final answer reflects returned data
tool_needed = tool call was appropriate for the task

```
---

## 17. Robustness Evaluation

Robustness evaluation checks whether the model remains stable under variation.

Prompt variations may include:

- paraphrasing
- typos
- extra whitespace
- different formatting
- adversarial instructions
- irrelevant context
- multilingual mixing
- reordered information
- longer or shorter prompts

A robust model should not change its answer dramatically when the meaning of the request is unchanged.

Robustness is especially important for production systems because user input is messy and unpredictable.

---

## 18. Calibration and Uncertainty

Calibration measures whether the model’s confidence matches its correctness.

A well-calibrated model should express uncertainty when information is missing, ambiguous, outdated, or outside its competence.

Examples of good uncertainty behavior:


```text
The provided context does not contain enough information to answer.
I cannot verify that claim from the available sources.
There are two possible interpretations of the question.

```
Poor calibration appears as:

- confident false claims
- invented citations
- unsupported certainty
- refusal despite sufficient evidence
- excessive hedging on simple facts

Calibration is central to trustworthiness.

---

## 19. Regression Testing

Every model update can introduce regressions. Regression tests ensure that improvements in one area do not silently break another.

A regression suite should include:

- previously failed examples
- core product workflows
- safety boundary cases
- multilingual prompts
- formatting tests
- tool-call tests
- retrieval-grounding tests
- latency and memory checks

Regression evaluation should compare new checkpoints against a baseline.

Example comparison:


```text
baseline_model vs candidate_model

```
Track both wins and losses. A model with higher average score may still be rejected if it fails critical workflows.

---

## 20. Online Evaluation

Offline evaluation uses fixed datasets before deployment. Online evaluation measures behavior in real usage.

Online signals may include:

- user ratings
- task completion
- click-through
- correction rate
- escalation rate
- refusal rate
- latency
- cost
- repeated query rate
- conversation abandonment
- safety incidents

Online metrics must be interpreted carefully. User behavior is noisy, and product metrics may not directly measure model quality.

A/B testing can compare models in production, but it requires clear success metrics, guardrails, logging, and rollback plans.

---

## 21. Evaluation Data Management

Evaluation datasets should be treated as versioned assets.

For each evaluation set, document:

- dataset name
- version
- source
- license
- task type
- language
- domain
- number of examples
- creation method
- contamination checks
- scoring method
- known limitations

Evaluation prompts should not be casually reused in training data. Once an evaluation set leaks into training, it no longer measures generalization reliably.

---

## 22. Error Analysis

Metrics show that something changed. Error analysis explains why.

Useful error categories include:

| Error Type | Example |
|---|---|
| Instruction error | The model ignores a constraint |
| Factual error | The model states a false claim |
| Grounding error | The answer is not supported by context |
| Reasoning error | The model uses invalid intermediate logic |
| Formatting error | Output violates required schema |
| Safety error | The model refuses incorrectly or complies unsafely |
| Retrieval error | The needed document was not retrieved |
| Language error | The answer degrades in a target language |

Error analysis should produce actionable fixes. If an evaluation only produces a score, it is incomplete.

---

## 23. Common Evaluation Mistakes

Common mistakes include:

- relying on one benchmark
- using contaminated test data
- ignoring multilingual regressions
- measuring only average score
- using vague human-rating guidelines
- using LLM judges without calibration
- mixing retrieval and generation failures
- ignoring latency and cost
- evaluating only happy paths
- selecting checkpoints based on a few examples
- changing prompts between model comparisons
- not versioning evaluation datasets

Evaluation should be reproducible. If a score cannot be reproduced, it should not guide major training decisions.

---

## 24. Practical Evaluation Pipeline

A practical evaluation pipeline may look like this:


```text
define target behaviors
-> create evaluation taxonomy
-> build held-out datasets
-> run automatic metrics
-> run LLM-as-judge where appropriate
-> sample for human review
-> perform error analysis
-> compare against baseline
-> decide release or rollback

```
For a RAG system, the pipeline should separately evaluate:

```text
retriever -> reranker -> compressor -> generator -> final answer

```
This separation is important. If the final answer is wrong, the cause may be bad retrieval, weak ranking, poor prompt construction, insufficient context compression, or model hallucination.

---

## 25. Practical Checklist

Before accepting a model checkpoint, verify:

- Validation loss is stable and comparable.
- Evaluation data is held out from training.
- Benchmarks are checked for contamination.
- Task-specific test sets exist.
- Persian and other target languages are evaluated separately.
- Safety evaluation includes both unsafe compliance and over-refusal.
- RAG evaluation separates retrieval quality from answer quality.
- Tool calls are validated with schemas.
- Structured outputs are parsed automatically.
- Long-context behavior is tested.
- Regression tests compare against the previous model.
- LLM judges are calibrated against human review.
- Error analysis is performed on failures.
- Latency, memory, and serving cost are measured.
- Evaluation datasets and prompts are versioned.

---

## 26. Key Takeaways

Evaluation is not a single benchmark score. It is a complete measurement system for model quality, safety, reliability, cost, and deployment readiness.

Pretraining evaluation focuses on loss, perplexity, and generalization. Post-training evaluation focuses on behavior, instruction following, safety, tool use, and user-facing quality.

For production systems, evaluation must be task-specific. RAG systems need separate retrieval and generation metrics. Multilingual systems need per-language reporting. Tool-using systems need schema-level validation.

A strong evaluation pipeline does more than rank models. It explains failures, detects regressions, guides training decisions, and protects the reliability of the final system.

---

## Common Failure Modes

- **Benchmark improves while real-task quality declines:** inspect distribution mismatch and task-specific failures.
- **Judge scores drift:** validate judge prompts, calibration examples, and agreement with human evaluation.
- **Regression is missed:** strengthen test coverage, data slicing, and versioned evaluation sets.

## Review Questions

1. Why is a single benchmark insufficient for evaluating an LLM?
2. How should offline evaluation connect to regression testing and online monitoring?
3. What information does error analysis provide that a single metric cannot?

[Previous: Post-Training](./07-post-training.md) |
[Contents](./index.md) |
[Next: Systems](./09-systems.md)
```
