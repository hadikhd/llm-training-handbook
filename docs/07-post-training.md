---
id: Post-Training
title: Post-Training
sidebar_label: Post-Training
sidebar_position: 8
description: How supervised and preference-based methods shape a pretrained model into a useful assistant or specialized model.
---

<div className="chapter-hero">

![Chapter 7 — Post-training](/static/static/img/chapters/post-training.png)

</div>
[Previous: Scaling Laws](./06-scaling-laws.md) |
[Contents](./index.md) |
[Next: Evaluation](./08-evaluation.md)

---
## Learning Objectives

By the end of this chapter, you should be able to:

- Explain how post-training changes the behavior of a pretrained model.
- Understand SFT, PEFT, preference data, reward modeling, RLHF, PPO, and DPO.
- Distinguish changes in behavior from changes in underlying pretrained knowledge.
- Diagnose catastrophic forgetting, over-optimization, format overfitting, and preference-data noise.


---


## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**


# Post-Training for LLM Fine-Tuning

Post-training is the stage where a pretrained language model is turned into a usable assistant, domain model, or task-specialized system. Pretraining teaches a model broad statistical structure from large-scale text. Post-training shapes how that knowledge is accessed, formatted, constrained, and aligned with user intent.

A useful mental model is:
```text
pretraining -> base model
post-training -> instruction-following / preference-aligned / task-adapted model
```
Pretraining optimizes next-token prediction over broad corpora. Post-training continues training on narrower, higher-signal data such as instructions, conversations, demonstrations, human preferences, tool-use traces, or domain-specific examples.

Post-training usually changes behavior more than raw knowledge. It can improve the model's ability to expose knowledge already represented in its parameters, follow instructions, refuse unsafe requests, maintain conversational format, and prefer helpful responses. It should not be treated as a reliable substitute for weak pretraining or missing domain coverage.

The main post-training methods are:

- Supervised Fine-Tuning (SFT)
- Parameter-Efficient Fine-Tuning (PEFT)
- Reward Modeling
- Reinforcement Learning from Human Feedback (RLHF)
- Proximal Policy Optimization (PPO)
- Direct Preference Optimization (DPO)

Instruction tuning is often described as Supervised Fine-Tuning because the model is explicitly fine-tuned on instruction datasets. The course material states this directly: *"Instruction tuning is often also called Supervised Fine-Tuning (SFT)"* because the model is fine-tuned with a dataset of instructions 

---

## 1. Base Models and Post-Trained Models

A base model is trained to continue text. Given a prefix, it predicts likely next tokens. This makes it flexible, but not necessarily obedient, safe, concise, or conversational.

For example, a base model given:

```text
Explain gradient descent.

may continue with a textbook paragraph, a list of unrelated examples, a training-data-like fragment, or even another prompt. It has not necessarily learned that the user expects a direct answer.
```
A post-trained model is optimized to interpret the same input as an instruction:

```text
User: Explain gradient descent.
Assistant: Gradient descent is an optimization algorithm...
```
The difference is not only formatting. Post-training teaches the model:

- what counts as a helpful answer
- how to follow task instructions
- how to maintain dialogue roles
- how to avoid irrelevant continuations
- how to prefer responses that humans or preference models rank highly
- how to balance helpfulness, honesty, harmlessness, and domain constraints

This is why post-training is central to modern LLM product quality.

---

## 2. Supervised Fine-Tuning

Supervised Fine-Tuning trains the model on demonstrations of desired behavior. Each example usually contains an instruction, optional context, and a target response.

A minimal SFT example is:

```json
{
  "instruction": "Summarize the following paragraph.",
  "input": "Large language models are trained on broad text corpora...",
  "output": "LLMs are trained on large text datasets to learn general language patterns."
}
```

The instruction-tuning material defines the instruction as the command we want the model to follow and the output as the desired response the model should generate 

For causal language models, SFT is still next-token prediction. The difference is the data distribution. Instead of arbitrary web text, the model sees structured examples of the behavior we want.

The standard SFT objective is:

$$ \mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t \in \mathcal{A}} \log \pi_{\theta}(y_t \mid x, y_{<t}) $$

where:

- $x$ is the prompt or instruction context
- $y$ is the target answer
- $\mathcal{A}$ is the set of answer tokens used for loss computation
- $\pi_{\theta}$ is the trainable language model policy

The key detail is that SFT generally should not train the model to predict the prompt. It should train the model to predict the assistant response.

---

## 3. Loss Masking

Loss masking is one of the most important implementation details in SFT.

In instruction tuning, the input sequence often contains both the user prompt and the assistant answer:


### Instruction:
Explain overfitting.


### Response:
Overfitting happens when a model performs well on training data but poorly on unseen data.

During training, the full sequence is passed into the causal LM. But the loss should usually be applied only to the response tokens. The course material states this directly: *"During training, we are usually only interested in predicting the response, so we mask the loss for the prompt"* 

Conceptually:

```text
tokens:      [prompt tokens................][response tokens..............]
loss mask:   0 0 0 0 0 0 0 0 0 0 0 0 0     1 1 1 1 1 1 1 1 1 1 1
```

Without loss masking, the model wastes capacity learning to reproduce user prompts and formatting prefixes. With masking, gradients focus on the assistant behavior.

In Hugging Face-style preprocessing, the formatted prompt and response are tokenized into a single sequence. The course examples include a function that returns tokenized `prompt_and_response` with truncation, and also refer to a `f pattern for masking the prompt-only part 

A simplified version looks like:

```python
def format_example(example):
prompt = f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
response = example["output"]

full_text = prompt + response
prompt_ids = tokenizer(prompt).input_ids
full_ids = tokenizer(full_text).input_ids

labels = full_ids.copy()
labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

return {
"input_ids": full_ids,
"labels": labels,
}
```

The value `-100` is commonly ignored by PyTorch cross-entropy loss.

---

## 4. Chat Templates

Chat models require consistent formatting. A chat template serializes roles and messages into the exact text format expected by the model.

A conversation such as:

```json
[
  {"role": "system", "content": "You are a concise assistant."},
  {"role": "user", "content": "What is LoRA?"},
  {"role": "assistant", "content": "LoRA is a parameter-efficient fine-tuning method..."}
]
```
may become:

```text
<|system|>
You are a concise assistant.
<|user|>
What is LoRA?
<|assistant|>
LoRA is a parameter-efficient fine-tuning method...
```

The exact template is model-specific. Using the wrong template can degrade performance because the model has learned behavior conditioned on particular role tokens, delimiters, and assistant prefixes.

Good post-training data should therefore standardize:

- system messages
- user messages
- assistant messages
- tool calls, if applicable
- refusal style
- multi-turn context formatting
- end-of-sequence behavior

For chat SFT, data formatting is part of the training objective, not just preprocessing.

---

## 5. Building SFT Data

SFT data should represent the behavior expected at inference time.

Common data sources include:

- human-written instruction-response pairs
- expert demonstrations
- synthetic instructions generated by stronger LLMs
- task datasets converted into instruction format
- domain-specific examples from production workflows
- tool-use traces and structured reasoning examples

High-quality SFT data should be:

- correct
- instruction-following
- diverse
- format-consistent
- free of duplicated examples
- aligned with deployment policy
- close to expected user queries

SFT data quality often matters more than raw quantity. A small dataset of precise, representative examples can outperform a much larger noisy dataset.

A practical SFT data record usually includes:

```json
{
  "messages": [
{"role": "system", "content": "You are a technical assistant."},
{"role": "user", "content": "Explain KL regularization in RLHF."},
{"role": "assistant", "content": "KL regularization penalizes the policy when it moves too far from a reference model..."}
  ],
  "metadata": {
"domain": "llm_training",
"difficulty": "advanced",
"source": "expert_written"
  }
}

Metadata is not always included in the model input, but it is valuable for filtering, evaluation splits, curriculum design, and auditability.
```

---

## 6. Full Fine-Tuning and PEFT

Full fine-tuning updates all model parameters. This can produce strong adaptation but requires substantial memory for:

- model weights
- gradients
- optimizer states
- activations
- distributed training communication

For large LLMs, full fine-tuning can be expensive or impractical.

Parameter-Efficient Fine-Tuning updates only a small number of additional or selected parameters. The most common method is LoRA.

LoRA freezes the original weight matrix $W$ and learns a low-rank update:

$$ W' = W + \Delta W $$

$$ \Delta W = BA $$

where:

- $A \in \mathbb{R}^{r \times d}$
- $B \in \mathbb{R}^{k \times r}$
- $r$ is the LoRA rank
- $r \ll \min(d, k)$

Instead of updating $W$, training updates only $A$ and $B$. This reduces trainable parameters and optimizer memory.

The Hugging Face instruction-tuning material explicitly uses `peft` to convert a base model to a LoRA model 

PEFT is useful when:

- GPU memory is limited
- multiple domain adapters are needed
- training must be repeated frequently
- the base model should remain unchanged
- adapter merging or routing is desirable

Common LoRA hyperparameters include:

- rank $r$
- scaling factor $\alpha$
- dropout
- target modules, such as attention projections
- learning rate
- whether to train embeddings or output heads

LoRA is not automatically better than full fine-tuning. It is a trade-off. It reduces cost and memory, but may limit adaptation capacity for large distribution shifts.

---

## 7. Preference Data

SFT teaches the model to imitate demonstrations. Preference learning teaches the model to choose between responses.

A preference example usually contains:

```json
{
  "prompt": "Explain RLHF.",
  "chosen": "RLHF fine-tunes a model using human preference feedback...",
  "rejected": "RLHF is when a model learns by itself from the internet..."
}
```

The key signal is comparative. The model does not only see one correct answer. It sees that one answer is preferred over another for the same prompt.

Preference data can capture qualities that are hard to specify with exact targets:

- helpfulness
- factuality
- harmlessness
- concision
- reasoning quality
- refusal quality
- tone
- instruction adherence

Preference data is used in two major ways:

1. Train a reward model, then optimize the policy with RL.
2. Train the policy directly with a preference objective such as DPO.

---

## 8. Reward Modeling

A reward model maps a prompt-response pair to a scalar score:

$$ r_{\phi}(x, y) \in \mathbb{R} $$

where:

- $x$ is the prompt
- $y$ is the response
- $\phi$ are reward model parameters

The reward model should assign a higher score to the preferred response than to the rejected response. The course material states: *"We want to train a reward model such that the score it gives to the preferred response is higher than the score it gives to the rejected response"* 

For pairwise preference data $(x, y_w, y_l)$, where $y_w$ is the chosen response and $y_l$ is the rejected response, a common loss is:

$$ \mathcal{L}_{\text{RM}}(\phi) = -\log \sigma\!\left(r_{\phi}(x, y_w) - r_{\phi}(x, y_l)\right) $$

This follows the Bradley-Terry preference model. The course notes explain that the Bradley-Terry model can be written as a sigmoid applied to the difference between two reward scores , and that the reward model loss is a cross-entropy loss between predicted and true preference.

A reward model is not a truth oracle. It is a learned approximation of preference labels. Its weaknesses can be exploited during RL optimization, causing reward hacking.

---

## 9. RLHF

Reinforcement Learning from Human Feedback uses human preference data to optimize a language model policy.

The RLHF material describes the technique as a way to incorporate human feedback into the fine-tuning process of LLMs 

A standard RLHF pipeline has three stages:

1. Train an SFT model on demonstrations.
2. Train a reward model on human preference comparisons.
3. Optimize the SFT policy against the reward model using reinforcement learning.

The reward model provides a scalar reward for prompt-response pairs. The course material states that once trained, the reward model can provide a scalar reward for any prompt-response pair

The RLHF objective is often written as:

$$ \max_{\pi_{\theta}} \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi_{\theta}(\cdot \mid x)} \left[ r_{\phi}(x, y) - \beta D_{\mathrm{KL}}\!\left( \pi_{\theta}(\cdot \mid x)\;\Vert\;\pi_{\text{ref}}(\cdot \mid x) \right) \right] $$

where:

- $\pi_{\theta}$ is the trainable policy
- $\pi_{\text{ref}}$ is a frozen reference policy, often the SFT model
- $r_{\phi}$ is the reward model
- $\beta$ controls the strength of KL regularization

The KL term prevents the policy from moving too far from the reference model. Without this constraint, the policy may exploit reward model artifacts and produce unnatural or low-quality text.

---

## 10. PPO for LLMs

Proximal Policy Optimization is a policy-gradient algorithm commonly used in RLHF.

PPO stabilizes policy updates by limiting how much the new policy can move away from the old policy in a single update. The PPO material describes the objective as a clipped version of the policy-gradient objective.

Define the probability ratio:

$$ \rho_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)} $$

The clipped PPO objective is:

$$ \mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t\!\left[ \min\!\left( \rho_t(\theta) A_t,\; \operatorname{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] $$

where:

- $A_t$ is the advantage estimate
- $\epsilon$ controls the clipping range
- $\rho_t$ measures how much the policy changed

For LLMs:

- the policy model generates tokens
- the reward model scores completed responses
- the value head estimates expected return
- the reference model regularizes behavior through KL penalty

The PPO training material uses a model with a value head, `AutoModelForCausalLMWithValueHead`, which reflects the need for value estimation during PPO 

The same material also shows loading both the model and a reference model.
 
`PPOTrainer`

PPO is powerful but operationally complex. It requires careful management of:

- reward scaling
- KL coefficient
- rollout generation
- batch size
- value loss
- advantage estimation
- sampling temperature
- reward model quality
- instability from policy drift

This complexity is one reason direct preference methods such as DPO became widely adopted.

---

## 11. Direct Preference Optimization

Direct Preference Optimization trains the policy directly from preference pairs without training a separate reward model and without running an explicit reinforcement learning loop.

The DPO material describes it as a method for fine-tuning LLMs with human preferences that avoids training a separate reward model and simplifies the training process.
DPO starts from the same preference data:

```text
prompt x
chosen response y_w
rejected response y_l
```
The DPO loss is:

$$ \mathcal{L}_{\text{DPO}}(\theta) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \left[ \log \frac{\pi_{\theta}(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi_{\theta}(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right] \right) \right] $$

where:

- $y_w$ is the chosen response
- $y_l$ is the rejected response
- $\pi_{\theta}$ is the trainable policy
- $\pi_{\text{ref}}$ is the frozen reference model
- $\beta$ controls the strength of the preference update relative to the reference model

The course material highlights the derivation path from optimal policy to the DPO objective  and separately discusses deriving the optimal policy solution 

The $\beta$ parameter is important. The Hugging Face DPO material states that beta controls the strength of KL-divergence regularization in the DPO loss 

A larger $\beta$ makes the model respond more aggressively to preference differences. A smaller $\beta$ keeps the model closer to the reference policy.

DPO is attractive because it avoids several RLHF/PPO difficulties:

- no separate reward model is required
- no online rollout loop is required
- no value model is required
- training resembles supervised learning
- implementation is simpler and more stable

However, DPO still depends heavily on preference data quality. If the chosen/rejected pairs are noisy, biased, or weakly separated, DPO can learn the wrong preferences.

---

## 12. Comparing SFT, RLHF, PPO, and DPO

Each post-training method solves a different problem.

| Method | Training Signal | Main Use | Strength | Limitation |
|---|---|---|---|---|
| SFT | Demonstration response | Teach desired behavior format | Simple and stable | Imitates data, does not directly optimize preferences |
| Reward Modeling | Chosen vs. rejected response | Learn scalar preference score | Separates preference learning from policy optimization | Reward model can be exploited |
| RLHF | Reward model + RL | Optimize policy for learned preferences | Strong alignment signal | Complex and unstable |
| PPO | Policy-gradient RL | Standard RLHF optimizer | Controls policy updates with clipping and KL | Operationally expensive |
| DPO | Chosen vs. rejected response | Direct preference tuning | Simpler than PPO, no reward model required | Sensitive to preference data quality |

A common practical recipe is:

```text
base model
 -> SFT
 -> preference tuning with DPO or PPO
 -> evaluation
 -> safety and domain-specific refinement
```
For many teams, DPO is the default preference-tuning method because it is easier to run than PPO. PPO remains useful when online reward optimization, environment interaction, or more complex RL setups are needed.

---

## 13. Practical Post-Training Workflow

A robust post-training workflow usually follows this sequence:

1. Select a base model appropriate for the target domain and budget.
2. Prepare instruction data with consistent formatting.
3. Apply SFT with correct loss masking.
4. Evaluate instruction-following behavior.
5. Collect or generate preference pairs.
6. Train with DPO or train a reward model for RLHF.
7. Monitor regressions, safety, and domain performance.
8. Run human or model-assisted evaluation.
9. Package the final model with the correct tokenizer and chat template.
10. Document training data, hyperparameters, and known limitations.
---

## 14. Failure Modes

Post-training can improve model behavior substantially, but it also introduces failure modes.

### Catastrophic Forgetting

The model may lose capabilities learned during pretraining or earlier fine-tuning. This is more likely when the fine-tuning data is narrow, repetitive, or overly domain-specific.

Mitigations include:

- lower learning rates
- fewer epochs
- mixed-domain training data
- regular evaluation on general benchmarks
- KL regularization
- PEFT instead of full fine-tuning
- replay data from earlier distributions

### Over-Optimization

In RLHF, the policy may exploit flaws in the reward model. It may learn responses that score highly but are verbose, evasive, formulaic, or factually weak.

Mitigations include:

- reward model evaluation
- KL control
- human spot checks
- adversarial preference data
- reward model retraining
- conservative PPO settings

### Format Overfitting

The model may become too dependent on one prompt template or answer style.

Mitigations include:

- template diversity
- realistic multi-turn data
- evaluation across prompt formats
- clear separation between system, user, and assistant roles

### Preference Data Noise

Preference labels may be inconsistent, subjective, or low quality. DPO and reward modeling are both sensitive to this.

Mitigations include:

- annotator guidelines
- pair quality filtering
- agreement measurement
- hard-negative mining
- domain expert review

### Safety Regression

Fine-tuning on domain or user data can weaken safety behavior if refusal, uncertainty, and policy examples are not represented.

Mitigations include:

- safety evaluation sets
- refusal examples
- red-team prompts
- policy-aligned preference pairs
- post-training audits

---

## 15. Key Takeaways

Post-training turns a base model into a usable model. SFT teaches the model to imitate desired responses. Loss masking ensures training focuses on assistant outputs rather than user prompts. PEFT methods such as LoRA make fine-tuning large models cheaper and more modular.

Preference learning goes beyond imitation. Reward modeling learns scalar preference scores from chosen/rejected pairs. RLHF uses those scores to optimize a policy, often with PPO and KL regularization against a reference model. DPO simplifies preference tuning by directly optimizing the policy on preference pairs without a separate reward model or explicit RL loop.

The main engineering challenge is not only choosing the algorithm. It is controlling data quality, formatting, evaluation, stability, and regressions across the full post-training pipeline.

## Common Failure Modes

- **Catastrophic forgetting:** compare task-specific gains against broad held-out evaluations.
- **Format overfitting:** test prompts and response structures outside the training distribution.
- **Preference-data noise:** inspect disagreement, ambiguity, and systematic biases in preference labels.

## Review Questions

1. What does SFT change compared with pretraining?
2. Why can preference optimization improve behavior without adding much factual knowledge?
3. What are the risks of over-optimization and catastrophic forgetting?
---
[Previous: Scaling Laws](./06-scaling-laws.md) |
[Contents](./index.md) |
[Next: Evaluation](./08-evaluation.md)
---