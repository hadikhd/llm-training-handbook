---
id: Glossary
title: Glossary
sidebar_label: Glossary
sidebar_position: 13
---
# Glossary

## A
- **Activations**: Intermediate values produced during a neural network forward pass.
- **All-reduce**: A collective communication operation that sums values across workers and shares the result back to all of them.
- **ALiBi (Attention with Linear Biases)**: A positional encoding method that adds distance-based bias to attention scores.
- **Autoregressive Language Modeling**: A generation setup where each token is predicted from previous tokens only.
- **Autoregressive Decoding**: Token-by-token generation conditioned on previously generated tokens.
- **Alignment**: Training and adjustment methods that make model behavior better match human intent, preferences, and safety constraints.

## B
- **Base Model**: A pretrained model before instruction or preference post-training.
- **Benchmark**: A standardized dataset or suite used to compare model performance.
- **Benchmark Contamination**: Overlap between training data and evaluation benchmarks.
- **BF16 (BFloat16)**: A 16-bit floating-point format with FP32-like exponent range.
- **BPE (Byte Pair Encoding)**: A subword tokenization algorithm based on frequent pair merges.
- **Bradley-Terry Preference Model**: A pairwise comparison model used in preference learning.

## C
- **Causal Attention Mask**: A mask that prevents a token from attending to future tokens.
- **Checkpointing**: Saving model and training state so training can resume later.
- **Chat Template**: A structured format for representing multi-turn conversations.
- **Compute**: The total training or inference work, often measured in FLOPs.
- **Context Length**: The maximum number of tokens a model can process in one sequence.
- **Continuous Batching**: Dynamically adding and removing requests from an active inference batch.
- **Corpus**: A large collection of text used for training or evaluation.
- **Cross-Entropy Loss**: The standard loss used for next-token prediction.

## D
- **Data Acquisition**: Collecting raw training data from sources such as web pages, books, code, or conversations.
- **Data Deduplication**: Removing exact or near-duplicate content from a corpus.
- **Data Filtering**: Removing unwanted data using rule-based or model-based criteria.
- **Data Mixture**: The weighted combination of multiple datasets for training.
- **Data Parallelism**: Replicating the model across devices and splitting batches across workers.
- **Decode**: The inference phase where new tokens are generated one at a time.
- **Decoding**: Converting model probabilities into actual generated tokens.
- **Deduplication**: Removal of duplicate or near-duplicate training content.
- **DPO (Direct Preference Optimization)**: A post-training method that learns from pairwise preference data without a full RL loop.
- **Dropout**: A regularization method that randomly disables activations during training.

## E
- **Embedding**: A dense vector representation of a discrete token or item.
- **Embedding Matrix**: The lookup table that maps token IDs to embeddings.
- **Embedding Lookup**: Retrieving a token embedding from its ID.
- **Emergent Capabilities**: Behaviors that appear at scale without being explicitly programmed as separate modules.
- **Evaluation**: The process of measuring model performance across tasks and criteria.
- **Expert Parallelism**: Distributing experts of a MoE model across devices.
- **FP8**: An 8-bit floating-point format used for higher efficiency with tighter numerical constraints.
- **FP16**: A 16-bit floating-point format commonly used in mixed precision training.
- **FP32**: A 32-bit floating-point format with high precision and range.
- **FSDP (Fully Sharded Data Parallel)**: A training strategy that shards parameters, gradients, and optimizer states across workers.
- **Feed-Forward Network (FFN)**: The position-wise neural sublayer inside a Transformer block.

## F
- **Factuality**: The degree to which model output matches verifiable facts.
- **Few-Shot Learning**: In-context learning with several examples in the prompt.
- **Fine-Tuning**: Further training a pretrained model on a narrower objective or dataset.
- **FlashAttention**: An optimized attention implementation that reduces memory movement.
- **Foundation Model**: A broad pretrained model intended for many downstream uses.
- **Greedy Decoding**: Selecting the highest-probability token at each generation step.
- **Ground Truth**: The reference label or answer used for training or evaluation.
- **Gradient Accumulation**: Summing gradients over multiple mini-batches before updating parameters.
- **Gradient Clipping**: Limiting gradient magnitude to improve training stability.
- **Hallucination**: Fluent but unsupported or factually incorrect model output.
- **Hidden Dimension**: The size of the model’s internal representation space.

## I
- **In-Context Learning**: Solving a task from examples or instructions placed directly in the prompt.
- **Inference**: Using a trained model to produce outputs without updating its weights.
- **Instruction Tuning**: Fine-tuning on instruction-response examples to improve task following.
- **Instruction-Tuned Model**: A model adapted to follow instructions and respond to requests.
- **LLM (Large Language Model)**: A large pretrained neural language model with broad capabilities.
- **Language Model**: A model that estimates probabilities over token sequences.
- **Learning Rate**: The step size used by an optimizer when updating parameters.
- **Learning Rate Decay**: Lowering the learning rate over training.
- **Learning Rate Schedule**: The rule governing how learning rate changes over time.
- **LoRA (Low-Rank Adaptation)**: A PEFT method that learns low-rank updates while freezing the base weights.
- **LoRA Rank**: The low-rank dimension used in LoRA adaptation.
- **Logits**: Raw, unnormalized model scores before softmax.
- **Layer Normalization**: A normalization method used to stabilize Transformer training.
- **Length Bias**: A tendency for evaluation or decoding to favor certain output lengths.

## K
- **KV Cache**: Stored key and value states reused during autoregressive decoding.
- **Knowledge Staleness**: Facts embedded in model weights becoming outdated over time.

## M
- **Masked Self-Attention**: Self-attention restricted to previous tokens only.
- **Mixed Precision**: Training with multiple numeric precisions to improve efficiency.
- **Model Parallelism**: Splitting model computation or weights across devices.
- **Multilingual Tokenization**: Tokenizer design that supports multiple languages well.
- **Model-Based Quality Filtering**: Using a model to judge and filter training data quality.
- **Model Parameters**: The trainable weights of a neural model.
- **Mixture of Experts (MoE)**: An architecture with multiple specialized sub-networks, typically routed per token.
- **Mixed Precision Training**: Training that combines lower and higher precision arithmetic.
- **MFU (Model FLOP Utilization)**: A measure of how efficiently hardware compute is used.

## N
- **N-gram Model**: A statistical model that predicts the next token from a fixed-size history.
- **Next-Token Prediction**: Predicting the next token from preceding tokens.
- **Normalization**: Text or activation standardization steps that improve consistency or training stability.

## O
- **Optimizer States**: Additional memory used by optimizers such as AdamW.
- **Overfitting**: Learning training-specific patterns that generalize poorly.
- **Overtraining**: Training beyond the point of efficient returns for a given compute budget.

## P
- **Parameter-Efficient Fine-Tuning (PEFT)**: Fine-tuning methods that update only a small subset of parameters.
- **Pairwise Comparison**: A judgment task where two responses are compared directly.
- **Pipeline Parallelism**: Splitting layers or stages across devices.
- **Positional Encoding**: A method for injecting token order information.
- **Preference Data**: Labeled comparisons showing which response is preferred.
- **Preference Optimization**: Post-training methods that align outputs with preferences.
- **Prefill**: The inference phase where the prompt is processed before decoding begins.
- **Pretraining**: Large-scale initial training on broad data.
- **Perplexity**: A metric for how well a language model predicts tokens.
- **Prompt**: The input text provided to a model at inference time.
- **Prompt Sensitivity**: Output instability caused by small prompt changes.

## Q
- **Quantization**: Reducing numerical precision to save memory and speed up inference.

## R
- **RAG (Retrieval-Augmented Generation)**: A system that retrieves external context and uses it in generation.
- **Reinforcement Learning**: Learning via reward-driven interaction.
- **RLHF (Reinforcement Learning from Human Feedback)**: Alignment training using human preference data and reinforcement learning.
- **Reward Hacking**: Exploiting flaws in a reward model to maximize score without real alignment.
- **Reward Model**: A model trained to score outputs according to human preferences.
- **Residual Connection**: A skip connection that adds a layer’s input to its output.
- **Rule-Based Filtering**: Filtering data with hand-crafted rules or heuristics.
- **Rotary Positional Embeddings (RoPE)**: A positional method that rotates query and key vectors by position.
- **Relative Positional Bias**: Encoding position using relative distance rather than absolute index.

## S
- **Sampling**: Probabilistic token selection during decoding.
- **Safety Tuning**: Post-training aimed at reducing harmful or unsafe behavior.
- **Scaling Laws**: Empirical relationships between model performance and scale.
- **Sequence Length**: The number of tokens in an input sequence.
- **Sequence Parallelism**: Splitting sequence-related work across devices.
- **SentencePiece**: A tokenization framework commonly used for subword modeling.
- **SFT (Supervised Fine-Tuning)**: Training on labeled input-output examples after pretraining.
- **Softmax**: A function that converts logits into probabilities.
- **Special Tokens**: Reserved tokens with specific roles such as padding or end-of-sequence.
- **Subword Tokenization**: Tokenization into units smaller than words but larger than characters.
- **Self-Attention**: The attention mechanism that relates tokens within the same sequence.
- **Sharding**: Splitting data into smaller pieces for efficient storage or training.
- **Sparse / Dense Decoder-Only Transformer**: A left-to-right Transformer used for autoregressive generation.

## T
- **Tensor Parallelism**: Splitting large tensor operations across devices.
- **Temperature**: A decoding parameter that controls randomness during sampling.
- **TF32**: A GPU numeric format that speeds up matrix operations while keeping FP32 range.
- **Token**: A discrete unit processed by the model.
- **Tokenization**: Converting text into tokens.
- **Token IDs**: Integer identifiers assigned to tokens.
- **Top-K Sampling**: Sampling only from the top `k` tokens by probability.
- **Top-P Sampling (Nucleus Sampling)**: Sampling from the smallest set of tokens whose cumulative probability exceeds `p`.
- **Transformer**: An attention-based neural architecture for sequence modeling.
- **Training Stability**: The ability of training to proceed without divergence or numerical failure.
- **Tokenizer**: The component that maps text to tokens and token IDs.
- **Tokenizer Training**: Learning the token vocabulary and tokenization rules.
- **Tokenizer Versioning**: Managing tokenizer changes across model versions.
- **Training Tokens**: The number of tokens used during pretraining.

## U
- **Undertrained Model**: A model that has too few training tokens relative to its capacity.
- **Unigram Language Model Tokenization**: A tokenization method that chooses segmentations probabilistically.
- **Unicode Normalization**: Text normalization that standardizes equivalent character forms.

## V
- **Vocabulary**: The set of all tokens recognized by a tokenizer.
- **Vocabulary Size**: The number of tokens in the vocabulary.

## W
- **Warmup**: An initial training phase where the learning rate gradually increases.
- **Word-level Tokenization**: Tokenization where each word is a token.
- **WordPiece**: A subword tokenization algorithm used in several language models.
- **Weak Calibration**: Confidence that does not reliably match correctness probability.

## Z
- **Zero-shot Learning**: Solving a task without examples in the prompt.
- **ZeRO (Zero Redundancy Optimizer)**: A memory optimization strategy that shards optimizer state, gradients, and sometimes parameters.
