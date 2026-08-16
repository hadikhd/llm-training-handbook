---
title: Home
slug: /
sidebar_position: 0
hide_table_of_contents: true
---

# LLM Training Handbook

**A practical, technically grounded guide to training Large Language Models.**

This handbook provides a structured path from the foundations of language modeling to the engineering realities of training, evaluating, and deploying modern LLMs. It covers the full lifecycle: data, tokenization, Transformer architecture, pretraining, scaling laws, post-training, evaluation, systems optimization, and inference.

The goal is not only to explain what these components are, but also **how they interact in real-world LLM development workflows**.

:::tip 🌍 Multilingual Support
This book is available in **English** and **Persian (فارسی)**. Use the language switcher in the top navigation bar to change languages.
:::

---

## 🎯 Who is this handbook for?

This handbook is written for:
- Machine learning engineers and deep learning practitioners
- NLP researchers and LLM engineers
- Applied AI teams
- Technical readers moving from classical ML into generative AI systems

It assumes basic familiarity with machine learning and neural networks, but aims to make the LLM training stack readable as a **connected engineering system** rather than a collection of isolated topics.

**Scope:** This handbook focuses on *training-oriented understanding*. It is primarily concerned with how LLMs are built, adapted, evaluated, and run in practice. It does not aim to be a lightweight popular overview, a purely mathematical monograph, or a product playbook detached from training mechanics.

---

## 📚 Table of Contents

The handbook covers the major technical pillars of LLM development, reflecting the actual pipeline of modern foundation model development:

### Part I: Foundations
- **[0. Preface](./00-preface.md)** — Why the handbook exists, who it is for, and how to read it.
- **[1. Introduction](./01-introduction.md)** — Conceptual foundations, statistical vs. neural models, and why scale matters.
- **[2. Data](./02-data.md)** — Web-scale collection, filtering, deduplication, decontamination, and the "data wall" problem.
- **[3. Tokenization](./03-tokenization.md)** — BPE, WordPiece, SentencePiece, multilingual challenges, and tokenization fertility.
- **[4. Transformer Architecture](./04-transformer-architecture.md)** — Self-attention, positional encodings, and decoder-only Transformers.

### Part II: Training
- **[5. Pretraining](./05-pretraining.md)** — Next-token prediction, optimization at scale, and training stability.
- **[6. Scaling Laws](./06-scaling-laws.md)** — Kaplan and Chinchilla laws, compute-optimal training, and practical implications.
- **[7. Post-Training](./07-post-training.md)** — SFT, RLHF, DPO, reward modeling, and alignment methods.

### Part III: Practice & Systems
- **[8. Evaluation](./08-evaluation.md)** — Benchmarks, human evaluation, LLM-as-a-Judge, and spurious correlations.
- **[9. Systems](./09-systems.md)** — GPU fundamentals, parallelism (Data, Tensor, Pipeline, ZeRO), and MFU.
- **[10. Inference and Decoding](./10-inference-and-decoding.md)** — KV cache, batching, quantization, and decoding strategies.
- **[11. Outlook](./11-outlook.md)** — MoE, State Space Models, multimodality, agents, and future directions.
- **[Glossary](./glossary.md)** — Key terms, symbols, and common acronyms.

---

## 🗺️ How to Read This Handbook

This handbook can be read in two ways:
1. **Sequentially**, from foundations to advanced systems topics.
2. **Selectively**, by jumping to the chapter most relevant to your current work.

**Recommended Reading Paths:**
- 🛠️ **For Implementation/Systems Engineers:** Move quickly into *Systems (9)*, *Post-Training (7)*, and *Inference (10)* after a brief review of the *Introduction (1)*.
- 🔬 **For Researchers/Theorists:** Spend more time in *Architecture (4)*, *Pretraining (5)*, and *Scaling Laws (6)*.
- 📊 **For Data Engineers:** Focus deeply on *Data (2)* and *Tokenization (3)*.

---

## 🔄 Future Revisions

This handbook is intended to evolve as the LLM landscape changes. Future revisions may expand sections on multimodality, agentic systems, long-context training, retrieval-augmented generation (RAG), alignment methods, and efficient serving stacks.
