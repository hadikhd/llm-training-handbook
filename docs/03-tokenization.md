---
id: tokenization
title: Tokenization
sidebar_label: Tokenization
sidebar_position: 4
description: How text becomes tokens and how tokenizer design affects efficiency, multilingual behavior, and training cost.
---
<div className="chapter-hero">

![Chapter 3 — Tokenization](/img/chapters/tokenization.png)

</div>

[Previous: Tokenization](./02-data.md) |
[Contents](./index.md) |
[Next: Transformer Architecture](./04-transformer-architecture.md)

## Learning Objectives

By the end of this chapter, you should be able to:

- Explain why neural networks require tokenization before processing text.
- Describe the relationship between raw text, tokens, token IDs, and embeddings.
- Distinguish between word-level, character-level, byte-level, and subword tokenization.
- Explain the core ideas behind BPE, WordPiece, Unigram Language Model tokenization, and SentencePiece.
- Analyze the trade-offs involved in choosing a vocabulary size.
- Explain how tokenization affects sequence length, context efficiency, memory usage, and training throughput.
- Identify language-specific tokenization challenges, especially for Persian and multilingual corpora.
- Design a practical tokenizer training pipeline for an LLM.
- Evaluate tokenizer quality using quantitative and qualitative metrics.
- Understand the role of special tokens, chat templates, normalization, and tokenizer versioning.
- Recognize common tokenization pitfalls that can degrade model quality or efficiency.

---

## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**


## 3.1 Why Tokenization Is Needed

Neural networks do not directly process raw strings. They operate on tensors of numbers.

Before a language model can process text, the text must be converted into a sequence of discrete symbols. These symbols are called **tokens**. Each token is mapped to an integer ID, and each ID is used to look up a vector from an embedding matrix.

A simplified flow is:
```text
Raw text
   |
   v
Tokenizer
   |
   v
Tokens
   |
   v
Token IDs
   |
   v
Embedding lookup
   |
   v
Dense vectors
   |
   v
Transformer
```
Example:

```text
Text:
Large language models

Tokens:
["Large", " language", " models"]

Token IDs:
[24576, 3303, 4211]
```

The exact tokens and IDs depend on the tokenizer.

Tokenization defines the model’s discrete interface with language. Once a model is trained with a tokenizer, that tokenizer becomes part of the model. Changing it after training is not a simple implementation detail; it changes the input representation itself.

---

## 3.2 Text as a Sequence Modeling Problem

Autoregressive LLMs are commonly trained to predict the next token.

Given a token sequence:

$$
x_1, x_2, \ldots, x_T
$$

the model estimates:

$$
P(x_t \mid x_{1:t-1})
$$

The training objective is usually next-token prediction:

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t \mid x_{1:t-1})
$$

This means tokenization determines what the model predicts at each step.

For example, the word `tokenization` may be represented as:

```text
["tokenization"]
```
or:

```text
["token", "ization"]
```
or:

```text
["t", "o", "k", "e", "n", "i", "z", "a", "t", "i", "o", "n"]

Each representation creates a different learning problem.
```

A tokenizer affects:

- Sequence length
- Prediction granularity
- Vocabulary size
- Embedding matrix size
- Effective context window
- Training compute
- Inference latency
- Multilingual performance
- Robustness to rare words
- Handling of spelling variation and noisy text

---

## 3.3 Tokens, Token IDs, and Embeddings

A tokenizer maps text to token IDs:

$$ 
\text{tokenizer}: \text{text} \rightarrow [i_1, i_2, \ldots, i_T]
$$

where each $i_t$ is an integer in:

$$
0 \le i_t < |V|
$$

and $|V|$ is the vocabulary size.

The embedding matrix is:

$$
E \in \mathbb{R}^{|V| \times d_{\text{model}}}
$$

Each token ID indexes one row of the embedding matrix:

$$
e_t = E[i_t]
$$

where:

- $$|V|$$ is the vocabulary size.
- $$d_{\text{model}}$$ is the hidden dimension.
- $$e_t$$ is the dense vector for token $$i_t$$.

If the vocabulary has 50,000 tokens and the model dimension is 4096, the input embedding matrix contains:

$$
50{,}000 \times 4096 = 204{,}800{,}000
$$

parameters.

If embeddings are stored in bf16 or fp16, this matrix alone requires approximately:

$$
204{,}800{,}000 \times 2 \approx 409.6 \text{ MB}
$$

This illustrates why vocabulary size is not free.

---

## 3.4 Tokenization Is a Modeling Choice

Tokenization is not merely preprocessing. It is a modeling decision that shapes the statistical task.

A tokenizer decides:

- Which units are frequent enough to become standalone tokens.
- How rare words are decomposed.
- How whitespace is represented.
- How punctuation is handled.
- Whether characters, bytes, or Unicode symbols are used.
- How different languages share vocabulary capacity.
- How special control symbols are represented.
- Whether normalization is performed before tokenization.

A poor tokenizer can damage model quality even if the Transformer architecture is strong.

For example, if a Persian sentence is split into too many small pieces, the model consumes more context length and compute to represent the same meaning. This is often called **token inflation**.

---

## 3.5 Word-Level Tokenization

In word-level tokenization, each word is treated as a token.

Example:

```text
Text:
Large language models are useful

Tokens:
["Large", "language", "models", "are", "useful"]
```

This approach is intuitive but has major problems.

### Advantages

- Easy to understand.
- Produces short sequences.
- Preserves word-level semantics.
- Works reasonably for small controlled vocabularies.

### Disadvantages

- Vocabulary becomes extremely large.
- Rare words require an unknown token.
- Misspellings and variants are poorly handled.
- Morphologically rich languages create many surface forms.
- New words cannot be represented unless they are added to the vocabulary.
- Multilingual corpora require very large vocabularies.

The unknown token problem is especially serious. If words outside the vocabulary are replaced by `<UNK>`, different words collapse into the same representation.

Example:

```text
Original:
The model supports backpropagation and quantization.
```

After unknown replacement:
The model supports <UNK> and <UNK>.

This loses important information.

For modern LLMs, pure word-level tokenization is rarely used.

---

## 3.6 Character-Level Tokenization

In character-level tokenization, each character is a token.

Example:

```text
Text:
model

Tokens:
["m", "o", "d", "e", "l"]
```

### Advantages

- Very small vocabulary.
- No unknown words.
- Robust to rare words and misspellings.
- Can represent any word composed of known characters.
- Useful for some low-resource or noisy settings.

### Disadvantages

- Sequences become much longer.
- The model must learn word formation from characters.
- Long-range dependencies become harder.
- Training and inference become more expensive.
- Context windows represent fewer words or sentences.

For example:

```text
Word-level:
["internationalization"]

Character-level:
["i", "n", "t", "e", "r", "n", "a", "t", "i", "o", "n", "a", "l", "i", "z", "a", "t", "i", "o", "n"]

The character-level sequence is much longer.
```
In a Transformer, attention cost grows approximately quadratically with sequence length:

$$
O(T^2)
$$

where $$T$$ is the sequence length. If tokenization doubles the number of tokens, attention computation can increase substantially.

Character-level tokenization is elegant but usually inefficient for large-scale LLMs.

---

## 3.7 Byte-Level Tokenization

Byte-level tokenization starts from byte representations rather than treating Unicode characters or words as the fundamental symbols.

UTF-8 encodes text as a sequence of bytes. A byte-level tokenizer begins from a vocabulary of 256 possible byte values and learns merges or patterns on top of them.

### Advantages

- Can represent any valid text.
- Avoids unknown characters.
- Handles multilingual and noisy data robustly.
- Does not require language-specific character coverage.
- Useful for web-scale corpora with mixed scripts and corrupted text.

### Disadvantages

- Some languages may require more bytes per character.
- Token sequences can become longer for non-Latin scripts.
- Raw bytes are less linguistically meaningful than characters or subwords.
- Without learned merges, byte-level sequences are inefficient.

Many modern tokenizers use byte-level or byte-fallback mechanisms to guarantee that every input can be represented.

---

## 3.8 Subword Tokenization

Subword tokenization is the standard approach for most modern LLMs.

It sits between word-level and character-level tokenization.

Frequent words may become single tokens:

```text
["language"]
```
Rare words may be decomposed:

```text
["token", "ization"]
```
Very rare or unknown strings can be decomposed further:

```text
["x", "q", "z", ...]

or into bytes.
```

### Why Subwords Work Well

Subword tokenization provides a practical balance:

- Vocabulary is much smaller than word-level vocabulary.
- Sequences are shorter than character-level sequences.
- Rare words can still be represented.
- Morphological patterns can be reused.
- Multilingual sharing becomes possible.
- No `<UNK>` is needed if byte fallback is available.

Example:

```text
unbelievable
= un + believable

tokenization
= token + ization

pretraining
= pre + training

Subword tokenization is especially important for:

- Rare technical terms
- Names
- Compound words
- Code identifiers
- URLs
- Misspellings
- Morphologically rich languages
- Multilingual corpora
```

---

## 3.9 Vocabulary

A tokenizer vocabulary is the set of all tokens that can be directly emitted.

Example vocabulary fragment:

```json
{
  "<PAD>": 0,
  "<BOS>": 1,
  "<EOS>": 2,
  " the": 3,
  " of": 4,
  "ing": 5,
  "model": 6,
  "language": 7
}
```

A vocabulary may include:

- Special tokens
- Whitespace-aware tokens
- Common words
- Subword units
- Characters
- Bytes
- Punctuation
- Digits
- Code fragments
- Language-specific units

The vocabulary is usually fixed before model pretraining starts.

---

## 3.10 Whitespace Handling

Whitespace is a critical part of tokenization.

Two common strategies are:

1. Treat whitespace as part of the following token.
2. Use explicit whitespace marker symbols.

For example, a tokenizer may represent:

```text
"Hello world"
```
as:

```text
["Hello", " world"]
```
where the leading space is part of the second token.

Another tokenizer may use a marker such as `▁`:

```text
["▁Hello", "▁world"]
```

This style is common in SentencePiece.

Whitespace handling matters because:

- It affects detokenization.
- It changes token boundaries.
- It affects code indentation.
- It impacts chat template formatting.
- It changes how prefixes and suffixes are learned.
- It can create subtle bugs if text is preprocessed inconsistently.

---

## 3.11 BPE: Byte Pair Encoding

Byte Pair Encoding, or BPE, is one of the most widely used subword tokenization algorithms.

Originally developed as a compression algorithm, BPE was adapted for neural machine translation and later became common in LLMs.

The core idea is simple:

> Start from small units and repeatedly merge the most frequent adjacent pair.

---

## 3.12 BPE Training Algorithm

A simplified BPE training process is:

1. Start with a vocabulary of basic symbols.
2. Split training text into sequences of these symbols.
3. Count all adjacent symbol pairs.
4. Find the most frequent pair.
5. Merge that pair into a new symbol.
6. Add the new symbol to the vocabulary.
7. Repeat until the target vocabulary size is reached.

Example corpus:

```text
low
lower
lowest
newer
wider

Initial character representation:

text
l o w
l o w e r
l o w e s t
n e w e r
w i d e r
```

If the pair `l o` is frequent, merge it:

```text
lo w
lo w e r
lo w e s t
n e w e r
w i d e r
```

If `lo w` is frequent, merge it:

```text
low
low e r
low e s t
n e w e r
w i d e r

Over many iterations, frequent patterns become tokens.
```

---

## 3.13 BPE Tokenization Algorithm

After BPE training, tokenization uses the learned merge rules.

Given an input string:

1. Convert it to the initial symbols.
2. Apply learned merges in training order.
3. Emit the final token sequence.
4. Map tokens to IDs.

Example:

```text
Input:
tokenization
```

Possible output:
["token", "ization"]

If `tokenization` itself was frequent in training, it may become:

```text
["tokenization"]
```

If it was rare, it may be decomposed more:

```text
["to", "ken", "ization"]
```

The output depends on the learned vocabulary and merge table.

---

## 3.14 Strengths and Weaknesses of BPE

### Strengths

- Simple and efficient.
- Works well in practice.
- Handles rare words through decomposition.
- Reduces sequence length compared to character-level tokenization.
- Can be implemented at large scale.
- Effective for many languages and code.

### Weaknesses

- Greedy merges may not always reflect linguistic structure.
- Frequent artifacts may become tokens.
- It may overfit to corpus-specific patterns.
- It can allocate too much vocabulary to dominant languages.
- It may split low-resource languages inefficiently.
- Normalization choices strongly affect results.

BPE is practical, but it is not linguistically perfect.

---

## 3.15 WordPiece

WordPiece is another subword tokenization method, historically used in models such as BERT.

Like BPE, WordPiece builds subword units, but its training criterion is different.

Instead of always merging the most frequent pair, WordPiece selects subword units using a likelihood-oriented criterion rather than simply choosing the most frequent adjacent pair.

A common WordPiece convention marks continuation subwords with a prefix such as `##`.

Example:

```text
unaffordable
= ["un", "##aff", "##ord", "##able"]
```

### BPE vs WordPiece

| Aspect | BPE | WordPiece |
|---|---|---|
| Merge criterion | Most frequent adjacent pair | Likelihood-based criterion |
| Common usage | GPT-style models, many LLMs | BERT-style models |
| Implementation | Simple | Slightly more complex |
| Output style | Depends on implementation | Often uses continuation markers |

WordPiece is important historically and practically, but many current generative LLMs prefer BPE-like or SentencePiece-based tokenizers.

---

## 3.16 Unigram Language Model Tokenization

The Unigram Language Model tokenizer starts with a large candidate vocabulary and removes tokens iteratively.

Instead of learning merges, it assumes a probabilistic model over possible segmentations.

For a text sequence $x$, multiple tokenizations may be possible:

```text
tokenization
= ["tokenization"]
= ["token", "ization"]
= ["to", "ken", "ization"]
```

The tokenizer selects the segmentation with high probability under the learned unigram token probabilities.

### Advantages

- Supports probabilistic segmentation.
- Can produce multiple possible tokenizations during training.
- Often works well in multilingual settings.
- Used by SentencePiece.

### Disadvantages

- More complex than BPE.
- Training and inference may require dynamic programming.
- Behavior is less intuitive than simple frequency-based merging.

---

## 3.17 SentencePiece

SentencePiece is a language-independent tokenizer framework that can train models such as BPE or the Unigram Language Model.

A key feature is that it treats the input as a raw Unicode string and does not require pre-tokenization by whitespace.

It commonly uses either:

- BPE
- Unigram Language Model

SentencePiece often represents spaces with the marker `▁`.

Example:

```text
Input:
Large language models

SentencePiece-style tokens:
["▁Large", "▁language", "▁models"]
```

### Why SentencePiece Is Useful

- Works without language-specific word segmentation.
- Useful for multilingual corpora.
- Avoids dependence on external pre-tokenizers.
- Handles languages where whitespace is not a reliable word boundary.
- Provides deterministic encoding and decoding.
- Supports normalization rules.

SentencePiece is widely used in multilingual models and many open-source LLMs.

---

## 3.18 Byte Fallback

Byte fallback allows a tokenizer to represent otherwise-uncovered input using byte tokens.

Without byte fallback, a tokenizer may require an `<UNK>` token for unseen characters.

With byte fallback:

text
Unknown character
   |
   v
UTF-8 bytes
   |
   v
Byte tokens

This guarantees coverage.

### Advantages

- Eliminates or reduces unknown-token failures.
- Handles emojis, rare symbols, and corrupted text.
- Improves robustness on web data.
- Helps multilingual and code-heavy corpora.

### Disadvantages

- Rare symbols may become multiple tokens.
- Some scripts may suffer from higher token counts.
- Byte sequences are less semantically meaningful.

For LLM pretraining, byte fallback is often preferred over relying on `<UNK>`.

---

## 3.19 Special Tokens

Special tokens are reserved tokens used to control model behavior or mark structure.

Common special tokens include:

| Token | Purpose |
|---|---|
| `<BOS>` | Beginning of sequence |
| `<EOS>` | End of sequence |
| `<PAD>` | Padding |
| `<UNK>` | Unknown token, if used |
| `<MASK>` | Masking token for masked language models |
| `<SEP>` | Separator |
| `<CLS>` | Classification token |
| `<SYSTEM>` | System message marker |
| `<USER>` | User message marker |
| `<ASSISTANT>` | Assistant message marker |
| `<TOOL>` | Tool-call marker |
| `<EOT>` | End of turn |

Not every model needs all of these.

For autoregressive LLM pretraining, common special tokens include:

- BOS
- EOS
- PAD
- Chat role tokens, if instruction tuning or chat formatting is planned
- Tool-use tokens, if tool calling is supported

Special tokens should be reserved before training the tokenizer and model.

---

## 3.20 BOS, EOS, and PAD

### BOS

`<BOS>` marks the beginning of a sequence.

Example:

```text
<BOS> The model learns from text.

Some models use BOS explicitly. Others do not.
```

### EOS

`<EOS>` marks the end of a document, sample, or turn.

Example:

```text
The model learns from text. <EOS>

EOS is especially important for generation. A model trained with EOS can learn when to stop.
```

### PAD

`<PAD>` is used to fill shorter sequences in a batch.

Example:

```text
[101, 233, 421, 2, 0, 0, 0]

where `0` may be the PAD token.
```

The loss should usually be masked on PAD positions:

$$
\mathcal{L}_{\text{pad}} = 0
$$

If PAD tokens contribute to loss incorrectly, the model may learn undesirable behavior.

---

## 3.21 Chat Templates and Control Tokens

Instruction-tuned and chat models require structured formatting.

A conversation may be represented as:

```text
<SYSTEM>
You are a helpful assistant.
<USER>
Explain tokenization.
<ASSISTANT>
Tokenization converts text into tokens.
<EOT>
```

The exact template is part of the model interface.

A tokenizer intended for chat models should reserve tokens for:

- System messages
- User messages
- Assistant messages
- Tool calls
- Tool results
- End-of-turn markers
- Function names or structured-call delimiters

Changing the chat template after training can significantly affect model behavior.

The tokenizer, chat template, training data formatter, inference server, and evaluation code must all agree.

---

## 3.22 Vocabulary Size Trade-Off

Vocabulary size is one of the most important tokenizer design choices.

Let:

- $$|V|$$ be vocabulary size.
- $$d_{\text{model}}$$ be hidden dimension.
- $$T$$ be sequence length.

Increasing vocabulary size usually reduces sequence length, but increases embedding and output projection size.

---

## 3.23 Small Vocabulary

A small vocabulary may contain, for example, 8k to 32k tokens.

### Advantages

- Smaller embedding matrix.
- Smaller output projection.
- Better sharing across related words.
- Lower memory cost for vocabulary-dependent parameters.
- Less risk of wasting tokens on rare artifacts.

### Disadvantages

- Longer token sequences.
- More compute in Transformer layers due to longer context.
- Higher token inflation for some languages.
- Poorer representation of frequent words if vocabulary is too small.
- More steps required to model long words or technical terms.

Example:

```text
internationalization

Small vocabulary:
["inter", "national", "ization"]

Very small vocabulary:
["i", "n", "ter", "n", "ation", "al", "iz", "ation"]
```
---

## 3.24 Large Vocabulary

A large vocabulary may contain 64k, 100k, 128k, or more tokens.

### Advantages

- Shorter token sequences.
- Better representation of frequent words and phrases.
- Lower token inflation for many languages.
- Potentially better throughput if sequence length reduction dominates.
- Improved handling of common domain-specific terms.

### Disadvantages

- Larger embedding matrix.
- Larger output layer.
- More memory use.
- More expensive logits computation.
- More vocabulary slots wasted on rare patterns.
- Possible sparsity: many tokens receive few training updates.
- Dominant languages may consume most of the vocabulary.

The output logits for next-token prediction are:

$$
z_t = h_t W_{\text{out}}
$$

where:

$$
W_{\text{out}} \in \mathbb{R}^{d_{\text{model}} \times |V|}
$$

A larger vocabulary increases the cost of computing logits and softmax.

---

## 3.25 Vocabulary Size and Model Parameters

If input and output embeddings are not tied, vocabulary-dependent parameters are approximately:

$$
2 \times |V| \times d_{\text{model}}
$$

If embeddings are tied, they are approximately:

$$
|V| \times d_{\text{model}} 
$$

Example with $$|V|=100{,}000$$ and $$d_{\text{model}}=4096$$:

Untied:

$$
2 \times 100{,}000 \times 4096 = 819{,}200{,}000
$$

parameters.

Tied:

$$
100{,}000 \times 4096 = 409{,}600{,}000
$$

parameters.

For smaller models, vocabulary parameters can become a large fraction of total parameters.

---

## 3.26 Tokenization and Context Efficiency

A model’s context length is measured in tokens, not words or characters.

If a model has a context length of 8192 tokens, the amount of text that fits depends on the tokenizer.

Example:

```text
English paragraph:
500 words -> 650 tokens

Persian paragraph:
500 words -> 950 tokens
```
In this example, Persian consumes more of the context window.

This affects:

- Long-document understanding
- Retrieval-augmented generation
- Summarization
- Chat history length
- Cost per request
- Training data balance
- Inference latency

A tokenizer with poor coverage for a language effectively gives that language a smaller usable context window.

---

## 3.27 Tokenization Fertility

Tokenization fertility measures how many tokens are produced per unit of text.

Common definitions include:

$$
\text{fertility}_{word} = \frac{ \text{number of tokens} }{ \text{number of words} }
$$

or:

$$
\text{fertility}_{char} = \frac{ \text{number of tokens} }{ \text{number of characters} }
$$

Lower fertility usually means more efficient tokenization.

Example:

| Language | Words | Tokens | Tokens/Word |
|---|---:|---:|---:|
| English | 1,000 | 1,250 | 1.25 |
| Persian | 1,000 | 1,900 | 1.90 |
| Code | 1,000 lexical units | 2,300 | 2.30 |

High fertility means:

- More training compute per word.
- Shorter effective context.
- Higher inference cost.
- Worse batching efficiency.
- Potential underrepresentation during training if mixture is token-based.

For multilingual LLMs, fertility should be measured per language.

---

## 3.28 Tokenization and Training Compute

Transformer compute depends strongly on token count.

For a fixed corpus measured in characters or words, a tokenizer that produces more tokens increases training cost.

Approximate attention cost per layer scales as:

$$ 
O(T^2 d)
$$

and feed-forward computation scales roughly as:

$$
O(T d^2)
$$

where:

- $$T$$ is sequence length.
- $$d$$ is hidden dimension.

Thus, token inflation increases both attention and non-attention computation.

Even when using optimized attention implementations, longer sequences still increase memory movement, activation storage, and training time.

---

## 3.29 Tokenization and Inference Cost

During generation, the model produces one token at a time.

If a response requires more tokens, inference cost increases.

A tokenizer affects:

- Prompt token count
- Generated token count
- KV-cache size
- Latency
- Billing cost
- Maximum usable conversation history

For autoregressive decoding, each new token extends the KV cache.

The KV-cache size is approximately proportional to:

$$
\text{batch size} \times \text{sequence length} \times \text{number of layers} \times \text{hidden size}
$$

Higher token counts directly increase memory usage during inference.

---

## 3.30 Tokenization and Loss Interpretation

Training and validation loss are measured per token.

If two tokenizers produce different token counts, their losses are not directly comparable.

For example:

```text
Tokenizer A:
Text -> 100 tokens
```

Tokenizer B:
Same text -> 150 tokens

A per-token loss comparison may be misleading because each token represents a different amount of information.

When comparing tokenizers, consider:

- Bits per byte
- Bits per character
- Downstream task performance
- Compression ratio
- Fertility
- Training throughput
- Inference cost

A tokenizer with lower per-token loss is not necessarily better if it produces many more tokens.

---

## 3.31 Tokenization and Multilingual Models

Multilingual tokenization is challenging because vocabulary capacity must be shared across languages.

A tokenizer trained on a corpus dominated by English may allocate many tokens to English words and few to Persian, Arabic, Hindi, Chinese, or other languages.

This causes:

- Higher token fertility for underrepresented languages.
- Poorer language modeling efficiency.
- Fewer complete words represented as tokens.
- More fragmented morphology.
- Reduced effective context length.
- Increased training and inference cost for those languages.

### Example

```text
English:
training -> ["training"]

Persian:
آموزش‌دادن -> ["آ", "موز", "ش", "‌", "داد", "ن"]
```

If Persian is poorly represented during tokenizer training, common Persian forms may be split unnecessarily.

---

## 3.32 Persian Tokenization Challenges

Persian tokenization has specific challenges that require careful design.

Important issues include:

- Zero-width non-joiner, or ZWNJ
- Arabic and Persian character variants
- Prefixes and suffixes
- Compound verbs
- Half-space usage
- Persian and Arabic digits
- Diacritics
- Bidirectional text
- Mixed Persian-English content
- Informal spelling variation
- Borrowed Arabic vocabulary
- OCR noise
- Social-media spelling variants

A Persian-aware tokenizer does not necessarily need hand-written linguistic segmentation, but the training corpus and normalization rules must reflect Persian text correctly.

---

## 3.33 Zero-Width Non-Joiner in Persian

The zero-width non-joiner, or ZWNJ, is commonly used in Persian orthography to prevent character joining while keeping morphemes visually connected.

Example:

```text
می‌شود
```

This may also appear inconsistently as:

```text
میشود
می شود
مي‌شود

These variants may be semantically equivalent or very close, but tokenization can treat them very differently.
```

Possible tokenizations:

```text
["می‌شود"]
["می", "‌", "شود"]
["می", "شود"]
["م", "ی", "ش", "و", "د"]
```

### Why ZWNJ Matters

ZWNJ affects:

- Prefix handling
- Suffix handling
- Compound words
- Vocabulary fragmentation
- Search and retrieval quality
- Model robustness
- Text normalization
- Detokenization quality

Removing every ZWNJ is not always correct. Preserving every inconsistent ZWNJ is also problematic.

A practical policy should be:

- Consistent
- Documented
- Tested on real Persian corpora
- Aligned with tokenizer training
- Aligned with downstream preprocessing

---

## 3.34 Arabic-Persian Character Normalization

Persian text often contains Arabic variants of Persian letters.

Common examples:

| Arabic Form | Persian Form | Description |
|---|---|---|
| `ي` | `ی` | Yeh |
| `ك` | `ک` | Kaf |
| `ة` | `ه` or context-dependent | Teh Marbuta |
| `ؤ` | context-dependent | Waw with Hamza |
| `إ`, `أ`, `آ` | context-dependent | Alef variants |

The most common normalization step for Persian is:

text
ي -> ی
ك -> ک

Without normalization, equivalent words may be split into separate vocabulary entries.

Example:

```text
كتاب
کتاب

These may look similar to users but are different Unicode sequences.
```

If both forms appear frequently, the tokenizer may waste vocabulary capacity by learning duplicate tokens.

---

## 3.35 Persian Morphology

Persian has productive affixes and compound constructions.

Examples:

```text
می‌روم
رفته‌ام
کتاب‌ها
خانه‌ام
دانش‌آموزان
برنامه‌نویسی
```

A useful tokenizer should learn reusable subword units such as:

```text
می
ها
ام
ان
نویسی
دانش
آموز

But it should not fragment common words too aggressively.
```

Good segmentation may help the model generalize across related forms:

```text
کتاب
کتاب‌ها
کتابم
کتاب‌هایشان
```

The tokenizer should represent morphological patterns efficiently while preserving common lexical units.

---

## 3.36 Informal and Noisy Persian

Persian web and social media text often includes informal spellings.

Examples:

```text
میشه
میشود
می‌شود
ميشه
خونه
خانه

There may also be:

- Repeated letters
- Latin transliteration
- Mixed Persian-English phrases
- Emoji
- Hashtags
- Arabic script variants
- Broken encodings
- OCR errors
```

A tokenizer intended for broad Persian coverage should be trained on diverse Persian text, not only formal written Persian.

However, noisy text should be filtered and balanced carefully. Too much noisy data can waste vocabulary capacity.

---

## 3.37 Code Tokenization

Code has different tokenization requirements from natural language.

Code contains:

- Identifiers
- Keywords
- Operators
- Indentation
- Brackets
- Strings
- Comments
- File paths
- URLs
- Numbers
- Naming conventions

Examples:

```python
def get_user_profile(user_id):
return database.fetch(user_id)
```

A tokenizer may split identifiers as:

```text
get_user_profile
= ["get", "_user", "_profile"]
```
or:

```text
get_user_profile
= ["get", "_", "user", "_", "profile"]
```

For code models, good tokenization should handle:

- snake_case
- camelCase
- PascalCase
- indentation
- common library names
- operators
- whitespace
- special characters

Bad code tokenization can significantly increase sequence length.

---

## 3.38 Numeric Tokenization

Numbers are challenging.

A tokenizer may represent:

```text
123456
```
as:

```text
["123456"]
```
or:

```text
["123", "456"]
```
or:

```text
["1", "2", "3", "4", "5", "6"]

Each choice has trade-offs.
```

### Whole-Number Tokens

Advantages:

- Shorter sequences for common numbers.

Disadvantages:

- Poor generalization to unseen numbers.
- Large vocabulary waste.
- Sparse training for rare numbers.

### Digit-Level Tokens

Advantages:

- Better compositional generalization.
- Handles arbitrary numbers.

Disadvantages:

- Longer sequences.
- Harder for the model to learn numeric magnitude.

Many tokenizers use a mixed strategy.

Numerical reasoning remains difficult for LLMs regardless of tokenization, but tokenization can make it easier or harder.

---

## 3.39 Tokenization of URLs, Emails, and Structured Text

Web data contains structured strings:

```text
https://example.com/path/to/file?id=123
user@example.com
2026-08-04

These strings may be tokenized into many pieces.
```

A tokenizer for web-scale data should handle:

- URLs
- email addresses
- dates
- file paths
- JSON
- XML
- Markdown
- LaTeX
- tables
- logs

Some of these strings may be useful, especially in code and technical corpora. Others may be noise.

Filtering and normalization decisions should be made before tokenizer training so that the tokenizer does not waste capacity on artifacts that will later be removed.

---

## 3.40 Unicode Normalization

Unicode allows multiple ways to represent visually similar text.

Common normalization forms include:

- NFC
- NFD
- NFKC
- NFKD

For tokenizer training, normalization must be carefully chosen.

### Benefits of Normalization

- Reduces duplicate representations.
- Improves vocabulary efficiency.
- Stabilizes tokenization.
- Helps multilingual consistency.

### Risks of Aggressive Normalization

- May destroy meaningful distinctions.
- Can damage mathematical notation.
- Can alter code.
- Can change names.
- Can remove script-specific information.
- Can break exact reproduction of text.

Normalization should be tested on:

- English
- Persian
- Arabic
- Code
- Mathematics
- URLs
- Mixed-language documents
- OCR outputs

---

## 3.41 Pre-Tokenization

Pre-tokenization splits text into coarse units before applying subword learning.

Examples of pre-tokenization rules:

- Split on whitespace.
- Split punctuation.
- Split digits.
- Separate Unicode categories.
- Preserve or isolate spaces.
- Split code symbols.

BPE and WordPiece implementations often use pre-tokenization.

SentencePiece can avoid external pre-tokenization and learn directly from raw text.

### Risks

Pre-tokenization can encode language-specific assumptions.

For example, whitespace-based pre-tokenization works reasonably for English and Persian, but not for all languages. It may also be problematic for code, URLs, or mixed-script text.

A multilingual tokenizer should minimize assumptions that disadvantage specific languages.

---

## 3.42 Training a Tokenizer

A practical tokenizer training pipeline looks like:

```text
Collect representative corpus
|
v
Clean and normalize text
|
v
Sample documents by language/domain
|
v
Reserve special tokens
|
v
Train tokenizer
|
v
Evaluate tokenizer fertility
|
v
Inspect tokenization examples
|
v
Adjust normalization/corpus/vocab size
|
v
Freeze tokenizer
|
v
Use for data tokenization and model training
```
The tokenizer should be trained on data that reflects the intended model usage.

For example, a bilingual English-Persian model should train its tokenizer on a balanced English-Persian tokenizer corpus, not only on the final raw token proportions. Otherwise, English may dominate vocabulary allocation.

---

## 3.43 Tokenizer Training Corpus

The tokenizer training corpus is usually a sample of the full pretraining corpus.

It should include:

- All target languages
- Major domains
- Formal and informal text
- Code, if needed
- Mathematical text, if needed
- Chat data, if needed
- Long documents and short documents
- Common structured formats
- Representative punctuation and whitespace

It should avoid overrepresenting:

- Duplicates
- Boilerplate
- Spam
- Corrupted text
- Repeated templates
- Benchmark text
- Low-value artifacts

The tokenizer does not need to see all training data, but it must see a representative sample.

---

## 3.44 Sampling for Multilingual Tokenizer Training

If tokenizer training data is sampled proportionally from a web corpus, high-resource languages may dominate.

A better approach is to use temperature sampling.

Let language $$i$$ have $$N_i$$ characters or documents. Define:

$$
p_i = \frac{N_i^\alpha} {\sum_j N_j^\alpha}
$$

where:

- $$\alpha = 1$$: proportional sampling.
- $$0 < \alpha < 1$$: boosts lower-resource languages.
- $$\alpha = 0$$: equal language sampling.

For tokenizer training, using $$\alpha < 1$$ can improve vocabulary allocation for lower-resource languages.

This is separate from the final pretraining data mixture.

---

## 3.45 Vocabulary Allocation in Multilingual Tokenizers

A multilingual tokenizer has finite vocabulary capacity.

If English receives too much capacity, other languages may be fragmented. If every language receives equal capacity, high-resource languages may become less efficient.

Vocabulary allocation should be evaluated empirically.

Useful questions include:

- How many tokens per word does each language need?
- Are common words split unnecessarily?
- Are affixes represented well?
- Are punctuation and whitespace handled correctly?
- Does one language dominate the learned vocabulary?
- Are rare noisy strings occupying many vocabulary slots?
- How does fertility change as vocabulary size changes?

A tokenizer should not be selected only by English performance.

---

## 3.46 Tokenizer Evaluation

Tokenizer quality should be evaluated before model training.

Useful evaluation dimensions include:

- Fertility by language
- Fertility by domain
- Character coverage
- Unknown-token rate
- Byte-fallback rate
- Compression ratio
- Average sequence length
- Distribution of token lengths
- Vocabulary usage frequency
- Common word fragmentation
- Rare word behavior
- Code fragmentation
- Numeric fragmentation
- Robustness to noisy text
- Detokenization correctness

No single metric is sufficient.

---

## 3.47 Unknown Token Rate

If a tokenizer uses `<UNK>`, measure:

$$
\text{UNK rate} = \frac{ \text{number of unknown tokens} }{ \text{total number of tokens} }
$$

For modern LLM tokenizers, the unknown rate should ideally be zero or extremely low.

Byte fallback can eliminate most unknown-token cases.

Unknown tokens are harmful because they collapse different inputs into the same representation.

Example:

```text
Input A:
quantization

Input B:
backpropagation

Bad tokenizer:
["<UNK>"]

Both become indistinguishable.
```
---

## 3.48 Byte-Fallback Rate

If the tokenizer uses byte fallback, measure how often byte tokens are used.

$$
\text{byte fallback rate} = \frac{ \text{number of byte tokens} }{ \text{total number of tokens} }
$$

High byte-fallback rates may indicate:

- Poor language coverage
- Bad Unicode normalization
- Too small vocabulary
- Excessive noisy text
- Unsupported symbols
- Broken encoding
- Emoji-heavy corpus

Byte fallback is useful, but frequent fallback may be inefficient.

---

## 3.49 Compression Ratio

Tokenizers can be evaluated as compressors.

A simple metric is:

$$
\text{characters per token} = \frac{ \text{number of characters} }{ \text{number of tokens} }
$$

Higher characters per token means more text is represented per token.

Another metric is bytes per token:

$$
\text{bytes per token} = \frac{ \text{number of UTF-8 bytes} }{ \text{number of tokens} }
$$

These metrics should be computed per language and domain.

---

## 3.50 Vocabulary Usage Distribution

After training a tokenizer, tokenize a large validation corpus and count token frequencies.

A healthy vocabulary should not have too many tokens that are almost never used.

Useful statistics:

- Number of unused tokens
- Number of tokens used fewer than 10 times
- Frequency of top 100 tokens
- Cumulative frequency of top 1,000 tokens
- Distribution by language/script
- Tokens dominated by noise or boilerplate

If many vocabulary entries are rare artifacts, retrain the tokenizer with better filtering or a smaller vocabulary.

---

## 3.51 Qualitative Tokenizer Inspection

Quantitative metrics are necessary but not enough.

Inspect examples manually.

For English:

```text
Tokenization should be efficient.
```
For Persian:

```text
مدل‌های زبانی بزرگ به داده‌های باکیفیت نیاز دارند.
```
For code:

```python
def calculate_attention_scores(query, key):
return query @ key.T
```

For noisy web text:

```text
loooool 😂 check this out: https://example.com/a/b?id=123
```
Check whether:

- Common words are split reasonably.
- Persian ZWNJ is handled consistently.
- English-Persian mixed text is stable.
- Code symbols are not excessively fragmented.
- URLs do not dominate vocabulary.
- Emojis fall back gracefully.
- Detokenization reconstructs the input correctly.

---

## 3.52 Detokenization

Detokenization converts token IDs back into text.

A tokenizer should satisfy:

```text
decode(encode(text)) ≈ text

Ideally, this should be exact for most text.
```

However, if normalization is applied during encoding, the decoded text may match the normalized form rather than the original raw text.

Example:

```text
Raw:
كتاب

Normalized:
کتاب

Decoded:
کتاب
```

This may be acceptable if the normalization policy is intentional.

Detokenization matters for:

- User-facing generation
- Evaluation
- Data debugging
- Reproducibility
- Chat formatting
- Code generation

A tokenizer that cannot reliably decode its own outputs can cause serious downstream issues.

---

## 3.53 Tokenizer and Data Pipeline Consistency

The tokenizer must be aligned with the data pipeline.

If the tokenizer was trained on normalized Persian text, then training and inference should use the same normalization.

Example inconsistency:

```text
Tokenizer training:
ي -> ی
ك -> ک

Inference:
No normalization
```

This can increase token fragmentation at inference time.

Similarly, if the tokenizer preserves ZWNJ during training but the inference pipeline removes it, the model receives a different distribution from what it saw during training.

All stages should share the same text policy:

```text
Raw text
   |
   v
Normalization
   |
   v
Tokenization
   |
   v
Model
```
---

## 3.54 Tokenizer Versioning

A tokenizer should be versioned like model code and datasets.

A tokenizer version should include:

- Vocabulary file
- Merge rules or model file
- Normalization configuration
- Pre-tokenization rules
- Special token IDs
- Chat template
- Training corpus description
- Training algorithm
- Vocabulary size
- Byte fallback settings
- Hash or fingerprint
- Creation date
- Evaluation report

Changing any of these may change model behavior.

A model checkpoint should record the exact tokenizer version used during training.

---

## 3.55 Tokenizer Fingerprinting

A tokenizer fingerprint can be computed from its files and configuration.

Example:

```text
tokenizer_fingerprint =
SHA256(
vocab.json
+ merges.txt
+ tokenizer_config.json
+ special_tokens_map.json
+ chat_template.txt
)
```

This helps ensure:

- Reproducibility
- Correct deployment
- Evaluation consistency
- Dataset regeneration consistency
- Debugging of model behavior

If two systems use different tokenizer files with the same model weights, results may be invalid.

---

## 3.56 Tokenizer Freezing

Once pretraining begins, the tokenizer is normally frozen.

Changing the tokenizer during training is difficult because:

- Token IDs change.
- Embedding rows no longer correspond to the same tokens.
- Tokenized datasets become invalid.
- Checkpoints become incompatible.
- Evaluation comparisons break.

In rare cases, vocabulary expansion is possible, but it requires careful embedding initialization and additional training.

For most LLM projects:

> Train, evaluate, and freeze the tokenizer before model pretraining.

---

## 3.57 Adding New Tokens After Training

Sometimes developers want to add new tokens after a model is trained.

Examples:

- Domain-specific terms
- Tool-use markers
- New chat tokens
- Programming language markers
- Special formatting symbols

This is possible but risky.

If a new token is added, the embedding matrix and output projection must be resized.

The new embedding may be initialized:

- Randomly
- From the average of existing embeddings
- From related subword embeddings
- Through additional training

Without sufficient fine-tuning, the model will not understand the new token.

Special tokens should therefore be planned before training whenever possible.

---

## 3.58 Tokenization for RAG Systems

In retrieval-augmented generation, tokenization affects both retrieval and generation.

It affects:

- Chunk size
- Context window usage
- Prompt construction
- Document packing
- Cost estimation
- Cross-language retrieval
- Passage truncation
- Embedding model input limits

For RAG, chunking should usually be based on the tokenizer used by the generation model or the embedding model, depending on the stage.

Example:

```text
Document chunk target:
512 tokens

But under which tokenizer?
- Embedding tokenizer?
- Generator tokenizer?
- Reranker tokenizer?
```

If these tokenizers differ, the same chunk may fit one model but overflow another.

For multilingual RAG, Persian token inflation can reduce how much retrieved context fits into the prompt.

---

## 3.59 Tokenization and Reranking

Rerankers often use separate encoders with their own tokenizers.

This can create mismatches:

```text
Retriever embedding model tokenizer:
Tokenizer A

Generator LLM tokenizer:
Tokenizer B

Reranker tokenizer:
Tokenizer C
```
A passage that is 400 tokens for the generator may be 600 tokens for the reranker.

Practical systems should track token counts for each model component separately.

For a production RAG pipeline, store metadata such as:

```json
{
  "chunk_id": "doc-42-chunk-7",
  "text": "...",
  "generator_tokens": 486,
  "embedding_tokens": 512,
  "reranker_tokens": 601
}
```

---

## 3.60 Tokenization and Sequence Packing

After tokenization, documents are often packed into fixed-length training sequences.

Example:

```text
[BOS] Document A [EOS] Document B [EOS] Document C [EOS]

Packing improves utilization by reducing padding.
```

However, packing requires decisions:

- Should documents attend to previous documents in the same packed sequence?
- Should position IDs reset between documents?
- Should EOS be inserted between documents?
- Should loss be applied to BOS, EOS, or PAD?
- Should chat turns be packed together or separated?

A tokenizer’s special tokens directly affect packing behavior.

---

## 3.61 Loss Masking and Special Tokens

Not every token should necessarily contribute to loss.

Common cases:

- PAD tokens should be masked.
- Some control tokens may be masked.
- Prompt tokens may be masked during supervised fine-tuning.
- Tool-result tokens may be masked depending on objective.
- User messages are often masked in assistant-response training.

Example SFT loss mask:

```text
<SYSTEM> You are helpful.
<USER> Explain BPE.
<ASSISTANT> BPE merges frequent pairs. <EOS>
```
Loss may be applied only to:

```text
BPE merges frequent pairs. <EOS>
```

This requires tokenizer and data formatting consistency.

---

## 3.62 Tokenization for Supervised Fine-Tuning

In supervised fine-tuning, formatting matters.

A sample may include:

```json
{
  "messages": [
{"role": "system", "content": "You are helpful."},
{"role": "user", "content": "Explain tokenization."},
{"role": "assistant", "content": "Tokenization converts text into tokens."}
  ]
}
```

The formatter converts it into text with role tokens.

If the formatter uses tokens not known to the tokenizer, they may be split into many pieces.

Bad example:

```text
<|assistant_response_starts_here|>
```

If not reserved as a special token, this may become many tokens.

Good practice:

- Reserve role tokens.
- Freeze the chat template.
- Mask losses correctly.
- Validate encoded examples.
- Test decoding.
- Keep formatting identical in training and inference.

---

## 3.63 Tokenization for Preference Training

Preference training methods such as reward modeling, DPO, or similar approaches compare chosen and rejected responses.

Tokenization matters because:

- Prompt and response boundaries must be clear.
- Loss masks must be correct.
- Chosen and rejected sequences must be truncated consistently.
- EOS handling must be stable.
- Role markers must match SFT formatting.

If truncation removes the answer but keeps the prompt, preference labels become meaningless.

A practical rule is:

> Always inspect tokenized and decoded preference samples before training.

---

## 3.64 Tokenizer Compatibility

Models and tokenizers are tightly coupled.

A model trained with tokenizer A cannot be used correctly with tokenizer B unless special conversion or adaptation is performed.

Even if both tokenizers have the same vocabulary size, token IDs may correspond to different tokens.

Example:

```text
Tokenizer A:
ID 1234 = " model"
```

Tokenizer B:
ID 1234 = " خانه"

Using the wrong tokenizer makes the model input nonsensical.

Always load the tokenizer distributed with the model checkpoint.

---

## 3.65 Tokenization and Evaluation

Evaluation prompts must be tokenized using the model’s tokenizer.

Tokenization affects:

- Multiple-choice formatting
- Stop sequences
- Prompt length
- Few-shot examples
- Answer extraction
- Exact-match scoring
- Log-probability scoring
- Per-token likelihood comparison

For log-probability evaluations, tokenization can change answer length.

Example:

```text
Answer A:
" yes" -> one token

Answer B:
" no" -> one token

But in another language:
" بله" -> maybe multiple tokens
" خیر" -> maybe one token
```
Comparisons should account for tokenization length where appropriate.

---

## 3.66 Practical Tokenizer Training Example

The following is a simplified example using a tokenizer library interface.

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, NFC
from tokenizers.processors import TemplateProcessing


special_tokens = [
"<PAD>",
"<BOS>",
"<EOS>",
"<UNK>",
"<SYSTEM>",
"<USER>",
"<ASSISTANT>",
"<EOT>"
]

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

tokenizer.normalizer = Sequence([
NFC()
])

tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=True)

trainer = BpeTrainer(
vocab_size=50000,
min_frequency=2,
special_tokens=special_tokens
)

files = [
"tokenizer_corpus/en_sample.txt",
"tokenizer_corpus/fa_sample.txt",
"tokenizer_corpus/code_sample.txt"
]

tokenizer.train(files, trainer)

tokenizer.post_processor = TemplateProcessing(
single="<BOS> $A <EOS>",
special_tokens=[
("<BOS>", tokenizer.token_to_id("<BOS>")),
("<EOS>", tokenizer.token_to_id("<EOS>")),
],
)

tokenizer.save("tokenizer.json")
```

This is a simplified illustration. A production tokenizer requires more careful normalization, multilingual sampling, evaluation, and versioning.

---

## 3.67 Persian Normalization Example

A minimal Persian normalization function may look like:

```python
import re
import unicodedata


def normalize_persian(text: str) -> str:
text = unicodedata.normalize("NFC", text)

replacements = {
"ي": "ی",
"ك": "ک",
"ۀ": "هٔ",
}

for src, dst in replacements.items():
text = text.replace(src, dst)

# Normalize whitespace except ZWNJ.
text = re.sub(r"[ \t\r\f\v]+", " ", text)
text = re.sub(r"\n+", "\n", text)

return text.strip()
```

This is not a complete Persian normalization policy. Production systems should add:

- ZWNJ tests
- Persian digit policy
- Arabic diacritic policy
- punctuation handling
- mixed-script examples
- OCR-specific cleanup
- evaluation on real corpora

---

## 3.68 Measuring Token Fertility

Example code:

```python
def fertility(tokenizer, texts):
total_words = 0
total_chars = 0
total_tokens = 0

for text in texts:
encoded = tokenizer.encode(text)
tokens = encoded.ids

total_tokens += len(tokens)
total_words += len(text.split())
total_chars += len(text)

return {
"tokens_per_word": total_tokens / max(total_words, 1),
"tokens_per_char": total_tokens / max(total_chars, 1),
"chars_per_token": total_chars / max(total_tokens, 1),
}
```

Run this separately for:

- English
- Persian
- Arabic
- Code
- Mathematics
- Chat data
- Noisy web data

Then compare tokenizers.

---

## 3.69 Tokenizer Evaluation Report Template

A tokenizer evaluation report may include:

```markdown
# Tokenizer Evaluation Report
```

## Tokenizer Metadata

- Name:
- Version:
- Algorithm:
- Vocabulary size:
- Special tokens:
- Normalization:
- Pre-tokenization:
- Byte fallback:
- Training corpus:
- Training date:
- Fingerprint:

## Quantitative Metrics

| Split | Tokens/Word | Chars/Token | UNK Rate | Byte Fallback Rate |
|---|---:|---:|---:|---:|
| English | | | | |
| Persian | | | | |
| Code | | | | |
| Math | | | | |

## Vocabulary Usage

- Unused tokens:
- Rare tokens:
- Top tokens:
- Suspicious tokens:

## Qualitative Examples

### English
Input:
Tokens:

### Persian
Input:
Tokens:

### Code
Input:
Tokens:

### Mixed Text
Input:
Tokens:

## Known Issues

## Final Decision

This report should be stored with the tokenizer release.

---

## 3.70 Choosing a Tokenizer for a New LLM

A practical decision process:

1. Define target languages and domains.
2. Estimate model size and context length.
3. Decide whether code is important.
4. Decide whether chat/tool tokens are needed.
5. Prepare a representative tokenizer training corpus.
6. Define normalization policy.
7. Train several candidate tokenizers.
8. Evaluate fertility and byte fallback by language.
9. Inspect qualitative examples.
10. Estimate embedding/output parameter cost.
11. Estimate training and inference token counts.
12. Select the tokenizer.
13. Freeze and fingerprint it.
14. Use it consistently across pretraining, fine-tuning, evaluation, and deployment.

Tokenizer selection should happen before full-scale data tokenization and training.

---

## 3.71 Practical Engineering Notes

- Treat the tokenizer as part of the model, not a separate utility.
- Train the tokenizer on representative data from all target languages and domains.
- Do not use an English-only tokenizer for a Persian or multilingual model unless you have measured the cost.
- Measure token fertility per language, not only globally.
- Use byte fallback or another robust mechanism to avoid unknown-token failures.
- Reserve all special tokens before training the model.
- Keep chat templates versioned with the tokenizer.
- Keep normalization consistent between tokenizer training, model training, evaluation, and inference.
- Inspect tokenized examples manually before large-scale training.
- Compute tokenizer fingerprints and store them with model checkpoints.
- Avoid overfitting the vocabulary to noisy web artifacts.
- Evaluate code, math, URLs, and structured text separately if they matter.
- Be careful when adding new tokens after training.
- Use the production tokenizer for data-mixture token counts.
- Validate detokenization with round-trip tests.
- Store token counts in dataset manifests.

---

## 3.72 Common Pitfalls

### Pitfall 1: Treating Tokenization as a Minor Preprocessing Step

Tokenization defines the input and output units of the model. It directly affects quality, cost, and context efficiency.

### Pitfall 2: Using an English-Centric Tokenizer for Persian

This often causes token inflation, poor morphology handling, and reduced effective context length.

### Pitfall 3: Ignoring ZWNJ

Persian ZWNJ handling must be consistent. Inconsistent policies fragment the vocabulary and reduce robustness.

### Pitfall 4: Forgetting Arabic-Persian Character Normalization

Characters such as `ي` and `ی`, or `ك` and `ک`, may look similar but are different Unicode characters.

### Pitfall 5: Choosing Vocabulary Size Without Measuring Fertility

A larger vocabulary is not always better. It reduces sequence length but increases embedding and softmax cost.

### Pitfall 6: Not Reserving Special Tokens Early

Adding chat or tool tokens after training is possible but often requires extra fine-tuning and careful initialization.

### Pitfall 7: Comparing Loss Across Tokenizers Directly

Per-token loss is not directly comparable when tokenizers produce different token counts.

### Pitfall 8: Training the Tokenizer on Noisy Raw Data

The tokenizer may waste vocabulary capacity on boilerplate, spam, broken encoding, and duplicated artifacts.

### Pitfall 9: Inconsistent Normalization Between Training and Inference

If inference text is normalized differently from training text, tokenization quality may degrade.

### Pitfall 10: Using the Wrong Tokenizer with a Model Checkpoint

Token IDs are tokenizer-specific. A model used with the wrong tokenizer will receive incorrect inputs.

### Pitfall 11: Ignoring Detokenization

A tokenizer must decode cleanly for generation, evaluation, and debugging.

### Pitfall 12: Not Versioning the Chat Template

For chat models, the template is part of the tokenizer-model interface.

---

## 3.73 Summary

- Tokenization converts raw text into discrete tokens that neural networks can process.
- Tokens are mapped to IDs, and IDs index rows in the embedding matrix.
- Tokenization determines the prediction units used in language modeling.
- Word-level tokenization is intuitive but suffers from huge vocabularies and unknown words.
- Character-level tokenization avoids unknowns but creates very long sequences.
- Byte-level tokenization guarantees coverage but may be inefficient without learned merges.
- Subword tokenization balances vocabulary size, coverage, and sequence length.
- BPE learns tokens by repeatedly merging frequent adjacent pairs.
- WordPiece uses a likelihood-oriented criterion and is common in BERT-style models.
- Unigram tokenization models possible segmentations probabilistically.
- SentencePiece provides language-independent tokenization and is widely used in multilingual models.
- Vocabulary size trades off sequence length against embedding and softmax cost.
- Tokenization affects context efficiency, training compute, inference latency, and validation loss interpretation.
- Persian tokenization requires careful handling of ZWNJ, Arabic-Persian character variants, morphology, and noisy informal writing.
- Multilingual tokenizers must be evaluated per language to avoid hidden inefficiencies.
- Special tokens and chat templates should be planned before model training.
- Tokenizer normalization must be consistent across tokenizer training, model training, evaluation, and inference.
- A tokenizer should be versioned, fingerprinted, evaluated, and frozen before large-scale pretraining.

---

## Review Questions

1. Why do neural networks require tokenization before processing text?
2. What is the relationship between tokens, token IDs, and embeddings?
3. Why is pure word-level tokenization unsuitable for modern LLMs?
4. What are the main advantages and disadvantages of character-level tokenization?
5. Why is byte fallback useful for web-scale multilingual data?
6. How does subword tokenization balance word-level and character-level approaches?
7. How does the BPE algorithm learn new tokens?
8. How does BPE tokenization use the learned merge rules?
9. What is the difference between BPE and WordPiece?
10. How does the Unigram Language Model tokenizer differ from BPE?
11. Why is SentencePiece useful for multilingual tokenization?
12. What are the trade-offs of increasing vocabulary size?
13. How does vocabulary size affect the embedding matrix and output projection?
14. Why does tokenization affect effective context length?
15. What is tokenization fertility, and why should it be measured per language?
16. Why can per-token loss be misleading when comparing tokenizers?
17. What makes Persian tokenization challenging?
18. Why is ZWNJ important in Persian text processing?
19. Why should Arabic `ي` and `ك` often be normalized in Persian corpora?
20. How can poor tokenizer training data waste vocabulary capacity?
21. What special tokens are commonly needed for chat models?
22. Why is a chat template part of the model interface?
23. Why should the tokenizer be frozen before pretraining?
24. What risks arise from adding new tokens after model training?
25. How does tokenization affect RAG chunking and context packing?

---

## Suggested Exercises

### Exercise 1: Compare Tokenization Strategies

Take the following sentence:

```text
Large language models require efficient tokenization.
```

Tokenize it manually using:

1. Word-level tokenization
2. Character-level tokenization
3. A hypothetical subword tokenizer

Compare sequence lengths and discuss trade-offs.

---

### Exercise 2: BPE by Hand

Use this small corpus:

```text
low
lower
lowest
new
newer
```

Start from characters and perform five BPE merge steps manually.

For each step, write:

- Pair frequencies
- Selected merge
- Updated corpus representation

---

### Exercise 3: Vocabulary Size Experiment

Train three tokenizers with vocabulary sizes:

```text
8k
32k
64k
```

Evaluate each on:

- English text
- Persian text
- Code
- Mixed Persian-English text

Report:

- Tokens per word
- Characters per token
- Unknown-token or byte-fallback rate
- Qualitative examples

---

### Exercise 4: Persian ZWNJ Audit

Collect examples such as:

```text
می‌شود
میشود
می شود
کتاب‌ها
کتاب ها
رفته‌ام
رفته ام
```

Tokenize them with your tokenizer.

Analyze:

- Whether equivalent forms tokenize similarly.
- Whether ZWNJ becomes a separate token.
- Whether common prefixes and suffixes are represented efficiently.
- Whether normalization improves consistency.

---

### Exercise 5: Arabic-Persian Normalization Test

Create pairs such as:

```text
كتاب / کتاب
يادگيري / یادگیری
كاربرد / کاربرد
```

Tokenize them before and after normalization.

Measure:

- Token count differences
- Token overlap
- Vocabulary fragmentation
- Detokenization behavior

---

### Exercise 6: Code Tokenization Analysis

Tokenize this snippet:

```python
def calculate_attention_scores(query_tensor, key_tensor):
return query_tensor @ key_tensor.T
```

Inspect how the tokenizer handles:

- snake_case identifiers
- underscores
- operators
- indentation
- punctuation
- Python keywords

---

### Exercise 7: Chat Template Validation

Define a chat template using:

```text
<SYSTEM>
<USER>
<ASSISTANT>
<EOT>
```

Encode and decode several examples.

Check:

- Whether role tokens are single tokens.
- Whether decoding preserves structure.
- Whether EOS/EOT handling is correct.
- Whether loss masks can be constructed unambiguously.

---

### Exercise 8: Tokenizer Evaluation Report

Create a tokenizer evaluation report with:

- Metadata
- Vocabulary size
- Special tokens
- Fertility by language
- Byte fallback rate
- Qualitative examples
- Known issues
- Final recommendation

Store it next to the tokenizer files.

---

## Further Reading

1. Sennrich, R., Haddow, B., and Birch, A.  
   *Neural Machine Translation of Rare Words with Subword Units*. ACL, 2016.

2. Kudo, T., and Richardson, J.  
   *SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*. EMNLP, 2018.

3. Schuster, M., and Nakajima, K.  
   *Japanese and Korean Voice Search*. ICASSP, 2012.  
   Introduces WordPiece-style subword modeling in a production context.

4. Kudo, T.  
   *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*. ACL, 2018.

5. Gage, P.  
   *A New Algorithm for Data Compression*. C Users Journal, 1994.  
   Original Byte Pair Encoding algorithm.

6. Radford, A. et al.  
   *Language Models are Unsupervised Multitask Learners*. OpenAI, 2019.  
   Discusses byte-level BPE in GPT-2.

7. Xue, L. et al.  
   *mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer*. NAACL, 2021.

8. Rust, P. et al.  
   *How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models*. ACL, 2021.

9. Petrov, A., and Katz, G.  
   *An Information-theoretic Analysis of Discrete Tokenizers*. 2023.

10. Hugging Face.  
*Tokenizers Documentation*.  
https://huggingface.co/docs/tokenizers

11. Google SentencePiece Repository.  
https://github.com/google/sentencepiece

---
[Previous: Tokenization](./02-data.md) |
[Contents](./index.md) |
[Next: Transformer Architecture](./04-transformer-architecture.md)
---