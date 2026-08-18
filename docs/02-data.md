---
id: Data for LLM Training
title: Data for LLM Training
sidebar_label: Data for LLM Training
sidebar_position: 3
description: How raw data becomes a governed, filtered, deduplicated, multilingual training corpus.
---
<div className="chapter-hero">

![Chapter 2 — Data](/static/img/chapters/data.png)

</div>

# Chapter 2: Data for LLM Training

[Previous: Introduction to Large Language Models](./01-introduction.md) |
[Contents](./index.md) |
[Next: Tokenization](./03-tokenization.md)

## Learning Objectives

By the end of this chapter, you should be able to:

- Explain why data quality, diversity, and scale jointly determine model behavior.
- Describe the complete LLM data lifecycle, from acquisition to training-ready shards.
- Distinguish document extraction, normalization, filtering, deduplication, and decontamination.
- Apply exact and approximate deduplication techniques, including MinHash and LSH.
- Explain how quality classifiers and heuristic filters are used in large-scale pipelines.
- Design and validate a mixture of multiple training datasets.
- Identify multilingual data-processing challenges, especially for Persian and English.
- Explain benchmark contamination and common decontamination strategies.
- Evaluate the opportunities and risks associated with synthetic training data.
- Define practical data-quality, governance, reproducibility, and monitoring requirements.

---

## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**

## 2.1 Why Data Is a First-Class Model Component

An LLM is shaped by more than its architecture and parameter count. Its training corpus determines much of what the model knows, which languages and domains it handles well, and which undesirable patterns it reproduces.

A useful abstraction is:

$$
\mathrm{Capability} = f(
\mathrm{architecture}, 
\mathrm{parameters}, 
\mathrm{data}, 
\mathrm{compute}, 
\mathrm{optimization}, 
\mathrm{post\text{-}training}
)
$$

The data distribution affects:

- Factual knowledge
- Vocabulary and writing style
- Multilingual capability
- Coding ability
- Domain expertise
- Reasoning patterns
- Social and cultural coverage
- Memorization behavior
- Safety characteristics
- Instruction-following potential

Increasing the number of tokens is not automatically beneficial. Repeated, corrupted, low-quality, or poorly balanced data may waste compute and reduce model quality.

For LLM training, data engineering is therefore part of model engineering.

---

## 2.2 Data Scale, Quality, and Diversity

A useful training corpus must balance three major properties.

### 2.2.1 Scale

Large datasets expose the model to many linguistic patterns, concepts, domains, and tasks.

Scale can improve:

- Coverage of rare concepts
- Generalization across domains
- Multilingual capability
- Robustness to different writing styles
- Long-tail knowledge

However, raw scale can introduce:

- More duplicates
- More spam
- More unsafe content
- More benchmark contamination
- More low-quality documents
- Higher storage and processing costs

### 2.2.2 Quality

Quality is multidimensional. It may include:

- Correctness
- Coherence
- Information density
- Linguistic fluency
- Structural completeness
- Source reliability
- Educational value
- Absence of spam or manipulation
- Suitability for the target model

There is no universal definition of a high-quality document. A code model, legal model, conversational model, and general-purpose model may require different quality criteria.

### 2.2.3 Diversity

A corpus should contain meaningful variation across:

- Languages
- Domains
- Document types
- Writing styles
- Difficulty levels
- Geographic and cultural contexts
- Time periods
- Formal and informal language
- Short and long documents

Diversity should not be confused with unfiltered randomness. The objective is broad and useful coverage, not indiscriminate inclusion.

---

## 2.3 The End-to-End Data Lifecycle

A production-scale LLM data pipeline commonly follows this sequence:
```text
Source discovery
|
v
Acquisition and licensing review
|
v
Raw data storage
|
v
Content extraction
|
v
Normalization
|
v
Language identification
|
v
Rule-based filtering
|
v
Model-based quality filtering
|
v
Safety and privacy processing
|
v
Exact deduplication
|
v
Near-duplicate detection
|
v
Benchmark decontamination
|
v
Dataset mixture construction
|
v
Tokenization and sequence packing
|
v
Sharding and training-time loading
|
v
Monitoring, auditing, and versioning
```
The exact order may vary. For example, cheap filters are usually applied before expensive model-based filters to reduce the volume that later stages must process.

A strong pipeline is:

- Scalable
- Deterministic where possible
- Auditable
- Versioned
- Fault tolerant
- Incrementally executable
- Measurable at every stage

---

## 2.4 Common Data Sources

LLM pretraining corpora may include several source categories.

### 2.4.1 Web Documents

Web crawls provide broad coverage, but they are extremely noisy.

Common problems include:

- Navigation menus
- Cookie banners
- Advertisements
- Search result pages
- Scraped or automatically generated pages
- Duplicated templates
- Link farms
- SEO spam
- Broken encoding
- Incomplete content
- Machine-translated pages
- Malicious or misleading text

Web data should be treated as raw material rather than training-ready text.

### 2.4.2 Books

Books may provide:

- Long-range coherence
- Narrative structure
- Edited prose
- Domain depth
- Rich vocabulary

Challenges include licensing, metadata quality, OCR errors, duplicated editions, and chapter-boundary extraction.

### 2.4.3 Scientific and Technical Literature

Scientific sources may improve:

- Technical vocabulary
- Mathematical writing
- Scientific reasoning
- Citation patterns
- Domain knowledge

Potential issues include:

- PDF extraction errors
- Broken equations
- Repeated headers and footers
- Incomplete tables
- Copyright restrictions
- Duplicate preprint and publication versions

### 2.4.4 Source Code

Code corpora can contain:

- Source files
- Documentation
- Tests
- Issue discussions
- Commit messages
- Notebooks

Important filtering dimensions include:

- Programming language
- Repository quality
- Generated code
- Dependency files
- Vendored libraries
- Secret leakage
- License compatibility
- Exact and near-duplicate code

### 2.4.5 Reference and Educational Material

Examples include:

- Encyclopedic content
- Tutorials
- Documentation
- Textbooks
- Frequently asked questions
- Knowledge bases

These sources can provide relatively dense and structured information.

### 2.4.6 Conversational Data

Conversational corpora may improve dialogue behavior, but raw conversations can include:

- Private information
- Low-effort responses
- Toxic exchanges
- Context loss
- Unclear speaker roles
- Repetitive patterns
- Strong platform-specific biases

Conversation structure and role metadata must be preserved carefully.

### 2.4.7 Licensed and Proprietary Data

Private or licensed datasets can add high-value domain coverage.

They require explicit controls for:

- Usage rights
- Retention policies
- Access control
- Privacy
- Geographic restrictions
- Derived model rights
- Auditability

---

## 2.5 Data Acquisition and Provenance

Every acquired item should be connected to provenance metadata.

A document record might contain:

```json
{
  "document_id": "doc_01H...",
  "source_name": "example-source",
  "source_uri": "https://example.org/document",
  "retrieved_at": "2026-08-04T08:30:00Z",
  "content_type": "text/html",
  "language": "en",
  "license": "source-specific",
  "raw_object_path": "raw/example-source/part-0001.jsonl.gz",
  "content_hash": "sha256:...",
  "pipeline_version": "data-pipeline-v2.3.0"
}
```
Useful provenance fields include:


- Source name
- Source URL or storage identifier
- Acquisition timestamp
- Original content type
- License or usage status
- Crawl snapshot or dataset version
- Content hash
- Processing-pipeline version
- Filter decisions
- Deduplication cluster identifier

Provenance enables:

- Error investigation
- Dataset removal
- Compliance review
- Reproducibility
- Source-level quality analysis
- Contamination tracing

If provenance is discarded early, it may be impossible to reconstruct later.

---

## 2.6 Raw Storage and Data Immutability

The raw dataset should normally be stored separately from processed outputs.

A practical storage model is:

```text
data/
├── raw/
│   ├── source-a/
│   └── source-b/
├── extracted/
├── normalized/
├── filtered/
├── deduplicated/
├── tokenized/
└── manifests/
```

Raw objects should preferably be immutable. Instead of modifying a raw file in place, create a new derived dataset version.

Benefits include:

- Reproducibility
- Easier debugging
- Safe pipeline reruns
- Comparison between processing versions
- Traceability of removed documents

At large scale, data is commonly stored in compressed shards using formats such as:

- JSON Lines
- Parquet
- Arrow
- WebDataset-style archives
- Custom indexed binary formats

The best format depends on access patterns, metadata requirements, compression, and training framework.

---

## 2.7 Content Extraction

Content extraction converts source-specific formats into usable text and structure.

### 2.7.1 HTML Extraction

An HTML page may contain much more than its main content.

A typical extractor should identify:

- Main article text
- Headings
- Lists
- Tables
- Code blocks
- Captions
- Metadata
- Document title

It should remove or separately label:

- Navigation
- Advertisements
- Related-content panels
- Cookie messages
- Repeated site templates
- Comments, when not needed

### 2.7.2 PDF Extraction

PDF is a presentation format rather than a semantic document format.

Common extraction failures include:

- Incorrect reading order
- Broken multi-column layouts
- Repeated headers and footers
- Lost mathematical symbols
- Table corruption
- Joined or split words
- Incorrect Unicode mapping
- Image-only pages

OCR may be required for scanned documents. OCR confidence and page-level quality should be retained as metadata.

### 2.7.3 Code and Notebook Extraction

Code extraction must preserve structure such as:

- File boundaries
- Language identifiers
- Comments
- Documentation strings
- Notebook cell order
- Markdown-code relationships

Blindly concatenating repository files can destroy useful context and create unrealistic training sequences.

### 2.7.4 Structural Preservation

Text alone is not always sufficient. Useful structural fields include:

```json
{
  "title": "Introduction to Optimization",
  "sections": [
{
"heading": "Gradient Descent",
"content": "..."
}
  ],
  "tables": [],
  "code_blocks": [],
  "references": []
}
```
Preserving structure makes later filtering, chunking, and domain-specific training more reliable.

---


## 2.8 Text Normalization

Normalization reduces irrelevant variation while preserving meaning.

Possible operations include:

- Unicode normalization
- Whitespace cleanup
- Line-break normalization
- Control-character removal
- Encoding repair
- Repeated punctuation handling
- Standardization of quotation marks
- Removal of null bytes
- Canonicalization of URLs or emails for selected tasks

Normalization should be conservative. Aggressive rewriting can damage:

- Source code
- Mathematical notation
- Poetry
- Tables
- Multilingual text
- Entity names
- Deliberate formatting

The normalized representation should not necessarily replace the original extracted text. Keeping both can support debugging and future pipeline improvements.

---

## 2.9 Multilingual Normalization

Multilingual pipelines require language-aware processing.

A transformation that is safe for English may damage Persian, Arabic, Chinese, or code.

### 2.9.1 Persian-Specific Considerations

Persian text may contain visually similar Arabic and Persian characters.

Examples include:

| Arabic form | Persian form | Description |
|---|---|---|
| `ي` | `ی` | Arabic and Persian Yeh |
| `ك` | `ک` | Arabic and Persian Kaf |

Other considerations include:

- Zero-width non-joiner, or ZWNJ
- Bidirectional text
- Persian and Latin digits
- Arabic diacritics
- Mixed Persian-English sentences
- Half-space usage
- Punctuation direction
- Incorrect character joining
- OCR-generated character substitutions

A normalization policy must be consistent with tokenizer training.

For example, removing every ZWNJ may merge words incorrectly, while preserving every inconsistent ZWNJ may increase vocabulary fragmentation.

### 2.9.2 Language-Aware Normalization Example

```python
def normalize_persian(text: str) -> str:
replacements = {
"ي": "ی",
"ك": "ک",
}

for source, target in replacements.items():
text = text.replace(source, target)

return normalize_whitespace(text)
```

This is only a minimal illustration. Production normalization requires:

- Test cases
- Corpus-level measurement
- Tokenization comparison
- Human linguistic review
- Versioned transformation rules

---

## 2.10 Language Identification

Language identification assigns one or more language labels to a document or segment.

A pipeline may use:

- Character-script rules
- N-gram classifiers
- Linear classifiers
- FastText-style models
- Transformer classifiers
- Ensemble methods

### 2.10.1 Document-Level vs Segment-Level Identification

Document-level classification assigns one language to the entire document.

This may fail for multilingual pages containing:

- Translations
- Code
- Citations
- Navigation
- Mixed-language discussions

Segment-level language identification is more precise but more expensive.

### 2.10.2 Confidence Thresholding

Instead of accepting the highest-scoring language unconditionally, use a confidence threshold:

$ \hat{\ell} = \arg\max_{\ell} P(\ell \mid x) $

Accept the prediction only if:

$ P(\hat{\ell} \mid x) \ge \tau $

Documents below the threshold may be:

- Removed
- Sent to a multilingual category
- Reclassified at segment level
- Manually inspected in samples

### 2.10.3 Language Identification Errors

Common errors occur with:

- Short text
- Named entities
- Code-heavy documents
- Closely related languages
- Transliteration
- Noisy OCR
- Mixed scripts
- Repetitive boilerplate

Language balance statistics should therefore be computed after filtering, not only before it.

---

## 2.11 Rule-Based Quality Filtering

Rule-based filters are inexpensive and interpretable.

Possible features include:

- Document length
- Average word length
- Character distribution
- Alphabetic-character ratio
- Punctuation ratio
- Repeated-line ratio
- Unique-token ratio
- Stop-word presence
- URL density
- Symbol density
- Number of headings
- Fraction of uppercase text
- Fraction of invalid Unicode
- Line-length distribution

A simple filter may look like:

```python
def passes_basic_filters(document: str) -> bool:
if len(document) < 200:
return False

if invalid_character_ratio(document) > 0.05:
return False

if repeated_line_ratio(document) > 0.30:
return False

if alphabetic_ratio(document) < 0.40:
return False

return True
```
Thresholds should be calibrated per language and source. Applying one global threshold may systematically remove useful content from certain scripts or document types.

---


## 2.12 Heuristic Filters and Their Risks

Heuristics can remove large amounts of obvious noise, but they can also introduce hidden bias.

For example:

- Minimum-length filters may remove valid short definitions.
- Stop-word filters may remove code or technical tables.
- Punctuation filters may reject mathematical content.
- Dictionary-based filters may reject dialects and rare names.
- ASCII-ratio filters may reject multilingual text.
- Profanity filters may remove educational or contextual discussions.

Each filter should be evaluated using:

- Precision: how much rejected content is truly undesirable?
- Recall: how much undesirable content is removed?
- Source-level retention rates
- Language-level retention rates
- Manual samples near the threshold

The objective is not to maximize the amount removed. It is to improve expected training value without destroying useful diversity.

---

## 2.13 Model-Based Quality Filtering

A quality classifier predicts whether a document resembles a target notion of useful text.

The classifier may use:

- Bag-of-words features
- Character or word n-grams
- Embeddings
- Small neural encoders
- Large language models
- Source and structural metadata

Let $x$ be a document. A classifier may produce:

$ q(x) = P(\text{high quality} \mid x) $

A hard filter retains the document if:

$ q(x) \ge \tau $

A soft approach converts quality into a sampling weight:

$ w(x) = g(q(x)) $

This allows lower-scoring documents to remain in the corpus at reduced frequency.

### 2.13.1 Constructing Quality Labels

Quality classifiers require positive and negative examples.

Possible positive sources include:

- Curated educational content
- Reference material
- Professionally edited text
- High-quality technical documentation

Possible negative sources include:

- Spam
- Scraped templates
- Keyword lists
- Random text fragments
- Repetitive machine-generated pages
- Corrupted extraction outputs

The classifier will learn the labeling policy, including its biases. If positive examples are too narrow, the model may reject useful but stylistically different content.

---

## 2.14 Perplexity-Based Filtering

A smaller language model can assign a perplexity score to documents.

Given a sequence $x_{1:T}$, average negative log-likelihood is:

$ L(x) = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_{<t}) $

Perplexity is:

$ \operatorname{PPL}(x) = \exp(L(x)) $

Extremely high perplexity may indicate:

- Corrupted text
- Unrecognized language
- Random symbols
- OCR errors
- Encoding problems

Extremely low perplexity may indicate:

- Repetition
- Template duplication
- Trivial boilerplate
- Memorized or formulaic content

Perplexity should not be interpreted as a universal quality score. Valuable technical, multilingual, or rare-domain text may naturally have high perplexity under the filtering model.

---

## 2.15 Safety, Privacy, and Sensitive Information

Data pipelines must account for potentially sensitive content.

Examples include:

- Personally identifiable information
- Authentication credentials
- Private communications
- Financial identifiers
- Medical records
- Exact addresses
- Phone numbers
- Email addresses
- API keys and access tokens

Detection techniques may include:

- Regular expressions
- Named-entity recognition
- Secret scanners
- Context-aware classifiers
- Source-level exclusion policies
- Human review for high-risk sources

Possible actions include:

- Redaction
- Replacement with typed placeholders
- Document removal
- Source exclusion
- Restricted storage
- Access logging

For example:

```text
Original:
Contact Jane at jane@example.com.


Redacted:
Contact [PERSON] at [EMAIL].
```
Redaction itself can create errors. A numerical expression may be mistaken for a phone number, and a public professional email may be treated differently from private contact information. Policies should therefore be risk-based and documented.

---

## 2.16 Exact Deduplication

Exact deduplication removes identical content.

A normalized document can be hashed:

$ h(x) = \operatorname{SHA256}(\operatorname{normalize}(x)) $

Documents with the same hash are duplicates.

Example:

```python
import hashlib

def document_hash(text: str) -> str:
normalized = normalize_for_deduplication(text)
return hashlib.sha256(
normalized.encode("utf-8")
).hexdigest()
```

Normalization for deduplication may include:

- Lowercasing where appropriate
- Whitespace normalization
- Line-ending normalization
- Removal of repeated boilerplate
- Unicode normalization

However, over-normalization may incorrectly merge distinct documents.

### 2.16.1 Deduplication Scope

Exact deduplication can operate at several levels:

- Entire document
- Paragraph
- Sentence
- Code file
- Repository
- Fixed-size text block

Document-level deduplication is simple, but does not remove documents that share most of their content with minor differences.

---

## 2.17 Why Near-Duplicate Detection Matters

Web documents are frequently copied and modified.

Examples include:

- Mirrors
- Reposted articles
- Print and mobile versions
- Templates with different navigation
- Slightly edited documents
- Repeated legal notices
- Multiple software repository forks
- Translations or machine-generated variants

A corpus may contain no exact duplicates while still having substantial semantic or lexical repetition.

Near-duplicate content can:

- Waste training compute
- Increase memorization
- Distort the data distribution
- Overrepresent popular sources
- Increase benchmark contamination risk
- Reduce effective corpus diversity

---

## 2.18 Shingling and Jaccard Similarity

A common near-duplicate method represents each document as a set of shingles.

A shingle is a contiguous sequence of tokens or characters.

For token trigrams:

```text
Document:
large language models learn patterns

Shingles:
("large", "language", "models")
("language", "models", "learn")
("models", "learn", "patterns")
```
Let $S(A)$ and $S(B)$ be the shingle sets of documents $A$ and $B$.

Their Jaccard similarity is:

$ J(A,B) = \frac{ |S(A)\cap S(B)| }{ |S(A)\cup S(B)| } $


A value close to 1 indicates strong overlap.

Directly comparing every pair of documents is impractical. For $N$ documents, all-pairs comparison requires approximately:

$ O(N^2) $

This motivates approximate methods such as MinHash and locality-sensitive hashing.

---

## 2.19 MinHash

MinHash creates a compact signature that approximates Jaccard similarity.

For a set of shingles $S$, define a hash function $h$. Its MinHash value is:

$ m_h(S) = \min_{s \in S} h(s) $

Using $k$ independent hash functions gives a signature:

$ M(S) = [ m_{h_1}(S), m_{h_2}(S), \ldots, m_{h_k}(S) ] $

A central MinHash property is:

$ P(m_h(A)=m_h(B)) = J(A,B) $

Therefore, the fraction of matching signature positions estimates Jaccard similarity:

$ \hat{J}(A,B) = \frac{1}{k} \sum_{i=1}^{k} \mathbf{1} [ M_i(A)=M_i(B) ] $

Larger signatures improve estimation accuracy but require more storage and computation.

---

## 2.20 Locality-Sensitive Hashing

Locality-Sensitive Hashing, or LSH, retrieves likely similar signatures without comparing every pair.

For MinHash LSH, a signature is divided into $b$ bands with $r$ rows per band:

$ k = b \times r $

Two documents become candidate duplicates if they match in every row of at least one band.

The approximate candidate probability is:

$ P(\text{candidate} \mid s) = 1-(1-s^r)^b $

where $s$ is the true Jaccard similarity.

Changing $b$ and $r$ changes the effective similarity threshold:

- More bands increase recall.
- More rows per band increase precision.
- Larger signatures improve stability but increase cost.

### Simplified Pipeline

```text
Document
   |
   v
Normalize
   |
   v
Create shingles
   |
   v
Compute MinHash signature
   |
   v
Insert signature into LSH buckets
   |
   v
Generate candidate pairs
   |
   v
Verify similarity
   |
   v
Build duplicate clusters
```

Candidate pairs should usually be verified using exact Jaccard similarity or another comparison method before removal.

---

## 2.21 Choosing a Representative from a Duplicate Cluster

Deduplication produces clusters, but the pipeline must decide which document to retain.

Possible ranking criteria include:

- Higher quality score
- Cleaner extraction
- More complete content
- Better provenance
- Preferred license status
- Original publication source
- More reliable metadata
- Earlier publication date
- Lower boilerplate ratio

For duplicate cluster $C$, select:

$ x^* = \arg\max_{x \in C} \operatorname{utility}(x) $

A utility function might combine several signals:

$ \operatorname{utility}(x) = \alpha q(x) + \beta c(x) + \gamma p(x) - \delta n(x) $

where:

- $q(x)$ is quality.
- $c(x)$ is completeness.
- $p(x)$ is provenance preference.
- $n(x)$ is noise.

Keeping an arbitrary first-seen document may retain a corrupted copy instead of the original.

---

## 2.22 Semantic Deduplication

Lexical methods may miss documents that convey the same information using different wording.

Semantic deduplication can use embeddings:

$ e_x = f_{\text{encoder}}(x) $

Similarity may be computed with cosine similarity:

$ \operatorname{sim}(x,y) = \frac{ e_x^\top e_y }{ \|e_x\|_2 \|e_y\|_2 } $

Candidate documents can be retrieved with an approximate nearest-neighbor index.

Semantic deduplication is useful for:

- Paraphrases
- Translated content
- Rewritten articles
- Synthetic variants
- Semantically repetitive instructions

However, it is more expensive and risks removing distinct documents on the same topic. It is often best applied selectively rather than globally.

---

## 2.23 Deduplication Leakage and Split Integrity

Deduplication should be performed across dataset splits, not only within each split.

Suppose training and validation contain near-identical versions of the same document. Validation loss will then underestimate generalization error.

A safer workflow is:

1. Construct duplicate clusters across the complete dataset.
2. Assign each cluster to exactly one split.
3. Prevent cluster members from crossing split boundaries.
4. Validate split overlap after assignment.

This principle applies to:

- Pretraining validation sets
- Fine-tuning datasets
- Preference datasets
- Domain benchmarks
- RAG evaluation corpora

---

## 2.24 Benchmark Contamination

Benchmark contamination occurs when evaluation content, close variants, or answer-bearing material appears in training data.

Contamination can occur through:

- Public benchmark repositories
- Tutorials reproducing benchmark questions
- Discussion forums containing answers
- Model-generated benchmark explanations
- Translated benchmark versions
- Dataset mirrors
- Benchmark documentation

A contaminated model may appear to reason correctly when it is retrieving memorized patterns.

---

## 2.25 Decontamination Methods

### 2.25.1 Exact Matching

Search for exact benchmark examples or normalized forms.

This method has high precision but misses paraphrases and partial overlap.

### 2.25.2 N-Gram Overlap

Represent benchmark examples and training documents as n-gram sets.

A document may be removed if overlap exceeds a threshold:

$ \operatorname{overlap}(d,b) \ge \tau $

This detects copied fragments but may create false positives for common phrases.

### 2.25.3 MinHash or LSH

Use approximate similarity to retrieve likely near-duplicates of benchmark examples.

### 2.25.4 Embedding Similarity

Embedding search can detect paraphrases and semantically related content.

This is powerful but may remove legitimate documents discussing the same topic without reproducing the benchmark.

### 2.25.5 Answer-Aware Matching

For multiple-choice or question-answer benchmarks, search for documents containing:

- The question
- Answer options
- The correct answer
- Characteristic explanations

This can provide stronger contamination evidence than question overlap alone.

### 2.25.6 Time-Based Controls

If reliable timestamps are available, evaluate on content created after the training cutoff.

This reduces direct contamination risk but does not eliminate template or task-family overlap.

---

## 2.26 Decontamination Trade-offs

Aggressive decontamination can remove legitimate educational material.

For example, a mathematics benchmark may contain standard problems that also appear in textbooks. Removing every related document may reduce the model’s mathematical training data.

It is useful to distinguish:

- **Exact contamination:** The same example is present.
- **Near contamination:** A lightly modified version is present.
- **Task exposure:** Similar task formats are present.
- **Domain exposure:** The underlying knowledge is present.

Task and domain exposure are usually expected. Exact answer memorization is the primary concern.

Decontamination decisions and thresholds should be reported with evaluation results.

---

## 2.27 Dataset Mixture Design

LLMs are usually trained on a mixture of datasets.

Let dataset $D_i$ contain $N_i$ available tokens. A simple sampling probability is:

$ p_i = \frac{N_i}{\sum_j N_j} $

This proportional strategy can cause large web datasets to dominate higher-quality or lower-resource sources.

A temperature-based mixture can be defined as:

$ p_i = \frac{N_i^\alpha} {\sum_j N_j^\alpha} $

where:

- $\alpha = 1$ gives proportional sampling.
- $0 < \alpha < 1$ increases the relative weight of smaller datasets.
- $\alpha = 0$ gives equal probability to every dataset.

For multilingual training, the same method can be applied to language-level token counts.

---

## 2.28 Upsampling and Downsampling

### Upsampling

Upsampling increases the frequency of a dataset or domain.

It can improve low-resource coverage, but excessive upsampling may cause:

- Memorization
- Repetition
- Overfitting
- Reduced generality

### Downsampling

Downsampling reduces the influence of a dominant dataset.

It can improve balance, but may discard useful diversity.

### Effective Epochs

For dataset $D_i$, define:

$ \text{effective epochs}_i = \frac{ \text{tokens sampled from }D_i }{ N_i } $

Tracking effective epochs helps identify small datasets that are repeated too many times.

---

## 2.29 Dynamic Data Mixtures

A data mixture does not have to remain constant throughout training.

A curriculum may:

- Begin with broad general data.
- Increase high-quality data later.
- Add long-context examples in a later stage.
- Increase code or reasoning data near the end.
- Change language weights over time.

Let $p_i(t)$ denote the sampling probability for dataset $i$ at training step $t$.

A schedule may define:

$ p_i(t) = (1-\lambda_t)p_i^{\text{initial}} + \lambda_t p_i^{\text{final}} $

where $ lambda_t$ changes from 0 to 1 during training.

Dynamic mixtures add operational complexity. Every schedule change must be logged to preserve reproducibility.

---

## 2.30 Domain and Language Balance

A dataset can appear diverse at the source level but remain unbalanced at the token level.

For example:

```text
Source count:
English documents   55%
Persian documents   45%
```
```text
Token count:
English tokens      78%
Persian tokens      22%
```
Differences may arise from:

- Document length
- Tokenizer efficiency
- Filtering retention
- Repetition
- Source composition

Balance should be measured using:

- Raw document count
- Character count
- Word count
- Token count
- Unique document count
- Deduplicated token count
- Training-time sampled token count

Training-time sampled tokens are the final measure of what the model actually sees.

---

## 2.31 Synthetic Data

Synthetic data is generated or transformed by models rather than collected directly from human-created sources.

Common uses include:

- Instruction generation
- Question-answer generation
- Reasoning traces
- Code exercises
- Translation
- Data augmentation
- Error correction
- Style transformation
- Domain-specific examples
- Preference pairs

A synthetic data pipeline may look like:

```text
Seed documents
|
v
Prompt construction
|
v
Teacher model generation
|
v
Rule-based validation
|
v
Model-based validation
|
v
Deduplication
|
v
Difficulty and diversity balancing
|
v
Training dataset

---
```

## 2.32 Risks of Synthetic Data

Synthetic data can introduce:

- Teacher-model bias
- Repeated response structures
- Incorrect facts
- Artificially simple tasks
- Excessive verbosity
- Loss of linguistic diversity
- Hidden benchmark contamination
- Self-reinforcing model errors
- Style homogenization

If synthetic data is repeatedly generated from model outputs and used without strong validation, dataset diversity may collapse.

### Mitigation Strategies

- Use diverse seed sources.
- Use multiple teacher models.
- Vary prompts and generation parameters.
- Verify answers using tools or deterministic checks.
- Remove near-duplicate generations.
- Measure lexical and semantic diversity.
- Mix synthetic and human-created data.
- Retain provenance linking outputs to generation configurations.
- Evaluate downstream effects with controlled ablations.

---

## 2.33 Data Validation and Quality Assurance

Every pipeline stage should produce measurable reports.

### Source-Level Metrics

- Number of documents acquired
- Total bytes
- Extraction success rate
- Average document length
- Missing metadata rate

### Filtering Metrics

- Retention rate
- Rejection reasons
- Quality-score distribution
- Retention by source
- Retention by language
- Retention by document type

### Deduplication Metrics

- Exact duplicate rate
- Near-duplicate rate
- Number of duplicate clusters
- Cluster-size distribution
- Tokens removed
- Representative-selection statistics

### Safety and Privacy Metrics

- Detected PII categories
- Redaction rates
- Secret detection counts
- Source-level exclusion counts
- Manual-review outcomes

### Final Corpus Metrics

- Documents
- Characters
- Tokens
- Languages
- Domains
- Source distribution
- Sequence-length distribution
- Effective epochs
- Validation overlap

---

## 2.34 Sampling-Based Human Inspection

Automated metrics are not sufficient.

Human reviewers should inspect:

- Random retained documents
- Random rejected documents
- Documents close to filtering thresholds
- Large duplicate clusters
- Low-resource languages
- High-risk sources
- Documents with extreme perplexity
- Synthetic examples
- Benchmark-overlap candidates

A stratified review design is better than purely random sampling.

For example:

```text
20% random retained documents
20% random rejected documents
20% near the quality threshold
15% low-resource languages
15% large duplicate clusters
10% safety-sensitive documents
```

Human findings should feed back into rules, models, and thresholds.

---

## 2.35 Tokenization and Token Counting

Token counts depend on the final tokenizer.

The same document may contain different numbers of tokens under different tokenizers:

$ T_A(x) \neq T_B(x) $

Tokenization affects:

- Training compute
- Sequence packing
- Language balance
- Effective context length
- Storage size
- Data-mixture weights

For multilingual datasets, fertility is a useful metric:

$ \operatorname{fertility} = \frac{ \text{number of tokens} }{ \text{number of words or characters} } $

A high tokenization fertility for one language means that the language consumes more sequence positions and training compute for the same amount of text.

Corpus planning should therefore use token counts from the actual production tokenizer.

---

## 2.36 Sequence Construction

After tokenization, documents must be converted into fixed or variable-length training sequences.

### 2.36.1 Truncation

Long documents may be divided into multiple sequences.

Care is needed to avoid:

- Losing section boundaries
- Cutting code in invalid locations
- Separating questions from answers
- Dropping important metadata
- Creating excessive overlap

### 2.36.2 Padding

Short sequences can be padded, but padding wastes compute.

If sequence length is $L$ and a sample contains only $l$ useful tokens, its utilization is:

$ u = \frac{l}{L} $

### 2.36.3 Sequence Packing

Packing combines multiple short documents into one training sequence.

```text
[BOS] Document A [EOS] Document B [EOS] Document C [EOS]

Packing improves token utilization, but the attention and loss policy must be defined.
```

Possible strategies include:

- Allow cross-document attention.
- Block cross-document attention.
- Reset position identifiers.
- Preserve continuous positions.
- Mask selected separator tokens from the loss.

The correct strategy depends on the model architecture and training implementation.

---

## 2.37 Shuffling and Data Ordering

Training data should usually be shuffled sufficiently to avoid long runs from one source or domain.

Poor ordering can cause:

- Temporary overfitting
- Optimization instability
- Source-specific loss spikes
- Correlated batches
- Reduced mixture fidelity

At large scale, perfect global shuffling may be expensive. Practical systems often use:

- Shard-level randomization
- Buffered sample shuffling
- Deterministic seeds
- Per-worker shard assignment
- Epoch-specific permutations

The data loader must avoid accidentally sending identical shards to multiple workers unless repetition is intentional.

---

## 2.38 Dataset Sharding

Tokenized data is commonly divided into shards.

Good sharding supports:

- Parallel reading
- Fault recovery
- Distributed worker assignment
- Efficient randomization
- Incremental regeneration
- Integrity checks

A shard manifest might contain:

```json
{
  "shard_id": "train-000042",
  "path": "tokenized/train-000042.bin",
  "num_sequences": 16384,
  "num_tokens": 33554432,
  "checksum": "sha256:...",
  "mixture_component": "technical-en",
  "tokenizer_version": "tokenizer-v1.2"
}
```

Shards should have checksums so corruption can be detected before or during training.

---

## 2.39 Streaming vs Materialized Datasets

### Materialized Dataset

All processed and tokenized data is written before training.

Advantages:

- Deterministic
- Easy to inspect
- Fast at training time
- Stable token counts

Disadvantages:

- High storage cost
- Slow iteration after pipeline changes
- Duplicate intermediate representations

### Streaming Dataset

Data is fetched or processed while training.

Advantages:

- Lower storage requirements
- Easier access to very large sources
- Potentially more flexible mixtures

Disadvantages:

- Harder reproducibility
- Variable latency
- More runtime failure modes
- Difficult exact accounting
- Potential source drift

Hybrid systems may materialize high-value data while streaming larger or frequently updated sources.

---

## 2.40 Data Loaders and Distributed Training

A distributed data loader must ensure that:

- Workers receive intended samples.
- Sampling probabilities match the configured mixture.
- Shards are not unintentionally duplicated.
- Checkpoint recovery restores the correct data position.
- Random seeds are managed deterministically.
- Failed workers do not silently alter the data distribution.

Data-loader state may include:

```json
{
  "global_step": 125000,
  "epoch": 1,
  "shard_permutation_seed": 78123,
  "current_shard": 42,
  "sample_offset": 8192,
  "mixture_schedule_version": "mix-v4"
}

Saving only model and optimizer state may be insufficient for exact training recovery.

---
```

## 2.41 Data Governance

Data governance defines how datasets are approved, used, modified, removed, and audited.

A governance process should answer:

- Where did the data originate?
- Under what terms may it be used?
- Which pipeline transformed it?
- Which model checkpoints consumed it?
- Can a source be removed later?
- Who approved sensitive datasets?
- Which evaluations test potential leakage?
- How are incidents documented?

A dataset card may include:

- Dataset purpose
- Source categories
- Languages
- Collection dates
- Processing stages
- Filtering policies
- Known limitations
- Privacy considerations
- Licensing status
- Intended and prohibited uses

---

## 2.42 Dataset Versioning and Reproducibility

A complete dataset version should identify:

- Source snapshots
- Acquisition code
- Extraction version
- Normalization rules
- Filtering models and thresholds
- Deduplication parameters
- Decontamination benchmark versions
- Mixture weights
- Tokenizer version
- Sequence-packing configuration
- Shard manifests
- Random seeds

A dataset fingerprint can be built from configuration files and shard checksums:

$ F = H( \text{configuration} \Vert \text{manifest} \Vert \text{checksums} ) $

This fingerprint can be stored with model checkpoints.

---

## 2.43 Data Ablation Studies

Ablation studies estimate the effect of individual data choices.

Examples include:

- With vs without near-duplicate removal
- Rule-based filtering vs learned quality filtering
- Different language mixture weights
- Different synthetic-data ratios
- Different benchmark decontamination thresholds
- Different quality-score cutoffs
- Different sequence-packing policies

A controlled ablation should keep other variables as stable as possible.

Useful outcomes include:

- Validation loss
- Domain benchmarks
- Memorization metrics
- Multilingual quality
- Safety metrics
- Training throughput
- Downstream instruction-following performance

Data decisions should be validated empirically rather than based only on intuition.

---

## 2.44 Data Quality as an Optimization Problem

Data selection can be viewed as constrained optimization.

Let $S$ be the selected corpus and $U(S)$ its expected training utility.

The objective is:

$ \max_S U(S) $

subject to constraints such as:

$ \operatorname{tokens}(S) \le B $

$ \operatorname{risk}(S) \le R $

$ \operatorname{cost}(S) \le C $

where:

- $B$ is the token budget.
- $R$ is the acceptable risk threshold.
- $C$ is the processing or licensing budget.

This framing explains why “keep everything” is rarely optimal. Training compute is finite, so low-value repetition has an opportunity cost.

---

## 2.45 Practical Example: A Simplified Processing Pipeline

```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
document_id: str
source: str
text: str
language: Optional[str] = None
quality_score: Optional[float] = None
content_hash: Optional[str] = None


def process_document(document: Document) -> Optional[Document]:
text = extract_main_content(document.text)
text = normalize_unicode(text)
text = normalize_whitespace(text)

language, confidence = identify_language(text)

if confidence < 0.80:
return None

if not passes_language_specific_rules(text, language):
return None

quality_score = predict_quality(text, language)

if quality_score < 0.55:
return None

text = redact_sensitive_information(text)

document.text = text
document.language = language
document.quality_score = quality_score
document.content_hash = compute_content_hash(text)

return document
```

A real implementation also needs:

- Batch processing
- Error handling
- Retry policies
- Distributed execution
- Metrics
- Logging
- Model version tracking
- Provenance preservation
- Exact and near deduplication
- Compliance controls

---

## 2.46 Recommended Pipeline Architecture

A modular design separates each transformation:

```text
DataSource
   |
   v
Extractor
   |
   v
Normalizer
   |
   v
LanguageIdentifier
   |
   v
RuleFilter
   |
   v
QualityScorer
   |
   v
PrivacyProcessor
   |
   v
ExactDeduplicator
   |
   v
NearDeduplicator
   |
   v
Decontaminator
   |
   v
MixtureBuilder
   |
   v
Tokenizer
   |
   v
SequencePacker
   |
   v
ShardWriter
```

Each stage should accept and emit a documented schema.

Example output record:

```json
{
  "document_id": "doc-123",
  "text": "Normalized document text...",
  "language": {
"label": "en",
"confidence": 0.98
  },
  "quality": {
"score": 0.87,
"model_version": "quality-v3"
  },
  "deduplication": {
"exact_hash": "sha256:...",
"cluster_id": "cluster-456"
  },
  "provenance": {
"source": "source-a",
"retrieved_at": "2026-08-04T08:30:00Z"
  },
  "pipeline_version": "pipeline-v2.3"
}

---
```

## 2.47 Practical Engineering Notes

- Preserve source provenance before applying irreversible transformations.
- Keep raw, extracted, normalized, and final data logically separate.
- Apply cheap filters before expensive classifiers and embedding models.
- Evaluate all filters by language, source, and document type.
- Perform deduplication before final train-validation split assignment.
- Select the best representative from each duplicate cluster instead of keeping an arbitrary copy.
- Calculate mixture weights using production-tokenizer token counts.
- Track effective epochs for every mixture component.
- Validate the actual training-time distribution produced by the data loader.
- Store filtering decisions and rejection reasons for auditability.
- Version decontamination benchmarks independently from training corpora.
- Treat synthetic data as untrusted until it passes validation.
- Save data-loader state with distributed training checkpoints.
- Use checksums for raw objects, processed shards, and manifests.
- Run small training ablations before applying major filtering policies at full scale.

---

## 2.48 Common Pitfalls

### Pitfall 1: Treating Raw Web Data as Training-Ready

Web data contains extensive noise, duplication, and structural artifacts.

### Pitfall 2: Optimizing Only for Token Count

More tokens can be harmful when they are repetitive, corrupted, or poorly balanced.

### Pitfall 3: Using One Filter for Every Language

A threshold calibrated for English may reject useful Persian, Arabic, code, or mathematical content.

### Pitfall 4: Performing Only Exact Deduplication

Exact hashes do not detect mirrors, lightly edited copies, or repeated templates.

### Pitfall 5: Deduplicating Splits Independently

Near-duplicates may leak between training and validation sets.

### Pitfall 6: Using Quality Models Without Studying Their Biases

A quality classifier may mistake stylistic similarity to its positive examples for actual quality.

### Pitfall 7: Ignoring Tokenizer Effects

Document and word counts do not represent the actual training compute consumed by each language.

### Pitfall 8: Upsampling Low-Resource Data Excessively

A small dataset repeated too many times may cause memorization and reduce diversity.

### Pitfall 9: Assuming Synthetic Data Is Correct

Synthetic outputs may contain fluent but systematic errors.

### Pitfall 10: Discarding Provenance

Without provenance, source removal, debugging, compliance checks, and reproducibility become difficult.

### Pitfall 11: Reporting Only Final Dataset Size

A final token count does not reveal which sources were removed, repeated, or overrepresented.

### Pitfall 12: Failing to Test the Data Loader

A correct offline mixture can become incorrect during distributed sampling, sharding, or checkpoint recovery.

---

## 2.49 Summary

- Training data is a central component of LLM behavior, not merely an input to the model.
- Scale, quality, and diversity must be optimized together.
- A production data pipeline includes acquisition, extraction, normalization, filtering, deduplication, decontamination, mixture design, tokenization, packing, and sharding.
- Provenance should be preserved throughout the complete lifecycle.
- Rule-based filters are scalable and interpretable, but require language- and source-specific calibration.
- Model-based quality filters can improve selection while also introducing hidden stylistic and cultural biases.
- Exact deduplication removes identical documents, while MinHash and LSH efficiently detect near-duplicates.
- Duplicate clusters should retain the highest-utility representative.
- Benchmark decontamination must distinguish exact memorization from legitimate domain exposure.
- Dataset mixtures should be measured in actual sampled tokens, not only document counts.
- Synthetic data can improve coverage but requires validation, diversity controls, and traceable generation metadata.
- Sequence packing, shuffling, sharding, and distributed loading directly affect training efficiency and reproducibility.
- Dataset governance, versioning, and quality reports are essential for responsible and repeatable model development.

---

## Review Questions

1. Why can increasing the raw number of tokens reduce model quality?
2. What is the difference between extraction and normalization?
3. Why should normalization policies be language-aware?
4. How does document-level language identification differ from segment-level identification?
5. What risks arise from applying one quality threshold to every source?
6. Why is exact hashing insufficient for web-scale deduplication?
7. How does MinHash approximate Jaccard similarity?
8. How do the number of LSH bands and rows affect candidate retrieval?
9. What criteria can be used to choose a representative from a duplicate cluster?
10. Why should deduplication occur across training and validation data?
11. How does exact benchmark contamination differ from domain exposure?
12. What does the temperature parameter control in dataset mixture sampling?
13. Why should effective epochs be monitored for smaller datasets?
14. How can synthetic data reduce rather than increase corpus diversity?
15. Why should data-mixture statistics be computed using the production tokenizer?
16. What is sequence packing, and why does it improve training efficiency?
17. Which data-loader states are needed for deterministic checkpoint recovery?
18. How do provenance metadata and dataset fingerprints support reproducibility?

---

## Suggested Exercises

### Exercise 1: Rule-Based Quality Analysis

Select a multilingual sample and calculate:

- Document length
- Alphabetic-character ratio
- Repeated-line ratio
- URL density
- Unique-token ratio

Inspect documents near each threshold and determine whether a single global policy is appropriate.

### Exercise 2: Exact Deduplication

Normalize and hash a small collection of documents. Compare duplicate rates before and after:

- Whitespace normalization
- Unicode normalization
- Lowercasing
- Boilerplate removal

Document any false merges caused by aggressive normalization.

### Exercise 3: MinHash Deduplication

Build a MinHash signature for each document using token 5-grams.

Experiment with:

- Signature size
- Number of LSH bands
- Number of rows per band
- Jaccard verification threshold

Measure candidate recall and false-positive rate.

### Exercise 4: Dataset Mixture Design

Given three datasets containing:

```text
General web:       800 billion tokens
Technical text:     80 billion tokens
Persian text:       20 billion tokens

Calculate sampling probabilities for:
```
$ \alpha \in \{1.0, 0.7, 0.5, 0.0\} $

Then estimate the effective epochs of each dataset for a one-trillion-token training run.


### Exercise 5: Decontamination Audit

Choose a small evaluation benchmark and search for:

- Exact matches
- Normalized matches
- N-gram overlaps
- Near-duplicates
- Semantically similar documents

Classify each result as exact contamination, near contamination, task exposure, or domain exposure.

### Exercise 6: Persian-English Tokenization Audit

For matched Persian and English documents:

1. Count characters.
2. Count whitespace-separated words.
3. Count tokens using the intended tokenizer.
4. Compare tokenization fertility.
5. Inspect the effects of Persian character and ZWNJ normalization.

---

## Further Reading

1. Dodge, J. et al.  
   *Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus*. EMNLP, 2021.

2. Gao, L. et al.  
   *The Pile: An 800GB Dataset of Diverse Text for Language Modeling*. 2020.

3. Penedo, G. et al.  
   *The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only*. 2023.

4. Lee, K. et al.  
   *Deduplicating Training Data Makes Language Models Better*. ACL, 2022.

5. Broder, A. Z.  
   *On the Resemblance and Containment of Documents*. Compression and Complexity of Sequences, 1997.

6. Leskovec, J., Rajaraman, A., and Ullman, J. D.  
   *Mining of Massive Datasets*. Cambridge University Press.

7. Kreutzer, J. et al.  
   *Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets*. TACL, 2022.

8. Longpre, S. et al.  
   *The Flan Collection: Designing Data and Methods for Effective Instruction Tuning*. ICML, 2023.

9. Soldaini, L. et al.  
   *Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research*. 2024.

10. Biderman, S. et al.  
*Data Provenance Initiative: A Large-Scale Audit of Dataset Licensing and Attribution in AI*. 2023.
---
[Previous: Introduction to Large Language Models](./01-introduction.md) |
[Contents](./index.md) |
[Next: Tokenization](./03-tokenization.md)
---