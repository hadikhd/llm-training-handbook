---
id: Systems
title: Systems
sidebar_label: Systems
sidebar_position: 10
description: The distributed systems, hardware, memory, communication, reliability, and cost engineering behind LLM training.
---

<div className="chapter-hero">

![Chapter 9 — Evaluation](/static/static/img/chapters/systems.png)

</div>

[Previous: Evaluation](./08-evaluation.md) |
[Contents](./index.md) |
[Next: Inference and Decoding](./10-inference-and-decoding.md)

---
## Learning Objectives

By the end of this chapter, you should be able to:

- Understand the hardware and distributed-systems layers required for LLM training.
- Analyze memory, compute, communication, and data-pipeline bottlenecks.
- Compare data, tensor, pipeline, sequence/context, expert, and hybrid parallelism.
- Use profiling, observability, checkpointing, fault tolerance, and cost metrics to operate training reliably.


---

## Engineering Lens

Read this chapter with four engineering questions in mind:

1. **What problem does this technique solve?**
2. **How do we measure whether it is working?**
3. **What trade-offs or resource costs does it introduce?**
4. **What failure modes should we expect, and how would we diagnose them?**


## 1. Why Systems Matter

Training a large language model is not only a machine-learning problem. It is also a distributed-systems problem.

The mathematical training objective may be simple:
```text
predict the next token
```
Executing that objective efficiently across hundreds or thousands of accelerators is much more complex. The system must continuously load data, run distributed computation, synchronize gradients, update parameters, save checkpoints, recover from failures, and record enough metadata to reproduce the run.

At small scale, inefficient code may only make training slower. At large scale, the same inefficiency can waste substantial compute, cause instability, or make the training run economically infeasible.

A successful LLM system must balance:

- numerical correctness
- training stability
- hardware utilization
- memory efficiency
- communication efficiency
- storage throughput
- fault tolerance
- observability
- reproducibility
- operational cost

The objective is not merely to make training run. The objective is to make it run correctly, efficiently, and recoverably.

---

## 2. The LLM Training Stack

An LLM training system contains several interacting layers:

```text
training data
-> storage and data pipeline
-> tokenizer and sequence packing
-> training framework
-> distributed runtime
-> communication libraries
-> accelerator kernels
-> GPU or TPU hardware
-> cluster network
-> checkpoint storage
-> monitoring and orchestration

```
A problem in any layer can limit the entire system.

Examples include:

- slow storage causing accelerators to wait for data
- incorrect loss masking corrupting the objective
- poor sequence packing wasting compute on padding
- communication overhead limiting distributed scaling
- checkpoint writes pausing training
- unstable kernels producing NaNs
- scheduler failures terminating healthy workers
- insufficient logging making failures impossible to diagnose

System performance must therefore be analyzed end to end.

---

## 3. Hardware Foundations

Most modern LLMs are trained on GPUs or other specialized accelerators.

Important hardware characteristics include:

| Characteristic |  Why It Matters |
|---|---|
| Compute throughput | Determines the rate of matrix operations |
| High-bandwidth memory | Stores parameters, gradients, activations, and optimizer states |
| Memory bandwidth | Determines how quickly data moves to compute units |
| Interconnect bandwidth | Determines distributed communication speed |
| Supported precision | Enables `FP32`, `TF32`, `FP16`, `BF16`, `FP8`, or other formats |
| Kernel support | Determines whether optimized operations are available |
| Reliability features | Help detect and recover from hardware errors |

Peak hardware specifications are theoretical. Real training throughput is usually lower because of memory movement, communication, kernel launches, synchronization, data loading, and pipeline imbalance.

The useful question is:

How much of the hardware's theoretical capacity does the training workload sustain?

---

## 4. Compute-Bound and Memory-Bound Operations

An operation is compute-bound when arithmetic throughput is the primary limitation. It is memory-bound when moving data is the primary limitation.

Large matrix multiplications are often compute-bound when their dimensions are sufficiently large. Operations such as normalization, activation functions, embedding lookup, and optimizer updates may be limited by memory bandwidth.

This distinction matters because different optimizations address different bottlenecks:

| Bottleneck | Potential Optimization |
|---|---|
| Compute throughput | Lower precision, optimized matrix kernels |
| Memory bandwidth | Kernel fusion, fewer memory reads and writes |
| GPU memory capacity | Sharding, checkpointing, smaller micro-batches |
| Communication | Better parallelism layout, overlap, faster interconnect |
| Data loading | Prefetching, caching, sharded datasets |
| Synchronization | Reduce barriers and load imbalance |

Adding more accelerators does not solve a memory-bandwidth or data-pipeline bottleneck automatically.

---

## 5. Memory Usage During Training

Training memory is consumed by more than model parameters.

The major components are:

```text
parameters
+ gradients
+ optimizer states
+ activations
+ temporary buffers
+ communication buffers

```
For Adam, optimizer states commonly include first- and second-moment estimates. Depending on precision and implementation, optimizer states and gradients can require substantially more memory than the parameters themselves.

A rough conceptual breakdown is:

| Component | Scales Mainly With |
|---|---|
| Parameters | Model size |
| Gradients | Model size |
| Optimizer states | Model size |
| Activations | Batch size, sequence length, layers, hidden size |
| Attention intermediates | Batch size, heads, sequence length |
| Communication buffers | Distributed strategy and bucket size |

Memory estimates should include temporary allocations and fragmentation. A configuration that fits according to a simplified calculation may still fail at runtime.

---

## 6. Mixed-Precision Training

Mixed-precision training uses lower-precision numerical formats for many operations while preserving higher precision where needed for stability.

Common formats include:

| Format | Characteristics |
|---|---|
| `FP32` | High precision and range, high memory and compute cost |
| `TF32` | Faster matrix operations on supported hardware |
| `FP16` | Low memory use, limited numerical range |
| `BF16` | Low memory use with a wider exponent range than `FP16` |
| `FP8` | Higher efficiency, but requires careful scaling and hardware support |

`BF16` is widely used for LLM training because it offers a useful balance between efficiency and numerical stability.

Mixed precision still requires attention to:

- master weights
- optimizer-state precision
- gradient accumulation precision
- loss scaling for `FP16`
- overflow and underflow detection
- sensitive operations such as normalization and reductions

Lower precision improves efficiency only when the complete training stack supports it correctly.

---

## 7. Distributed Training

When a model or training workload no longer fits efficiently on one accelerator, work must be distributed.

The primary forms of parallelism are:

- data parallelism
- tensor parallelism
- pipeline parallelism
- sequence or context parallelism
- expert parallelism

These methods can be combined into multi-dimensional parallelism.

For example:

```text
total_accelerators =
data_parallel_size
* tensor_parallel_size
* pipeline_parallel_size
* context_parallel_size

```
The correct strategy depends on model size, sequence length, hardware memory, network topology, batch size, and implementation maturity.

---

## 8. Data Parallelism

Data parallelism places a copy of the model on each worker and gives each worker a different portion of the batch.

The basic process is:

```text
each worker:
runs forward pass
computes local gradients

all workers:
synchronize gradients
apply equivalent parameter updates

```
The effective global batch size is approximately:

```text
global_batch_size =
micro_batch_size
* gradient_accumulation_steps
* data_parallel_workers

```
Standard data parallelism is simple and effective when the model fits on each device. Its main limitation is that parameters, gradients, and optimizer states are replicated across workers.

Gradient synchronization commonly uses an `all-reduce` collective operation. At large scale, the cost of this communication can become significant.

---

## 9. Fully Sharded Data Parallelism

Fully sharded data parallelism reduces memory replication by sharding model states across data-parallel workers.

Depending on the implementation, it may shard:

- optimizer states
- gradients
- parameters

This family of strategies includes concepts commonly associated with `ZeRO` and Fully Sharded Data Parallel, or `FSDP`.

A simplified progression is:

| Strategy | What Is Sharded |
|---|---|
| Basic data parallelism | Nothing |
| Optimizer sharding | Optimizer states |
| Gradient sharding | Optimizer states and gradients |
| Full sharding | Optimizer states, gradients, and parameters |

Full sharding reduces memory usage but adds communication. Parameters may need to be gathered before computation and released or resharded afterward.

Important configuration choices include:

- sharding granularity
- wrapping policy
- prefetch behavior
- communication bucket size
- mixed-precision policy
- CPU offloading
- checkpoint format

`FSDP` settings should be validated at small scale before large training runs.

---

## 10. Tensor Parallelism

Tensor parallelism divides individual tensor operations across multiple accelerators.

For example, a large linear layer may be partitioned by rows or columns:

Y = X`W`

The weight matrix `W` is split across devices, and workers cooperate to compute the complete result.

Tensor parallelism is useful when a layer or model cannot fit efficiently on one accelerator. It generally requires frequent communication inside each Transformer layer, so it benefits from fast, low-latency interconnects.

Tensor-parallel groups are usually kept within tightly connected hardware domains, such as accelerators connected by high-bandwidth links within a node.

Poor placement of tensor-parallel workers can make communication dominate computation.

---

## 11. Pipeline Parallelism

Pipeline parallelism assigns different groups of Transformer layers to different devices or stages.

A batch is divided into micro-batches that move through the stages:

```text
stage 1 -> stage 2 -> stage 3 -> stage 4

```
`W`ithout micro-batching, most stages would remain idle while one stage is active. Pipeline schedules improve utilization by processing multiple micro-batches concurrently.

Pipeline parallelism introduces several challenges:

- pipeline bubbles
- stage imbalance
- activation transfers
- scheduling complexity
- interactions with gradient accumulation
- more complex checkpointing
- sensitivity to slow workers

Layers should be assigned so that stages have approximately balanced computation and memory requirements. Equal layer counts do not always produce equal workloads.

---

## 12. Sequence and Context Parallelism

Long sequences can make activations and attention computation too large for a single accelerator.

Sequence parallelism partitions operations along the sequence dimension. Context parallelism distributes long-context processing across devices, allowing each worker to handle part of the context.

These methods can reduce per-device activation memory, but attention requires information exchange across sequence partitions.

The design must account for:

- causal masking
- positional encoding
- key-value exchange
- communication volume
- load balancing
- compatibility with optimized attention kernels

Long-context training is a systems challenge as much as an architectural one.

---

## 13. Expert Parallelism

Mixture-of-Experts models contain multiple feed-forward experts while activating only a subset for each token.

A simplified process is:

```text
token representation
-> router
-> selected experts
-> weighted expert outputs

```
Expert parallelism distributes experts across workers.

This reduces active compute relative to total parameter count, but introduces new systems challenges:

- token routing communication
- uneven expert utilization
- overloaded experts
- capacity limits
- dropped or rerouted tokens
- `all-to-all` communication
- expert placement
- routing stability

A poorly balanced router can leave some devices overloaded while others are underused.

---

## 14. Three-Dimensional and Hybrid Parallelism

Large training runs often combine several parallelism strategies.

A common design is:

```text
data parallelism
+ tensor parallelism
+ pipeline parallelism

```
Additional sequence, context, or expert parallelism may be introduced for long-context or MoE models.

The parallelism layout should reflect the physical cluster topology:

```text
fastest links:
tensor parallelism

high-bandwidth local or nearby links:
pipeline or expert communication

larger cross-node groups:
data parallelism

```
This is not a universal rule, but communication-heavy operations should generally use the fastest available links.

More dimensions of parallelism increase implementation and operational complexity. The smallest configuration that satisfies memory and throughput requirements is usually easier to validate and maintain.

---

## 15. Communication Collectives

Distributed training relies on collective communication operations.

Common collectives include:

| Collective | Purpose |
|---|---|
| All-reduce | Aggregate values across workers |
| All-gather | Collect shards from all workers |
| Reduce-scatter | Aggregate and distribute shards |
| Broadcast | Send data from one worker to others |
| All-to-all | Exchange different data among all workers |

Different parallelism strategies produce different communication patterns.

Communication performance depends on:

- message size
- network bandwidth
- network latency
- topology
- collective implementation
- process placement
- contention
- overlap with computation

Small, frequent communication can be latency-bound. Large transfers are more likely to be bandwidth-bound.

---

## 16. Overlapping Communication and Computation

Distributed training becomes more efficient when communication occurs concurrently with useful computation.

Examples include:

- synchronizing one gradient bucket while computing another
- prefetching parameters for the next layer
- loading the next data batch during the current step
- writing checkpoint shards asynchronously
- overlapping pipeline transfers with stage computation

Overlap is not automatic. It may be limited by stream synchronization, dependency ordering, memory pressure, or resource contention.

A profiler is necessary to verify that communication is actually hidden rather than merely scheduled concurrently.

---

## 17. Efficient Attention

Standard attention has quadratic complexity with sequence length:

```text
attention_compute ∝ sequence_length^2

```
Naive implementations may also materialize large intermediate attention matrices.

Memory-efficient attention kernels, such as FlashAttention-style implementations, reorganize computation to reduce high-bandwidth-memory traffic and avoid storing unnecessary intermediates.

Benefits may include:

- lower activation memory
- higher throughput
- support for longer sequences
- fewer memory reads and writes

Correctness still depends on support for:

- causal masks
- padding masks
- variable sequence lengths
- dropout
- grouped-query attention
- sliding-window attention
- the required numerical precision

An optimized kernel should be tested against a trusted reference implementation.

---

## 18. Kernel Fusion and Compilation

A Transformer contains many operations that may otherwise require separate kernel launches and repeated memory access.

Kernel fusion combines compatible operations, such as:

- bias addition and activation
- normalization and residual addition
- optimizer update operations
- rotary-position application
- softmax-related operations

Compilation systems may also fuse operations, specialize graphs for known shapes, and reduce interpreter overhead.

Potential benefits include:

- fewer kernel launches
- less memory traffic
- higher accelerator utilization
- improved throughput

Potential risks include:

- long compilation times
- graph breaks
- shape-dependent recompilation
- numerical differences
- difficult debugging
- hardware-specific behavior

Performance gains should be measured with representative sequence lengths and batch sizes.

---

## 19. Activation Checkpointing

Activations from the forward pass are normally stored for use during backpropagation. For large models or long sequences, activation memory can become a major constraint.

Activation checkpointing stores only selected activations and recomputes missing intermediate values during the backward pass.

The trade-off is:

```text
less memory
in exchange for
more computation

```
Checkpointing can allow larger batches, longer sequences, or larger models. However, aggressive recomputation may significantly reduce throughput.

Useful configuration choices include:

- checkpoint every Transformer block
- checkpoint selected submodules
- use non-reentrant implementations where supported
- preserve only expensive or necessary states
- coordinate checkpointing with sharding

The best policy depends on whether the workload is limited by memory or compute.

---

## 20. CPU and Storage Offloading

When accelerator memory is insufficient, parameters, optimizer states, or activations may be moved to CPU memory or storage.

Offloading can make larger models trainable, but data transfer may become a severe bottleneck.

Potential limitations include:

- PCIe bandwidth
- CPU memory bandwidth
- storage latency
- synchronization delays
- increased implementation complexity

Offloading is most useful when transfers can be prefetched and overlapped with computation. It should not be treated as free memory.

---

## 21. Data Pipeline Engineering

Accelerators cannot remain efficient if the input pipeline cannot supply token batches fast enough.

A training data pipeline may include:

```text
document storage
-> shard selection
-> reading and decompression
-> tokenization or token loading
-> filtering
-> sequence construction
-> packing
-> batching
-> host-to-device transfer

```
For large runs, tokenization is commonly performed before training. Pre-tokenized datasets reduce CPU work and make the training stream more reproducible.

Important data-pipeline properties include:

- deterministic shuffling
- balanced shards
- sequential storage access
- asynchronous prefetching
- pinned host memory
- worker fault handling
- restartable iteration
- dataset-version tracking

Data-loader time and accelerator idle time should be monitored directly.

---

## 22. Sequence Packing Efficiency

Variable-length documents can produce substantial padding waste.

Packing places multiple documents into a fixed-length training sequence:

```text
[document A][EOS][document B][EOS][document C][padding]

```
Packing efficiency can be measured as:

```text
packing_efficiency =
non_padding_tokens / total_sequence_capacity

```
High packing efficiency improves useful tokens per second. However, packing must preserve document boundaries and apply attention or loss rules correctly.

Design questions include:

- Can tokens attend across document boundaries?
- Are separator tokens added correctly?
- Are labels shifted correctly?
- Is loss applied to separator tokens?
- Are partially filled sequences retained?
- Is the packing order deterministic?

A packing bug can silently corrupt the training objective while throughput appears healthy.

---

## 23. Global Batch Size and Gradient Accumulation

The global batch is distributed across devices and optimizer steps.

A common formula is:

```text
global_batch_tokens =
micro_batch_size
* sequence_length
* gradient_accumulation_steps
* data_parallel_size

```
For variable-length or packed examples, actual non-padding tokens should also be measured.

Gradient accumulation allows a large global batch without storing the entire batch at once. Each worker processes several micro-batches before applying an optimizer update.

Important concerns include:

- correct loss normalization
- synchronization only at the intended accumulation boundary
- consistent token counts across workers
- handling variable sequence lengths
- learning-rate adjustment after batch changes
- avoiding unintended gradient reduction on every micro-step

Batch size is an optimization parameter, not only a systems setting.

---

## 24. Throughput Metrics

Training performance should be measured with several metrics.

Common metrics include:

| Metric | Meaning |
|---|---|
| Tokens per second | Total processed tokens per unit time |
| Useful tokens per second | Non-padding training tokens per unit time |
| Samples per second | Processed examples per unit time |
| Step time | Time per optimizer or micro-batch step |
| Model FLOPs utilization | Achieved throughput relative to theoretical compute |
| GPU utilization | Fraction of time accelerator engines are active |
| Communication time | Time spent in distributed communication |
| Data wait time | Time waiting for input |
| Checkpoint time | Training time consumed by checkpointing |

Tokens per second should specify whether it measures per-device or global throughput and whether padding tokens are included.

A single utilization percentage is not enough to diagnose performance.

---

## 25. Scaling Efficiency

Scaling efficiency measures how effectively additional accelerators reduce execution time or increase throughput.

For strong scaling, the workload remains fixed while the number of accelerators increases:

```text
strong_scaling_efficiency =
speedup / number_of_accelerators

```
For weak scaling, workload size increases with the number of accelerators while per-device work remains approximately constant.

Efficiency commonly declines at larger scale because of:

- increased communication
- synchronization overhead
- pipeline bubbles
- stragglers
- network contention
- smaller local matrix dimensions
- scheduler overhead

A larger cluster can finish sooner while using more total accelerator hours. Both wall-clock time and total cost must be considered.

---

## 26. Profiling and Bottleneck Analysis

Optimization should begin with measurement.

A useful profiling process is:

```text
measure end-to-end throughput
-> identify idle periods
-> separate data, compute, and communication time
-> inspect expensive operators
-> test one change
-> verify numerical correctness
-> measure again

```
Profiling tools can reveal:

- kernel execution timelines
- communication collectives
- host-to-device transfers
- memory allocations
- synchronization barriers
- pipeline bubbles
- graph breaks
- data-loader stalls

Profiling short warm runs is usually more practical than collecting traces for an entire training run.

---

## 27. Checkpointing

A checkpoint must contain enough information to resume training correctly.

Depending on the system, it may include:

- model parameters
- optimizer states
- learning-rate scheduler state
- gradient-scaler state
- random-number-generator states
- data-loader position
- consumed-token count
- training configuration
- tokenizer and template versions
- distributed topology metadata

Saving only model weights is insufficient for an exact training resume.

Checkpoint design must balance:

- save frequency
- storage capacity
- write bandwidth
- recovery-point objective
- portability
- resharding requirements
- validation cost

A checkpoint is not reliable until restoration has been tested.

---

## 28. Distributed Checkpoint Formats

Distributed models may store parameters and optimizer states as shards across workers.

A checkpoint may be:

- consolidated into a complete model
- saved as distributed shards
- converted between training and inference formats
- resharded for a different number of workers

Distributed checkpoints can be faster to write and avoid gathering the full state on one worker. However, they may depend on a particular framework version, sharding layout, or world size.

A robust checkpoint workflow should support:

- atomic completion markers
- detection of missing shards
- metadata validation
- checksum verification
- restoration tests
- conversion into a portable release format

Incomplete checkpoints should never be treated as valid.

---

## 29. Fault Tolerance

Long training runs will eventually encounter failures.

Common causes include:

- accelerator errors
- host failures
- network interruptions
- storage outages
- process crashes
- out-of-memory errors
- scheduler preemption
- corrupted checkpoints
- numerical instability

Fault tolerance requires more than periodic checkpointing. The system must detect failure, terminate or isolate affected workers, restore a consistent state, and resume without duplicating or skipping unintended data.

Recovery procedures should be tested deliberately before a costly run begins.

---

## 30. Stragglers and Cluster Variability

Synchronous training proceeds at the speed of the slowest worker.

A straggler may be caused by:

- degraded hardware
- network congestion
- shared storage contention
- thermal throttling
- data imbalance
- background system activity
- uneven pipeline stages
- intermittent errors and retries

Monitoring only average step time can hide these issues. Per-rank timing and communication statistics are often necessary.

Persistent stragglers should be investigated or removed because small delays accumulate across thousands of synchronized steps.

---

## 31. Numerical Stability

Large distributed runs can fail because of numerical instability even when the systems infrastructure is healthy.

Warning signs include:

- NaN or infinite loss
- exploding gradient norms
- sudden loss spikes
- corrupted optimizer states
- divergence after resuming
- inconsistent results across ranks

Useful protections include:

- gradient clipping
- stable normalization implementations
- careful initialization
- learning-rate warmup
- precision-aware reductions
- overflow detection
- finite-value checks
- validation after checkpoint restoration

When a numerical failure occurs, the system should capture enough state to determine whether the cause was data, optimization, precision, or hardware.

---

## 32. Determinism and Reproducibility

Exact determinism is difficult in distributed accelerator training. Some kernels, communication orders, and floating-point reductions can produce small differences across runs.

Reproducibility still requires recording:

- code revision
- framework and library versions
- compiler settings
- hardware type
- distributed topology
- random seeds
- dataset version
- shard order
- tokenizer version
- configuration
- checkpoint lineage

The goal is not always bitwise-identical output. It is often to ensure that results remain within an expected statistical range and that the complete training lineage can be reconstructed.

---

## 33. Observability

A large training run needs structured monitoring.

Important signals include:

### Model and optimization

- training loss
- validation loss
- learning rate
- gradient norm
- parameter norm
- loss scale
- token count
- sequence-length distribution

### Systems

- step time
- tokens per second
- accelerator utilization
- memory usage
- communication time
- data-loader time
- network throughput
- storage throughput
- temperature and power
- hardware error counts

### Operational state

- checkpoint status
- worker health
- job restarts
- data-shard progress
- configuration changes
- software versions

Metrics should be accompanied by logs and traces. Metrics show that a problem occurred; logs and traces help explain where it occurred.

---

## 34. Alerting

Alerts should detect conditions requiring human or automated intervention.

Examples include:

- loss becomes non-finite
- gradient norm exceeds a threshold
- throughput drops significantly
- a worker stops reporting
- checkpoint save fails
- validation loss regresses
- data-loader wait time increases
- memory usage approaches capacity
- hardware errors accumulate
- no progress occurs for a defined interval

Thresholds should distinguish transient variation from persistent failure. Excessive noisy alerts make important incidents easier to miss.

---

## 35. Experiment Configuration

Training configurations should be explicit, versioned, and immutable after a run begins.

A configuration may include:

```yaml
model:
  hidden_size: 4096
  num_layers: 32
  num_attention_heads: 32

training:
  sequence_length: 4096
  global_batch_tokens: 4194304
  learning_rate: 0.0003
  precision: bf16

distributed:
  data_parallel_size: 8
  tensor_parallel_size: 4
  pipeline_parallel_size: 2
```

The exact configuration format is less important than ensuring that the final resolved values are recorded.

Environment-variable overrides and command-line flags should also be captured. Otherwise, the stored configuration may not represent the run that actually executed.

---

## 36. Orchestration and Job Management

Cluster orchestration manages resource allocation, process startup, environment configuration, health checks, retries, and job termination.

A training job should define:

- requested hardware
- process topology
- container or environment image
- storage mounts
- network requirements
- startup commands
- failure policy
- checkpoint location
- logging destination
- graceful shutdown behavior

Distributed workers must agree on ranks, group membership, and rendezvous information. Incorrect process initialization can cause hangs that resemble network failures.

Operational scripts should be tested with small multi-node jobs before full-scale training.

---

## 37. Security and Data Governance

LLM systems often process licensed, private, or sensitive data. Systems design must therefore include access control and auditability.

Relevant controls include:

- encrypted storage and transport
- least-privilege access
- credential rotation
- dataset access logs
- isolated training environments
- secret management
- retention policies
- artifact provenance
- license metadata
- checkpoint access restrictions

Logs and debugging artifacts can also expose training examples or prompts. Observability systems should avoid recording sensitive content unnecessarily.

---

## 38. Cost Engineering

Training cost includes more than accelerator rental.

The complete cost may include:

- accelerator time
- CPU and memory resources
- storage

## Common Failure Modes

- **Low accelerator utilization:** determine whether the job is input-, memory-, compute-, or communication-bound before optimizing.
- **Checkpoint overhead dominates:** measure storage bandwidth, serialization, and checkpoint frequency.
- **Scaling efficiency collapses:** inspect synchronization, collective communication, stragglers, and load imbalance.

## Review Questions

1. How can a training job be compute-bound, memory-bound, communication-bound, or input-bound?
2. When is sharding preferable to simple data parallelism?
3. Which measurements would you collect before attempting a systems optimization?

---
[Previous: Evaluation](./08-evaluation.md) |
[Contents](./index.md) |
[Next: Inference and Decoding](./10-inference-and-decoding.md)
---