# Experimental Challenges, System-Level Tradeoffs, and Mitigation Strategies

## Introduction

While the primary objective of this study was to evaluate quantization methods (AWQ, BitsAndBytes NF4/INT8, FP8) across modern inference engines (vLLM and SGLang), a significant portion of the work evolved into understanding the interaction between model compression, serving systems, hardware limitations, and deployment infrastructure.

Unlike many benchmark studies that focus only on final metrics, this work exposed numerous practical challenges that directly influence real-world deployment. These challenges affected model quality, throughput, memory utilization, latency, scalability, and reproducibility.

The following sections document the major issues encountered throughout the experimentation process, the investigative process followed to isolate root causes, and the mitigation strategies applied.

---

# 1. Hardware Constraints and Experimental Design

## Problem

All experiments were conducted on an RTX 3050 Laptop GPU with approximately 4 GB of VRAM.

This immediately imposed several constraints:

* Limited ability to run larger models
* Inability to maintain large KV caches
* Restricted batch sizes
* Reduced concurrency limits
* Difficulty running FP16 baselines alongside monitoring tools
* Limited room for calibration datasets during quantization

Modern serving systems such as vLLM and SGLang are generally optimized for GPUs with substantially larger memory pools (16 GB–80 GB).

## Impact

Without careful resource management:

* Models failed to load
* CUDA Out-of-Memory errors occurred
* Throughput measurements became inconsistent
* Scaling experiments could not complete

## Mitigation

Several design choices were made:

### Model Selection

Instead of using larger models (7B–13B), the study focused on:

* Qwen-2.5-1.5B

This allowed:

* Quantization experiments
* Multi-engine benchmarking
* Concurrency scaling tests

within hardware limits.

### Reduced Calibration Size

For quantization workflows:

```text
Calibration Samples:
512
```

were used instead of several thousand examples.

This reduced memory pressure while still allowing representative calibration.

---

# 2. GPU Utilization Bottlenecks

## Initial Observation

During early benchmarks, GPU utilization frequently remained between:

```text
40% – 60%
```

despite inference workloads being active.

This suggested that the GPU was not being fully saturated.

## Investigation

Monitoring was performed using:

```bash
watch -n 1 nvidia-smi
```

while concurrently tracking:

* throughput
* latency
* VRAM consumption

The bottleneck was traced primarily to:

* insufficient batching
* low concurrency
* conservative memory allocation

rather than raw compute limitations.

## Mitigation

vLLM startup parameters were adjusted.

Examples included:

```bash
--gpu-memory-utilization 0.95
```

and tuning:

```bash
--max-model-len
```

to fit available memory.

## Outcome

* **Theoretical Expectation:** Maximizing static batching and cache allocation parameters should push GPU compute and memory bandwidth utilization close to saturation (90%–100%) during high-concurrency inference loops.
* **Practical Reality:** Peak GPU utilization did exceed **90%** for compute-intensive pipelines (like dynamic BitsAndBytes NF4 at **93.3%** and FP8 W8A8 at **87.1% - 91.6%**). However, for all **speculative decoding sweeps**, GPU utilization remained low (averaging **51%–63%** under vLLM and SGLang, and dropping to **31.6%** for GPU speculative decoding).
* **Reason for Difference:** Speculative decoding on a lightweight 1.5B model is heavily bottlenecked by CPU-side draft token proposal (N-gram lookup) and prompt context scanning. The GPU is forced to wait for sequences of proposed draft tokens, resulting in GPU starvation and sub-optimal utilization despite concurrency scaling.

---

# 3. CUDA Graphs vs Eager Execution Tradeoff

## Background

Modern inference engines use CUDA Graphs to reduce CPU scheduling overhead and improve throughput.

By default:

```text
CUDA Graphs = Enabled
```

in vLLM.

## Problem

CUDA Graph execution introduced several complications during debugging:

* difficult error tracing
* non-deterministic failures
* harder inspection of intermediate states

This became particularly problematic during FP8 debugging.

## Mitigation

A secondary testing configuration was created using:

```bash
--enforce-eager
```

This disabled CUDA Graph execution.

## Tradeoff

### Advantages

* Easier debugging
* More predictable execution
* Better visibility into failures

### Disadvantages

* Reduced decode throughput
* Lower overall serving efficiency (increased inter-token latency due to host-to-device kernel launch dispatch overheads)

## Outcome

* **Theoretical Expectation:** Eager execution is less optimized for GPU scheduling and should result in worse overall latency characteristics, theoretically increasing Time to First Token (TTFT).
* **Practical Reality:** In practice, eager execution actually **decreased** the initial TTFT on first-request cold starts, while CUDA Graphs significantly increased the first-run latency.
* **Reason for Difference:** Enforcing eager execution bypasses the slow CUDA Graph capture, compilation, and profiling warmup phase (which occurs at server startup or during the first client request). While eager execution avoids this initial prefill compilation spike, its decode throughput is substantially lower because it lacks compiled scheduling optimization for the subsequent token generation steps.

---

# 4. Concurrency Scaling Challenges

## Problem

A major goal was evaluating how inference throughput scaled under increasing concurrent load.

Tested concurrency levels:

```text
1
2
4
8
16
```

requests.

## Observation

Scaling was not linear.

Typical behavior:

```text
C=1  → baseline
C=2  → near doubling
C=4  → strong scaling
C=8  → diminishing gains
C=16 → saturation
```

The saturation point differed across:

* vLLM
* SGLang
* quantization methods

## Challenge

Without sufficient warmup:

* throughput varied dramatically
* latency measurements fluctuated

leading to unreliable results.

## Mitigation

Every benchmark repetition was preceded by warmup requests.

This ensured:

* kernels compiled
* caches initialized
* memory allocations stabilized

before measurements were recorded.

---

# 5. LoRA Merge and Tokenizer Consistency

## Problem

The experimental model was created by merging:

```text
Base Qwen Model
+
LoRA Adapter
```

into a single checkpoint.

During early FP8 experiments, the tokenizer associated with the base model was accidentally used instead of the tokenizer from the merged checkpoint.

## Why This Matters

Quantization itself does not modify vocabulary.

However, tokenizer inconsistencies can produce:

* incorrect token IDs
* invalid decoding
* degraded quality
* apparent gibberish output

even when model weights are correct.

## Investigation

Outputs were compared between:

* merged tokenizer
* base tokenizer

and inconsistencies were observed.

## Mitigation

All later experiments explicitly loaded:

```python
AutoTokenizer.from_pretrained(
    "./models/qwen-1.5b-merged"
)
```

ensuring tokenizer-model alignment.

---

# 6. Serving Engine Behavioral Differences

## Problem

A major assumption at the beginning of the project was:

> If a model works in one engine, it should work in another.

This assumption proved false.

## Observation

Several checkpoints behaved differently across:

### Transformers

* Correct outputs

### vLLM

* Correct outputs for AWQ and NF4
* Incorrect outputs for FP8

### SGLang

* Similar behavior to vLLM

## Implication

Model correctness and serving correctness are separate concerns.

A valid checkpoint does not guarantee successful deployment.

## Significance

This became one of the most important findings of the project.

The serving stack itself became an experimental variable.

---

# 7. AWQ Deployment Challenges

## Objective

AWQ was selected as a modern weight-only quantization baseline.

## Problem

Several startup configurations produced inconsistent behavior.

The quantized weights required explicit handling by the inference engine.

Incorrect startup configurations sometimes resulted in:

* loading failures
* degraded performance
* unexpected memory usage

## Mitigation

Dedicated AWQ serving commands were standardized.

The same configuration was reused throughout all benchmark runs to maintain consistency.

## Outcome

AWQ became the most stable quantization method tested.

---

# 8. BitsAndBytes Quantization Tradeoffs

## Objective

Evaluate:

* INT8
* NF4

compression techniques.

## Observation

* **Theoretical Expectation:** Quantizing the 16-bit model (2.89 GB) to 4-bit NF4 should compress the active weights to ~1.2 GB, yielding substantial absolute VRAM savings.
* **Practical Reality:** When served via vLLM, BitsAndBytes NF4 consumed a peak of **3.19 GB of VRAM**, which is higher than the absolute memory footprint of the unquantized FP16 model (which consumed only **3.11 GB of peak VRAM** when executed in a lightweight Hugging Face Transformers wrapper).
* **Reason for Difference:** Advanced serving engines like vLLM pre-allocate large static memory pools for KV caching and maintain complex engine runtimes. While NF4 compression does reduce memory compared to serving FP16 *within the same vLLM engine* (which would otherwise OOM), the runtime engine's overhead, combined with the activation space required for dynamic dequantization, results in a higher absolute VRAM footprint than native execution in a lightweight runtime wrapper.

---

# 9. FP8 Quantization: The Most Significant Challenge

## Objective

Evaluate FP8 as a modern quantization technique increasingly used in production systems.

Particularly:

```text
W8A8
W8A16
FP8_DYNAMIC
```

configurations.

## Initial Success

Quantization completed successfully.

Inspection showed:

```text
torch.float8_e4m3fn
```

weights present.

Checkpoint sizes were reduced substantially.

Transformers loaded the models correctly.

CPU inference produced accurate answers.

Example:

Prompt:

```text
What is the capital of France?
```

Response:

```text
The capital of France is Paris.
```

## Critical Failure

When the same checkpoint was served through vLLM:

outputs became nonsensical.

Examples included:

* multilingual fragments
* repeated tokens
* unrelated factual statements
* random symbols

despite:

* successful loading
* stable memory usage
* high throughput
* normal GPU utilization

## Investigation

Several hypotheses were tested.

### Calibration Issues

Re-quantized using calibration datasets.

No improvement.

### Tokenizer Issues

Verified tokenizer correctness.

No improvement.

### W8A8 vs W8A16

Generated separate checkpoints.

No improvement.

### CPU Validation

Checkpoint worked correctly.

This eliminated:

* corrupted weights
* broken tokenizer
* failed quantization

### Serving Validation

* **Theoretical Expectation:** Dynamic FP8 W8A8 should run natively on modern GPU architectures, yielding high execution speeds with minimal precision loss.
* **Practical Reality:** SGLang FP8 output was completely corrupted (generating the word `limp` continuously), but vLLM was successfully patched to run FP8 with **90.00% Task Accuracy** and **79.90 tok/s** throughput.
* **Reason for Difference:** The failure was not due to Ampere hardware limitations. Evidence strongly suggests the issue originates from FP8 kernel handling of square projection matrices ($N == K$, specifically `q_proj` and `o_proj`) within the serving stack. Applying a custom weight transposition patch to vLLM's loader resolved the issue, while the SGLang engine remained unpatched and corrupted.

---

# 10. Experimental Reproducibility Challenges

## Problem

Benchmark results varied significantly without strict control of execution conditions.

Sources of variability included:

* cold starts
* kernel compilation
* cache initialization
* memory fragmentation

## Mitigation

All experiments followed a consistent procedure:

1. Start server
2. Warmup requests
3. Run benchmark repetitions
4. Record averages
5. Compute standard deviations

This produced far more stable measurements.

---

# 11. Threats to Validity

* **Hardware constraints:** The evaluation was conducted on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM). While valuable for testing edge deployment limits, this low-spec hardware is not representative of large-scale production accelerators (e.g., H100 or A100 GPUs), which possess much higher memory bandwidth and dedicated FP8 tensor cores.
* **Evaluation dataset size:** The quality evaluation was carried out on a 50-prompt subset of the validation dataset. Although sufficient for detecting obvious semantic collapse, a larger and more diverse dataset would be needed to establish robust statistical confidence across all metrics.
* **Model size limitations:** These experiments were restricted to a Qwen2.5-1.5B parameter model. Quantization dynamics, dequantization overheads, and engine scheduling efficiencies can behave differently when scaling to larger model sizes (e.g., 7B, 72B).
* **Rapidly evolving software versions:** Inference libraries (vLLM, SGLang) and quantization toolkits (llmcompressor, AutoAWQ) are updated frequently. The performance profiles, compatibility patches, and memory utilization numbers reflect a specific point-in-time state of these software ecosystems.

---

# Key Lessons Learned

1. Quantization quality and serving quality are independent problems.
2. Hardware limitations heavily influence achievable optimizations.
3. GPU utilization must be actively optimized; it does not automatically reach maximum levels.
4. CUDA Graphs improve performance but complicate debugging.
5. Throughput scaling eventually saturates regardless of quantization method.
6. Tokenizer consistency is critical after LoRA merging.
7. Memory savings do not necessarily translate into throughput improvements.
8. Serving engines can behave differently for identical checkpoints.
9. FP8 remains highly sensitive to hardware and backend implementation details.
10. Benchmarking infrastructure and methodology are just as important as the quantization algorithm itself.

## Final Assessment

Although the original objective was to compare quantization techniques, the project ultimately evolved into a broader investigation of the interaction between model compression, serving systems, hardware constraints, and deployment infrastructure.

The challenges encountered—particularly around FP8 serving, engine compatibility, GPU resource management, and benchmarking methodology—provided insights that extend beyond simple performance numbers and better reflect the realities of deploying compressed large language models in production environments.
production environments.
