# Inference Benchmarking Report: Post-Training Quantization & Runtime Engines

## Abstract

This report evaluates the inference performance, memory footprint, and output quality of six serving configurations for a fine-tuned Qwen2.5-1.5B model. Benchmarks compare Hugging Face Transformers and vLLM runtimes across unquantized (BF16) and quantized (NF4, AWQ, BitsAndBytes) weight formats, targeting realistic production serving conditions on consumer-grade GPU hardware.

The best configuration — 4-bit AWQ served through vLLM's Marlin kernels with N-Gram Speculative Decoding — achieves **199.72 tokens/sec**, a **23× speedup** over the LoRA adapter baseline, with sub-5ms inter-token latency and **zero additional VRAM overhead**.

Qualitative analysis confirms that post-training quantization faithfully preserves model behavior across all configurations. Observed output failures are inherent properties of the trained weights themselves, not artifacts of compression or runtime engine selection — a finding with direct implications for production deployment decisions.

---

## 1. Methodology

### 1.1 Benchmark Protocol

All benchmarks use a strict deterministic protocol to ensure reproducibility and fair cross-case comparison.

| Parameter | Value |
|:---|:---|
| Decoding strategy | Greedy (`temperature=0` / `do_sample=False`) |
| Output length | Exactly 128 tokens (`min_new_tokens=128`, `max_new_tokens=128`) |
| Run structure | 1 discarded warmup run + 5 measured runs, results averaged |
| Memory isolation | `torch.cuda.empty_cache()` + `reset_peak_memory_stats()` between runs |
| Process isolation | vLLM tests use subprocess spawning for a clean GPU state before each benchmark |

### 1.2 Metrics Tracked

| Metric | Definition |
|:---|:---|
| **Throughput** | Tokens generated per second (`tokens ÷ wall_time`) |
| **E2E Latency** | Total wall-clock time for a full 128-token generation |
| **TTFT** | Time to First Token — prompt prefill latency |
| **ITL** | Inter-Token Latency — average decode time per token step |
| **Time to 100 tokens (E2E)** | `TTFT + (100 × ITL)` — includes prefill |
| **Time to 100 tokens (Decode)** | `100 × ITL` — decode-only, excludes prefill |
| **Peak VRAM** | Maximum GPU memory allocated at any point during generation |

### 1.3 Benchmark Prompt

A single fixed prompt is used across all throughput benchmark runs to ensure deterministic, comparable outputs:

```
<|im_start|>system
You are an educational assistant generating questions aligned with Bloom's Taxonomy.<|im_end|>
<|im_start|>user
generate a level 2 question testing comprehension: Photosynthesis is a system of
biological processes by which photosynthetic organisms, such as most plants, algae,
and cyanobacteria, convert light energy, typically from sunlight, into the chemical
energy necessary to fuel their activities.<|im_end|>
<|im_start|>assistant
```

### 1.4 Quality Evaluation

In addition to throughput benchmarking, 10 diverse prompts spanning four cognitive complexity levels were used for qualitative output analysis. These prompts cover a range of technical subject domains and are designed to stress-test the model's instruction-following capabilities and formatting constraints under different context conditions. All generated outputs are saved in the `outputs/` directory for reference.

---

## 2. Hardware Profiles

| | Profile A — Local Workstation | Profile B — Cloud (Google Colab) |
|:---|:---|:---|
| **GPU** | NVIDIA RTX 3050 Laptop | NVIDIA Tesla T4 |
| **VRAM** | 4 GB | 16 GB |
| **Architecture** | Ampere (CC 8.6) | Turing (CC 7.5) |
| **BF16 Tensor Cores** | Native support | FP32 emulation only |
| **FlashInfer support** | Available | Requires Triton fallback |
| **Key constraint** | Hard 4 GB ceiling — forces quantization for most configs | No native BF16; requires `FP16 + TRITON_ATTN` backend |

> Cases 1–4 and Case 6 run on Profile A. Case 5 runs on Profile B due to the BitsAndBytes VRAM requirement exceeding the 4 GB local ceiling.

---

## 3. Configurations, Performance & Output Quality

### Case 1 — Base Model + LoRA Adapter (HF Transformers, NF4)

| Field | Detail |
|:---|:---|
| **Script** | `inference.py` |
| **Runtime** | Hugging Face Transformers + PEFT |
| **Weights** | Base Qwen2.5-1.5B (HF Hub) + fine-tuned LoRA adapter, overlaid at runtime |
| **Quantization** | 4-bit NF4 via BitsAndBytes (double quantization enabled, BF16 compute dtype) |
| **Hardware** | Profile A (RTX 3050) |

**How it works:** The base model is loaded in 4-bit NF4 quantization. The fine-tuned LoRA adapter (rank 16, alpha 32, targeting attention projection layers) is dynamically overlaid at runtime via PEFT. Every forward pass routes activations through both the frozen quantized base weights and the separate LoRA projection matrices — this dual-path computation is the primary source of overhead and makes this the slowest configuration.

| Metric | Value |
|:---|:---|
| Throughput | 8.68 tok/s |
| E2E Latency | 14.88s |
| TTFT | 160.1ms |
| ITL | 114.7ms/tok |
| Time to 100 tok (E2E) | 11.63s |
| Time to 100 tok (Decode) | 11.47s |
| Peak VRAM | **1.21 GB** |
| Disk | N/A (loaded from HF Hub) |

**Output quality:** The model generates structurally formatted outputs with taxonomy tags but quickly degrades into repetitive content within the 128-token budget. It consistently defaults to lower cognitive-level tags regardless of the requested level, and higher-level prompts produce unformatted text dumps rather than well-structured responses.

---

### Case 2 — Fully Merged Model (HF Transformers, BF16)

| Field | Detail |
|:---|:---|
| **Script** | `inference.py` |
| **Runtime** | Hugging Face Transformers |
| **Weights** | LoRA adapter permanently merged into base weights via `merge_and_unload()` |
| **Quantization** | None — native BF16 full precision |
| **Hardware** | Profile A (RTX 3050) |

**How it works:** The LoRA adapter weights are baked into the base model weights on CPU, producing a standalone 2.89 GB BF16 checkpoint with no runtime adapter routing. This eliminates the dual-path overhead entirely, yielding a **2.9× speedup** over Case 1. The tradeoff is significantly higher VRAM consumption (3.11 GB vs. 1.21 GB), which nearly saturates the 4 GB RTX 3050 ceiling and leaves little headroom for longer sequences or larger batches.

| Metric | Value |
|:---|:---|
| Throughput | 24.97 tok/s |
| E2E Latency | 5.13s |
| TTFT | 50.2ms |
| ITL | 40.0ms/tok |
| Time to 100 tok (E2E) | 4.05s |
| Time to 100 tok (Decode) | 4.00s |
| Peak VRAM | 3.11 GB |
| Disk | 2.89 GB |

**Output quality:** The merged model exhibits the same failure patterns as Case 1 but with different surface expressions. Structurally valid responses are followed by catastrophic degeneration — zero-digit repetition loops (`100000000000...`) filling the remaining token budget on several prompts. This confirms the failure modes are encoded in the weights themselves and are not a side-effect of the LoRA adapter overlay.

---

### Case 3 — AWQ 4-bit (HF Transformers, Marlin Kernels)

| Field | Detail |
|:---|:---|
| **Script** | `inference.py` |
| **Runtime** | Hugging Face Transformers (via GPTQModel) |
| **Weights** | Pre-quantized AWQ checkpoint (32 calibration samples, `quantize_awq.py`) |
| **Quantization** | 4-bit AWQ, GEMM config — Marlin kernels auto-selected at load time by GPTQModel |
| **Hardware** | Profile A (RTX 3050) |

**How it works:** The merged BF16 model is statically quantized offline to 4-bit AWQ using a representative calibration set (group size 128, zero-point correction). At inference time, GPTQModel auto-selects the `AwqMarlinLinear` kernel. The model loads at 1.08 GB — a 63% disk reduction over the BF16 checkpoint.

Despite loading with Marlin kernels, HF AWQ (19.15 tok/s) is **slower** than unquantized BF16 (24.97 tok/s). This is a known characteristic at small model scales and batch size 1: Python-level dispatch overhead in the Transformers inference loop, combined with compilation cost for the Marlin extension on first use, offsets the memory bandwidth savings that AWQ provides. The speedup from AWQ becomes apparent only when the serving engine eliminates this overhead — as demonstrated in Case 4.

| Metric | Value |
|:---|:---|
| Throughput | 19.15 tok/s |
| E2E Latency | 6.70s |
| TTFT | 52.6ms |
| ITL | 52.3ms/tok |
| Time to 100 tok (E2E) | 5.29s |
| Time to 100 tok (Decode) | 5.23s |
| Peak VRAM | **1.17 GB** |
| Disk | **1.08 GB** |

**Output quality:** AWQ outputs show slightly different surface-level formatting patterns compared to Cases 1 and 2 — a consequence of the calibration-based quantization shifting the probability mass at the token level. The underlying failure modes (repetition loops, cognitive-level tag errors, unformatted text dumps) are identical. Compression faithfully preserves both the model's capabilities and its limitations.

---

### Case 4 — AWQ 4-bit (vLLM, CUDA Graphs + Marlin)

| Field | Detail |
|:---|:---|
| **Script** | `inference.py --case 4` |
| **Runtime** | vLLM with AWQ-Marlin fused kernels |
| **Weights** | Same pre-quantized AWQ checkpoint as Case 3 |
| **Quantization** | 4-bit AWQ — Marlin fused GEMM, no intermediate dequantization to FP16 |
| **Hardware** | Profile A (RTX 3050) |

**How it works:** The identical AWQ checkpoint from Case 3 is served through vLLM's optimized inference engine. Three engine-level optimizations stack on top of each other:

- **CUDA Graphs** (`enforce_eager=False`): The full forward pass execution graph is captured and replayed, eliminating CPU→GPU kernel dispatch overhead on every decode step.
- **Marlin fused kernels**: GEMM operations run directly on 4-bit compressed weights in-register — no intermediate unpacking to FP16 in high-bandwidth memory.
- **Memory budgeting** (`gpu_memory_utilization=0.85`): Allocates VRAM headroom for the CUDA Graph pool (~0.47 GB) and pre-allocated KV cache (~0.39 GB), all within the 4 GB ceiling.

The result is a **6.9× speedup** over HF Marlin (Case 3) and a **5.3× speedup** over HF BF16 (Case 2) — on the exact same model weights. This isolates the vLLM engine as the primary driver of throughput, not the quantization format.

| Metric | Value |
|:---|:---|
| Throughput | **132.93 tok/s** |
| E2E Latency | **0.96s** |
| TTFT | **14.3ms** |
| ITL | 7.5ms/tok |
| Time to 100 tok (E2E) | 0.76s |
| Time to 100 tok (Decode) | 0.75s |
| Peak VRAM | 2.46 GB |
| Disk | 1.08 GB |

**Output quality:** Outputs are **bit-for-bit identical** to Case 3. vLLM uses the same Marlin dequantization pathway and the same greedy decoding algorithm, producing token-level decisions equivalent to the HF Transformers path. This confirms that the vLLM engine is a pure throughput optimization — it does not alter model behavior in any way.

---

### Case 5 — BitsAndBytes 4-bit (vLLM + Triton) *(Cloud — Tesla T4)*

| Field | Detail |
|:---|:---|
| **Script** | `inference.py --case 5` |
| **Runtime** | vLLM with BitsAndBytes in-flight quantization |
| **Weights** | Unquantized merged BF16 checkpoint (quantized on-the-fly at engine load) |
| **Quantization** | 4-bit BitsAndBytes — dequantizes to FP16 per forward pass |
| **Hardware** | Profile B (Tesla T4, 16 GB VRAM) |

**How it works:** The unquantized 2.89 GB BF16 checkpoint is loaded and quantized in-flight during vLLM engine initialization — no offline calibration or quantized checkpoint required. This eliminates the AWQ pipeline entirely, making BNB ideal for rapid model iteration and checkpoint testing. Key configuration:

- **`attention_backend="TRITON_ATTN"`**: Required on Turing GPUs (CC 7.5), which lack FlashInfer support.
- **`enable_chunked_prefill=False`**: Removes chunked prefill scheduler overhead for short-sequence workloads.
- **`gpu_memory_utilization=0.90`**: On the 16 GB T4, this pre-allocates a large KV cache (~393K tokens).
- **`fix_cuda_paths()`**: Auto-discovers CUDA 13 pip-package lib directories and patches `LD_LIBRARY_PATH` before CUDA imports — required on Colab with PyTorch 2.11+ (see Section 7).

BNB runs at **58.85 tok/s** — **2.3× slower** than Case 4 (AWQ + vLLM). The performance gap exists because BNB dequantizes weights to FP16 before every GEMM operation, whereas vLLM's Marlin kernels in Case 4 compute directly on 4-bit weights without the intermediate conversion step.

| Metric | Value |
|:---|:---|
| Throughput | 58.85 tok/s |
| E2E Latency | 2.18s |
| TTFT | 21.1ms |
| ITL | 17.0ms/tok |
| Time to 100 tok (E2E) | 1.72s |
| Time to 100 tok (Decode) | 1.70s |
| Peak VRAM | 13.50 GB (allocated pool)† |
| Disk | 2.89 GB |

† BitsAndBytes pre-allocates a large VRAM pool at engine startup; actual model weights after quantization occupy approximately 1.18 GB.

**Output quality:** Outputs closely mirror Case 2 (merged BF16). BNB in-flight quantization operates directly on the original weight distribution without a calibration search step, so the weight semantics are preserved more directly than AWQ. The same structural failure patterns appear — tag-level errors, repetition loops, and unformatted text dumps — confirming that the failure modes are weight-level properties, not quantization artifacts.

---

### Case 6 — AWQ + N-Gram Speculative Decoding (vLLM)

| Field | Detail |
|:---|:---|
| **Script** | `inference.py --case 6` |
| **Runtime** | vLLM with model-less N-Gram Speculative Decoding |
| **Weights** | Same pre-quantized AWQ checkpoint as Cases 3–4 |
| **Quantization** | 4-bit AWQ — identical to Case 4 |
| **Hardware** | Profile A (RTX 3050) |

**How it works:** N-Gram (Prompt Lookup) Speculative Decoding is layered on top of Case 4 at zero additional cost. At each decode step, the engine scans the input prompt for n-gram suffix matches and speculatively proposes up to 5 candidate tokens. All 5 candidates are validated in a single forward pass of the target model. Accepted tokens advance the sequence; rejected tokens fall back to standard single-step decoding.

Key parameters:
- `num_speculative_tokens=5` — propose up to 5 tokens per decode step
- `prompt_lookup_max=4` — match n-gram suffixes up to length 4
- `prompt_lookup_min=1` — accept matches as short as 1 token

This is **model-less** — no draft model is required, meaning **zero additional VRAM**. Because speculative token validation is mathematically equivalent to standard greedy decoding, outputs under `temperature=0` are **provably identical** to Case 4. The mechanism delivers a **+50.2% throughput boost over Case 4** by exploiting structural repetition in the output format to skip forward pass iterations.

| Metric | Value |
|:---|:---|
| Throughput | **199.72 tok/s** |
| E2E Latency | **0.64s** |
| TTFT | 14.2ms |
| ITL | **4.9ms/tok** |
| Time to 100 tok (E2E) | **0.51s** |
| Time to 100 tok (Decode) | **0.49s** |
| Peak VRAM | 2.46 GB |
| Disk | 1.08 GB |

**Output quality:** Outputs are **bit-for-bit identical** to Case 4. Every speculative token is validated against the target model's full probability distribution; any mismatch triggers an immediate fallback. Under deterministic greedy search, mathematical equivalence is guaranteed by the speculative decoding algorithm. All failure modes from Case 4 are identically reproduced.

---

## 4. Consolidated Performance Results

| Metric | Case 1 | Case 2 | Case 3 | Case 4 | Case 5* | Case 6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config** | HF + LoRA (NF4) | HF Merged (BF16) | HF AWQ (Marlin) | vLLM AWQ | vLLM BNB | vLLM AWQ + Spec |
| **Runtime** | Transformers | Transformers | Transformers | vLLM | vLLM | vLLM |
| **Throughput (tok/s)** | 8.68 | 24.97 | 19.15 | 132.93 | 58.85 | **199.72** |
| **E2E Latency (s)** | 14.88 | 5.13 | 6.70 | 0.96 | 2.18 | **0.64** |
| **TTFT (ms)** | 160.1 | 50.2 | 52.6 | **14.3** | 21.1 | 14.2 |
| **ITL (ms/tok)** | 114.7 | 40.0 | 52.3 | 7.5 | 17.0 | **4.9** |
| **Time to 100 tok E2E (s)** | 11.63 | 4.05 | 5.29 | 0.76 | 1.72 | **0.51** |
| **Time to 100 tok Decode (s)** | 11.47 | 4.00 | 5.23 | 0.75 | 1.70 | **0.49** |
| **Peak VRAM (GB)** | **1.21** | 3.11 | **1.17** | 2.46 | 13.50† | 2.46 |
| **Disk (GB)** | N/A | 2.89 | **1.08** | **1.08** | 2.89 | **1.08** |
| **Hardware** | Profile A | Profile A | Profile A | Profile A | Profile B | Profile A |

\* Case 5 ran on Profile B (Tesla T4, 16 GB VRAM). Direct throughput comparison with other cases should account for the different GPU architecture and available memory.

† BitsAndBytes pre-allocates a large VRAM pool at startup; actual model weight footprint is approximately 1.18 GB after in-flight quantization.

### Speedup Summary (relative to Case 1 baseline)

| Case | Config | Speedup vs. Case 1 |
|:---:|:---|:---:|
| 1 | HF + LoRA NF4 | 1.0× (baseline) |
| 2 | HF Merged BF16 | 2.9× |
| 3 | HF AWQ Marlin | 2.2× |
| 4 | vLLM AWQ | 15.3× |
| 5 | vLLM BNB *(T4)* | 6.8× |
| 6 | vLLM AWQ + Spec | **23.0×** |

---

## 5. Output Quality Analysis

### 5.1 Evaluation Prompts List

Ten evaluation prompts representing four cognitive complexity levels were used to stress-test formatting constraints, instruction-following, and generation quality across all configurations:

1. **Prompt 1 (Target Level: Recall):** "Generate a Level 1 (Remember) question based on the following passage: Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, and water into glucose and oxygen. It primarily occurs in the chloroplasts of plant cells."
2. **Prompt 2 (Target Level: Comprehension):** "Generate a Level 2 (Understand) question based on the following passage: Mitochondria are known as the powerhouses of the cell because they generate ATP through cellular respiration. ATP serves as the primary energy currency for cellular activities."
3. **Prompt 3 (Target Level: Comprehension):** "Generate a Level 2 (Understand) question based on the following passage: The water cycle involves evaporation, condensation, precipitation, and collection. It continuously moves water through Earth's atmosphere, land, and oceans."
4. **Prompt 4 (Target Level: Application):** "Generate a Level 3 (Apply) question based on the following passage: Newton's First Law states that an object remains at rest or in uniform motion unless acted upon by an external force. This principle is also called the law of inertia."
5. **Prompt 5 (Target Level: Application):** "Generate a Level 3 (Apply) question based on the following passage: DNA contains genetic instructions used in the growth, development, functioning, and reproduction of living organisms. DNA is organized into chromosomes within the nucleus of eukaryotic cells."
6. **Prompt 6 (Target Level: Analysis):** "Generate a Level 4 (Analyze) question based on the following passage: Cellular respiration and photosynthesis are complementary biological processes. Photosynthesis stores energy in glucose, while cellular respiration releases energy from glucose to produce ATP."
7. **Prompt 7 (Target Level: Analysis):** "Generate a Level 4 (Analyze) question based on the following passage: An ecosystem consists of living organisms interacting with one another and with their physical environment. Energy flows through food chains, while nutrients cycle through ecosystems."
8. **Prompt 8 (Target Level: Analysis):** "Generate a Level 4 (Analyze) question based on the following passage: Plate tectonics explains the movement of Earth's lithospheric plates. Interactions between plates can result in earthquakes, volcanic activity, and mountain formation."
9. **Prompt 9 (Target Level: Application):** "Generate a Level 3 (Apply) question based on the following passage: Chemical reactions involve the transformation of reactants into products. According to the law of conservation of mass, matter is neither created nor destroyed during a chemical reaction."
10. **Prompt 10 (Target Level: Analysis):** "Generate a Level 4 (Analyze) question based on the following passage: Natural selection is the process by which organisms with advantageous traits are more likely to survive and reproduce. Over many generations, this process can lead to evolutionary change within populations."

*Full generated text for each prompt and configuration is available in the `outputs/` directory.*

### 5.2 Failure Mode Breakdown

Across all configurations, three distinct failure modes appear:

| Failure Mode | Description | Prompts Affected |
|:---|:---|:---:|
| **Repetition loop** | Generation enters a high-probability repeating token sequence (e.g. digit strings) and fills the remaining token budget | 7, 8, 10 |
| **Cognitive level drift** | Model generates a lower complexity level than requested — defaults to the most common structure seen during training | 2, 3, 4 |
| **Unstructured text dump** | Output begins with a valid response prefix then continues with unformatted reference material rather than closing the output | 1, 4, 6, 9 |

### 5.3 Key Quality Findings

**1. Quantization does not degrade output quality.**
The unquantized BF16 model (Case 2) exhibits the exact same catastrophic failures as the 4-bit AWQ and BNB models. 4-bit compression faithfully preserves the model's representational space — including its flaws. This is the most important finding for production decisions: choosing a quantization format is a throughput and memory trade-off, not a quality trade-off at this model scale.

**2. The vLLM engine does not alter model outputs.**
Cases 3 and 4 produce bit-for-bit identical outputs despite using entirely different serving stacks (HF Transformers vs. vLLM). Cases 4 and 6 produce identical outputs despite speculative decoding. The runtime engine is a pure throughput optimization layer — it has no effect on generation quality.

**3. The observed failures are weight-level properties, not compression artifacts.**
At the 1.5B parameter scale, fine-tuning on a narrowly-formatted domain-specific dataset produces characteristic failure modes: consistent defaulting to the most-common-seen output structure regardless of instruction, memorization of source-material reference formatting patterns, and degenerate repetition loops under greedy decoding. These behaviors are present in the full-precision model (Case 2) and survive all quantization methods unchanged.

**4. Greedy decoding amplifies failure modes.**
Greedy search (`temperature=0`) always selects the single highest-probability token. Once the model enters a high-probability repeating pattern, there is no stochastic escape mechanism. Sampling-based decoding would likely reduce (but not eliminate) loop failures, since lower-probability tokens would occasionally be selected. For production deployments where output diversity is acceptable, temperature > 0 is recommended to mitigate degeneration.

---

## 6. Production Relevance & Conclusions

The benchmark results in this study — collected at batch size 1, 128 output tokens, greedy decoding — reflect the single-request latency regime most relevant to interactive production serving. At this operating point, the findings translate directly to real deployment decisions:

**Serving engine selection matters more than quantization format.**
Cases 3 and 4 use identical model weights (same AWQ checkpoint). The only difference is the serving stack: HF Transformers vs. vLLM. The result is a **6.9× throughput difference** — far larger than any quantization-level gain. In production, choosing the right serving engine is the highest-leverage optimization available.

**Quantization unlocks hardware — the throughput gain is secondary.**
AWQ's primary value on a 4 GB GPU is not speed, it is headroom. Quantization brings the model from 3.11 GB (Case 2) to 1.17 GB (Case 3), leaving 2.83 GB free for KV cache and larger batch sizes. At scale, that memory headroom translates to higher concurrency and better hardware utilization — which eventually does drive throughput. At batch size 1, the raw speed difference between BF16 and AWQ is negligible or negative (as seen in Cases 2 vs. 3); the payoff comes at higher concurrency.

**Model-less speculative decoding is zero-cost throughput.**
N-Gram Speculative Decoding (Case 6) delivers a +50.2% throughput improvement over an already-optimized vLLM baseline with no additional model, no additional VRAM, and no change to output correctness. For any structured or repetitive output format — fixed schema responses, templated outputs, code generation — it should be considered a default-on optimization in production vLLM deployments.

**In-flight quantization (BNB) enables frictionless iteration.**
Case 5 demonstrates that BitsAndBytes in-flight quantization — which requires no offline calibration or quantized checkpoint — achieves 58.85 tok/s on a T4. For rapid experimentation, checkpoint testing, or development workflows where a quantization pipeline adds friction, BNB is a viable serving option that delivers substantial speedup over a Transformers baseline without any upfront preparation cost.

### Summary of Key Results

| Finding | Evidence |
|:---|:---|
| Best single-request throughput | 199.72 tok/s — AWQ + vLLM + N-Gram Spec (Case 6) |
| Best VRAM efficiency | 1.17 GB — HF AWQ (Case 3) |
| Largest single optimization step | 6.9× — switching from HF Transformers to vLLM (Cases 3→4) |
| Zero-cost throughput gain | +50.2% — adding N-Gram Spec Decode (Cases 4→6) |
| Quality delta across all methods | None — all configurations produce equivalent outputs |

---

## 7. Environment Setup Notes

### Colab CUDA 13 Dynamic Linker Issue

PyTorch 2.11+ installs CUDA 13 runtime libraries as pip packages (`nvidia-nvjitlink`, `nvidia-cuda-runtime`, etc.) under `/usr/local/lib/python3.12/dist-packages/nvidia/*/lib/`. These directories are **not** on `LD_LIBRARY_PATH` by default, causing `dlopen` failures when vLLM spawns its EngineCore subprocess on Google Colab.

**Solution:** `inference.py` (when run with `--case 5`) includes a `fix_cuda_paths()` function which auto-discovers all nvidia pip-package lib directories, prepends them to `LD_LIBRARY_PATH`, and re-execs the script before any CUDA imports. Alternatively, see [`colab_vllm_setup.md`](colab_vllm_setup.md) for clean environment installation commands.
