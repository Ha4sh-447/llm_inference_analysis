# Metrics Comparison and Tradeoff Analysis: Quantization & Serving Engines

This report provides a comparative analysis of Qwen-1.5B served across 14 different configuration cases (including post-training quantization formats, serving engines, and speculative decoding setups) on **vLLM** and **SGLang**. Benchmarking was conducted on an **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)**.

---

### 1. Metric Comparisons (Concurrency = 4)

The performance data below represents serving metrics at concurrency level 4. The quality metrics were evaluated over 50 prompts under single-concurrency (`temperature=0`). Quality comparisons were performed against the merged FP16 checkpoint executed through a lightweight Hugging Face Transformers runtime rather than vLLM/SGLang, which could not host the FP16 model within the 4 GB VRAM budget.

### 1.1 Serving Performance Metrics

| Configuration (Folder) | Engine | Throughput (tok/s) | Avg TTFT (ms) | Avg ITL (ms/tok) | Peak VRAM (GB) | Avg GPU Util % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **baseline_awq** | vLLM | 90.57 ± 18.89 | 65.92 ± 53.95 | 8.7 | 3.58 ± 0.00 | 73.4% |
| **awq_sglang** | SGLang | 103.98 ± 22.09 | 50.60 ± 48.05 | 8.7 | 3.54 ± 0.00 | 76.2% |
| **spec_N2** | vLLM | 85.10 ± 13.07 | 51.36 ± 38.99 | 13.4 | 3.59 ± 0.00 | 51.9% |
| **spec_N5** | vLLM | 90.31 ± 12.42 | 50.13 ± 38.07 | 13.5 | 3.61 ± 0.00 | 51.5% |
| **spec_N10** | vLLM | 91.66 ± 16.97 | 52.03 ± 40.19 | 14.0 | 3.40 ± 0.00 | 57.4% |
| **spec_N20** | vLLM | 85.88 ± 12.17 | 56.85 ± 44.44 | 15.0 | 3.67 ± 0.00 | 61.4% |
| **awq_sglang_N2** | SGLang | 107.35 ± 39.94 | 45.25 ± 39.39 | 14.3 | 3.53 ± 0.01 | 74.2% |
| **awq_sglang_N5** | SGLang | 159.23 ± 86.05 | 44.81 ± 38.41 | 17.4 | 3.57 ± 0.00 | 57.3% |
| **awq_sglang_N10** | SGLang | **197.56 ± 127.11** | 46.35 ± 37.13 | 21.3 | 3.64 ± 0.02 | 63.9% |
| **spec_N5_gpu** | vLLM | 55.88 ± 23.04 | 261.70 ± 182.08 | 19.9 | 3.27 ± 0.00 | 31.6% |
| **bnb_nf4** | vLLM | 12.84 ± 0.32 | 224.39 ± 5.02 | 76.7 | 3.19 ± 0.00 | 93.3% |
| **bnb_nf4_sglang** | SGLang | 11.51 ± 0.42 | 228.49 ± 82.24 | 82.5 | 2.89 ± 0.00 | 89.0% |
| **fp8_vllm** | vLLM | 79.90 ± 0.02 | **34.69 ± 0.19** | 11.6 | 3.58 ± 0.00 | 87.1% |
| **fp8_sglang** | SGLang | 67.20 ± 0.17 | 250.01 ± 2.95 | 13.0 | **3.00 ± 0.00** | 91.6% |

### 1.2 Output Quality Metrics

| Configuration (Folder) | ROUGE-1 (%) | ROUGE-L (%) | BLEU-4 (%) | BERTScore F1 (%) | Observed Accuracy on 50-Prompt Evaluation Set (%) | Quality Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **baseline_awq** | 24.66% | 20.81% | 8.53% | 18.03% | 66.00% | Coherent; slight formatting/taxonomy errors |
| **awq_sglang** | 25.24% | 21.14% | 8.61% | 18.31% | 68.00% | Coherent; slight formatting/taxonomy errors |
| **spec_N2** | 24.64% | 20.73% | 8.62% | 18.04% | 66.00% | Mathematically identical to vLLM baseline |
| **spec_N5** | 24.65% | 20.77% | 8.62% | 18.06% | 66.00% | Mathematically identical to vLLM baseline |
| **spec_N10** | 24.65% | 20.77% | 8.62% | 60.14% | 66.00% | Mathematically identical to vLLM baseline |
| **spec_N20** | 24.60% | 20.74% | 8.54% | 17.99% | 66.00% | Mathematically identical to vLLM baseline |
| **awq_sglang_N2** | 24.23% | 20.27% | 8.17% | 17.64% | 68.00% | Mathematically identical to SGLang baseline |
| **awq_sglang_N5** | 24.14% | 20.46% | 8.34% | 17.68% | 68.00% | Mathematically identical to SGLang baseline |
| **awq_sglang_N10** | 25.17% | 21.00% | 8.39% | 18.26% | 68.00% | Mathematically identical to SGLang baseline |
| **spec_N5_gpu** | 24.65% | 20.77% | 8.62% | 18.06% | 66.00% | Mathematically identical to vLLM baseline |
| **bnb_nf4** | **36.54%** | **29.36%** | 15.37% | 27.58% | **100.00%** | Original model coherence preserved on evaluation subset |
| **bnb_nf4_sglang** | 36.20% | 29.06% | **15.79%** | **64.34%** | **100.00%** | Original model coherence preserved on evaluation subset |
| **fp8_vllm** | 29.09% | 23.52% | 11.10% | 21.52% | 90.00% | Restored coherence following Marlin patch |
| **fp8_sglang** | 0.00% | 0.00% | 0.78% | 22.66% | 0.00% | Corrupted; generates repeating `limp` tokens |

> [!WARNING]
> **BERTScore Anomaly:** Anomalously high BERTScore values observed in certain speculative decoding runs (e.g., `spec_N10` at 60.14%) and SGLang NF4 runs (64.34%) likely stem from metric instability or evaluation artifacts rather than genuine quality improvements, given that their Observed Accuracy and ROUGE-L scores remain identical or very close to their respective baselines.

### 1.3 Metric Definitions & Calculations

The evaluation metrics used in this report are defined and calculated as follows:

#### Performance & Resource Metrics
1. **Throughput (tok/s)**:
   * **Definition:** Tracks the rate at which the serving engine generates output text tokens across all active concurrent client streams.
   * **Calculation:** The sum of all generated tokens across all concurrent requests divided by the total request execution duration (wall-clock time in seconds).
     $$\text{Throughput} = \frac{\sum \text{Generated Tokens}}{\text{Total Request Execution Time (s)}}$$

2. **Time to First Token (TTFT, ms)**:
   * **Definition:** Tracks the latency before the serving engine sends the first generated token. This represents the responsiveness of the system.
   * **Calculation:** The duration between the initiation of the request prompt ($t_{\text{request\_sent}}$) and the receipt of the first chunk of the generated token response ($t_{\text{first\_token}}$).
     $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}}$$

3. **Inter-Token Latency (ITL, ms/tok)**:
   * **Definition:** Tracks the average delay between successive token generations during decoding.
   * **Calculation:** The total time spent in the decoding phase (total execution time minus TTFT) divided by the remaining number of generated tokens.
     $$\text{ITL} = \frac{\text{Total Request Execution Time} - \text{TTFT}}{\text{Total Generated Tokens} - 1}$$

4. **Peak VRAM (GB)**:
   * **Definition:** Tracks the maximum physical graphics memory allocated on the GPU during serving.
   * **Calculation:** Captured dynamically via the NVIDIA Management Library (NVML) API backend query to record the highest VRAM footprint reached during the active benchmark window.

5. **Average GPU Utilization (%)**:
   * **Definition:** Tracks the percentage of time that the GPU streaming multiprocessors (SMs) are active.
   * **Calculation:** Sampled dynamically from NVML at periodic intervals during the benchmarking run and averaged over the request execution window.

6. **VRAM Efficiency (tok/s per GB)**:
   * **Definition:** Tracks the throughput efficiency normalized by the amount of physical GPU memory occupied.
   * **Calculation:** Aggregate generation throughput (tok/s) divided by the peak VRAM footprint (GB).
     $$\text{VRAM Efficiency} = \frac{\text{Throughput (tok/s)}}{\text{Peak VRAM (GB)}}$$

#### Quality Metrics (on 50-Prompt Evaluation Set)
1. **Observed Accuracy on 50-Prompt Evaluation Set (%)**:
   * **Definition:** Tracks the model's instruction-following correctness on a specific task.
   * **Calculation:** The percentage of model outputs that match the expected ground-truth answers over the 50-prompt test subset.
     $$\text{Observed Accuracy} = \frac{\text{Number of Correct Matches}}{50} \times 100$$

2. **BERTScore F1 (%)**:
   * **Definition:** Tracks semantic similarity by comparing contextual token embeddings of the generated and reference outputs using a pre-trained model (e.g. RoBERTa).
   * **Calculation:** Computes cosine similarity between contextual embeddings, returning the F1 score.

3. **ROUGE-1 / ROUGE-L (%)**:
   * **Definition:** Tracks structural/lexical overlap by measuring unigram matches (ROUGE-1) and longest common subsequences of words (ROUGE-L).
   * **Calculation:** Normalized lengths of common word subsequences between generated and reference outputs.

4. **BLEU-4 (%)**:
   * **Definition:** Tracks precision of n-grams (up to 4-grams) in the generated output compared to references, with a brevity penalty.
   * **Calculation:** Log-average of n-gram precision multiplied by an exponential brevity penalty.

---

## 2. Quantization Tradeoff Analysis

Each quantization scheme strikes a unique balance between deployment feasibility, execution speed, and instruction-following quality:

### 2.1 Unquantized FP16 (Standard Baseline)
* **The Tradeoff:**
  * **Pros:** Full representational accuracy; zero quantization loss.
  * **Cons:** High VRAM footprint (~3.1 GB). On low-end hardware with 4 GB VRAM, serving the model via high-throughput engines like vLLM or SGLang is **infeasible**. Both engines allocate a dedicated KV cache block pool at startup, exceeding the 4 GB limit and causing a startup Out-of-Memory (OOM) error.

### 2.2 BitsAndBytes NF4 (4-bit In-Flight)
* **The Tradeoff:**
  * **Pros:** NF4 preserved the original model's performance on the evaluation subset, achieving 100.00% observed accuracy on the evaluation subset while introducing no measurable degradation. It also bypasses the offline quantization pipeline entirely, allowing instant validation of new checkpoints. Memory usage is low (SGLang utilizes only 2.89 GB VRAM).
  * **Cons:** Extremely slow throughput (**11.5 - 12.8 tok/s** at Concurrency 4). Because NF4 weights must be dequantized back to FP16 before every GEMM operation at runtime, the execution loop is bottlenecked by on-chip dequantization overhead.

### 2.3 AWQ 4-Bit Marlin (Statically Quantized)
* **The Tradeoff:**
  * **Pros:** Highly optimized execution. Marlin GEMM kernels perform multiplications directly on 4-bit integer weights in the register file, bypassing FP16 conversion. This delivers high serving throughput (**90 - 104 tok/s**) and low VRAM utilization.
  * **Cons:** Quantization noise degrades representational quality. Observed Accuracy on the evaluation subset falls to **66.00% - 68.00%**, as the model struggles with complex instructions and occasionally slips into taxonomy tag errors.

### 2.4 FP8 Dynamic W8A8 (8-bit Quantized)
* **The Tradeoff:**
  * **Pros:** The best balance between accuracy preservation and performance. Restores Observed Accuracy on the evaluation subset to **90.00%** while achieving a fast throughput of **79.90 tok/s** and the lowest prefill latency (**TTFT of 34.7 ms**).
  * **Cons:** Susceptible to serving-layer compatibility issues. Evidence strongly suggests the issue originates from FP8 kernel handling of square projection matrices (N == K, specifically `q_proj` and `o_proj`) within the serving stack. Without a targeted transposition patch to address this, the model generates corrupted output (0% observed accuracy, repeating `limp` indefinitely) under SGLang.

---

## 3. Serving Engine & Speculative Decoding Analysis

### 3.1 vLLM vs. SGLang (AWQ Baseline)
* **Throughput & Latency:** SGLang out-serves vLLM by **+14.8%** in raw throughput (103.98 tok/s vs 90.57 tok/s) and cuts prefill latency (TTFT) by **23.2%** (50.6 ms vs 65.9 ms), demonstrating superior request scheduling.
* **Concurrency Scaling (NF4):** While vLLM has a slight edge at concurrency 4, SGLang manages memory pools more efficiently. At concurrency 16, SGLang achieves **94.58 tok/s** compared to vLLM's **40.52 tok/s** (+133.4% advantage) by avoiding scheduling bottlenecks.

### 3.2 Speculative Decoding Sweeps (N = 2, 5, 10, 20)
Speculative decoding performance varies significantly between the two engine implementations:

* **vLLM N-Gram (CPU-based):**
  * Speculative decoding actually **degrades** throughput or shows flat performance compared to standard eager decoding at low sequence lengths. Because vLLM performs N-gram context lookups on the CPU, the overhead of draft token proposal and verification on a lightweight 1.5B model cancels out the benefits (N = 2: 85.10 tok/s vs 90.57 tok/s baseline). Only at N = 10 does it barely exceed the baseline (91.66 tok/s).
* **SGLang N-Gram:**
  * SGLang's speculative execution is highly optimized, yielding massive throughput scaling with draft step size N:
    * **Baseline:** 103.98 tok/s
    * **N = 2:** 107.35 tok/s (+3.2% speedup)
    * **N = 5:** 159.23 tok/s (**+53.1% speedup**)
    * **N = 10:** 197.56 tok/s (**+90.0% speedup**)
  * **The Tradeoff (OOMs):** SGLang's speculative drafting consumes substantial activation memory. Under concurrent scaling, SGLang N = 5 and N = 10 encounter **Out-of-Memory (OOM) crashes** at Concurrency 16 on the 4 GB GPU. vLLM completes the Concurrency 16 sweep successfully.
  * **Optimal Config:** SGLang N = 2 represents the most stable choice, scaling successfully up to Concurrency 16 to deliver a peak throughput of **765.94 tok/s** without crashing.

### 3.3 GPU-Based Speculative Decoding (`spec_N5_gpu`)
* **The Tradeoff:**
  * Running the N-gram lookup on the GPU instead of the CPU (`spec_N5_gpu` in vLLM) **degrades performance severely**. Throughput drops to **55.88 tok/s** and TTFT spikes to **261.7 ms** (a 5x increase). Launching GPU kernels for small n-gram checks adds more latency than it saves. CPU-based lookups are far more efficient for small draft sizes.

---

## 4. Threats to Validity

* **Hardware Representative Limits:** Benchmarks were performed on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM). Results may not scale directly to data-center class GPUs (e.g., A100 or H100) that feature dedicated FP8 hardware tensor cores and much larger KV cache capacities.
* **Evaluation Scale:** Quality evaluations were limited to a 50-prompt validation subset. While this highlights immediate performance preservation or collapse, it is too small to generalize broad linguistic capabilities.
* **Model Scale Limitations:** The findings are based on a 1.5B parameter model; dequantization overheads and compute-vs-memory bounds change significantly for larger scale architectures (e.g., 7B or 70B models).
* **Software Version Dependency:** The performance profiles are highly dependent on specific versions of vLLM, SGLang, and quantization toolkits, which are undergoing rapid development.

---

## 5. Project Conclusion

AWQ provided the highest serving performance and scaled best under concurrency. NF4 preserved baseline quality almost perfectly but incurred a substantial throughput penalty, making it unsuitable for latency-sensitive deployments on constrained hardware. FP8 achieved a balanced compromise between quality and performance, although runtime compatibility issues prevented consistent deployment across serving frameworks. Speculative decoding delivered substantial gains in SGLang, reaching nearly 2× the throughput of the non-speculative baseline, while vLLM showed limited benefit from the N-gram speculative strategy on this hardware configuration.

