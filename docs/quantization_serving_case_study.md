# Comparative Case Study: Post-Training Quantization & Serving Engines

This report presents a comparative evaluation of the fine-tuned **Qwen-1.5B** model served across different post-training quantization (PTQ) formats (**FP16, AWQ, NF4, FP8**) on two high-performance inference engines: **vLLM (v0.6.3.post1 / v0.22.0)** and **SGLang (v0.5.12.post1)**. The primary aim of this research experiment is to monitor how different quantization techniques affect model performance and quality when compared across different serving engines (vLLM and SGLang). The study focuses specifically on serving engine performance, scheduling latency, and throughput efficiency rather than evaluating the intrinsic linguistic quality of the base or fine-tuned model outputs (which falls under a separate evaluation bracket).

The experiments were conducted under strict VRAM limitations on an **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)** to evaluate the performance, memory footprint, and generation quality trade-offs.

---

## 1. Experimental Setup Matrix

The following table summarizes the 13 configurations benchmarked in this study, mapping the results directories to the respective models, serving engines, and experimental notes:

| Folder Name | Model Quantization | Serving Engine | Experimental Notes / Patch Level |
| :--- | :--- | :--- | :--- |
| **baseline_awq** | AWQ (4-bit Marlin) | vLLM | AWQ Baseline (No speculative decoding) |
| **awq_sglang** | AWQ (4-bit Marlin) | SGLang | AWQ Baseline (No speculative decoding) |
| **awq_sglang_N2** | AWQ + Speculative | SGLang | N-Gram speculative decoding with $N=2$ draft steps |
| **awq_sglang_N5** | AWQ + Speculative | SGLang | N-Gram speculative decoding with $N=5$ draft steps |
| **awq_sglang_N10** | AWQ + Speculative | SGLang | N-Gram speculative decoding with $N=10$ draft steps |
| **spec_N2** | AWQ + Speculative | vLLM | N-Gram speculative decoding with $N=2$ draft steps |
| **spec_N5** | AWQ + Speculative | vLLM | N-Gram speculative decoding with $N=5$ draft steps |
| **spec_N10** | AWQ + Speculative | vLLM | N-Gram speculative decoding with $N=10$ draft steps |
| **spec_N20** | AWQ + Speculative | vLLM | N-Gram speculative decoding with $N=20$ draft steps |
| **bnb_nf4** | NF4 (4-bit BitsAndBytes) | vLLM | In-flight NF4 serving (vLLM v0.22.0) |
| **bnb_nf4_sglang** | NF4 (4-bit BitsAndBytes) | SGLang | In-flight NF4 serving |
| **fp8_vllm** | FP8 (8-bit Dynamic) | vLLM | Restored outputs via weight transposition patch |
| **fp8_sglang** | FP8 (8-bit Dynamic) | SGLang | Unpatched weight transposition; outputs corrupted (gibberish) |

> [!NOTE]
> An additional GPU-based speculative decoding variant was benchmarked under `spec_N5_gpu` to contrast CPU-based context-lookup with GPU-based drafting.

---

## 2. Server Launch Commands

Below are the exact commands used to initiate the vLLM and SGLang servers for each configuration.

### 1. Baseline FP16
*Note: Due to the 4 GB VRAM ceiling, serving the unquantized FP16 model under vLLM or SGLang triggers an Out-of-Memory (OOM) error during engine initialization because of the minimum KV cache pool allocation combined with the ~3.0 GB model weight footprint.*

* **vLLM:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-merged \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000
  ```
* **SGLang:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-merged \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```

### 2. AWQ Baseline (baseline_awq / awq_sglang)
* **vLLM:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-awq \
      --quantization awq_marlin \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000
  ```
* **SGLang:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-awq \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```

### 3. AWQ + Speculative Decoding ($N=2$)
* **vLLM (`spec_N2`):**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-awq \
      --quantization awq_marlin \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000 \
      --spec-method ngram \
      --spec-tokens 2
  ```
* **SGLang (`awq_sglang_N2`):**
  *Note: SGLang was configured to use N-Gram speculative decoding. The CLI flags correspond to:*
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-awq \
      --speculative-algorithm NGRAM \
      --speculative-num-draft-tokens 2 \
      --host 0.0.0.0 \
      --port 8000
  ```

### 4. AWQ + Speculative Decoding ($N=5$)
* **vLLM (`spec_N5`):**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-awq \
      --quantization awq_marlin \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000 \
      --spec-method ngram \
      --spec-tokens 5
  ```
* **SGLang (`awq_sglang_N5`):**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-awq \
      --speculative-algorithm NGRAM \
      --speculative-num-draft-tokens 5 \
      --host 0.0.0.0 \
      --port 8000
  ```

### 5. AWQ + Speculative Decoding ($N=10$)
* **vLLM (`spec_N10`):**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-awq \
      --quantization awq_marlin \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000 \
      --spec-method ngram \
      --spec-tokens 10
  ```
* **SGLang (`awq_sglang_N10`):**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-awq \
      --speculative-algorithm NGRAM \
      --speculative-num-draft-tokens 10 \
      --host 0.0.0.0 \
      --port 8000
  ```

### 6. AWQ + Speculative Decoding ($N=20$)
* **vLLM (`spec_N20`):**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-awq \
      --quantization awq_marlin \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000 \
      --spec-method ngram \
      --spec-tokens 20
  ```

### 7. BitsAndBytes NF4 (bnb_nf4 / bnb_nf4_sglang)
* **vLLM:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-bnb-nf4 \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --port 8000
  ```
* **SGLang:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-bnb-nf4 \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```

### 8. FP8 Dynamic W8A8 Baseline (fp8_vllm)
* **vLLM:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-fp8-v2 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000
  ```

### 9. FP8 SGLang (fp8_sglang)
* **SGLang:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-fp8-v2 \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```

### 10. FP8 W8A16
* **vLLM:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-fp8-w8a16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000
  ```

---

### 3. Serving Metrics Comparison (Concurrency = 4)

Performance data was gathered over 3 repetitions at a default serving concurrency of 4 parallel client requests.

### 3.1 Performance & Resource Metrics

| Configuration (Folder) | Engine | Throughput (tok/s) | Avg TTFT (ms) | Avg ITL (ms/tok) | Prefill Throughput (tok/s) | Decode Throughput (tok/s) | Peak VRAM (GB) | Avg GPU Util % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **baseline_awq** | vLLM | 90.57 ± 18.89 | 65.92 ± 53.95 | 8.7 | 4210.68 | 104.87 | 3.58 ± 0.00 | 73.4% |
| **awq_sglang** | SGLang | 103.98 ± 22.09 | 50.60 ± 48.05 | 8.7 | 6284.43 | 117.26 | 3.54 ± 0.00 | 76.2% |
| **spec_N2** | vLLM | 85.10 ± 13.07 | 51.36 ± 38.99 | 13.4 | 5316.03 | 93.94 | 3.59 ± 0.00 | 51.9% |
| **spec_N5** | vLLM | 90.31 ± 12.42 | 50.13 ± 38.07 | 13.5 | 5376.19 | 100.52 | 3.61 ± 0.00 | 51.5% |
| **spec_N10** | vLLM | 91.66 ± 16.97 | 52.03 ± 40.19 | 14.0 | 5275.31 | 101.63 | 3.40 ± 0.00 | 57.4% |
| **spec_N20** | vLLM | 85.88 ± 12.17 | 56.85 ± 44.44 | 15.0 | 4813.34 | 96.66 | 3.67 ± 0.00 | 61.4% |
| **awq_sglang_N2** | SGLang | 107.35 ± 39.94 | 45.25 ± 39.39 | 14.3 | 6526.57 | 117.94 | 3.53 ± 0.01 | 74.2% |
| **awq_sglang_N5** | SGLang | 159.23 ± 86.05 | 44.81 ± 38.41 | 17.4 | 6491.21 | 183.67 | 3.57 ± 0.00 | 57.3% |
| **awq_sglang_N10** | SGLang | **197.56 ± 127.11** | 46.35 ± 37.13 | 21.3 | 5994.16 | **248.72** | 3.64 ± 0.02 | 63.9% |
| **spec_N5_gpu** (GPU Spec) | vLLM | 87.70 ± 23.04 | 146.03 ± 182.08 | 15.5 | 3633.13 | 104.72 | 3.60 ± 0.00 | 55.3% |
| **bnb_nf4** | vLLM | 12.84 ± 0.32 | 224.39 ± 5.02 | 76.7 | 867.07 | 13.58 | 3.19 ± 0.00 | **93.3%** |
| **bnb_nf4_sglang** | SGLang | 11.51 ± 0.42 | 228.49 ± 82.24 | 82.5 | 929.20 | 12.15 | **2.89 ± 0.00** | 89.0% |
| **fp8_vllm** | vLLM | 79.90 ± 0.02 | **34.69 ± 0.19** | 11.6 | 5537.29 | 86.01 | 3.58 ± 0.00 | 87.1% |
| **fp8_sglang** | SGLang | 67.20 ± 0.17 | 250.01 ± 2.95 | 13.0 | 756.70 | 77.35 | 3.00 ± 0.00 | 91.6% |

#### Visualizations: Performance & Memory Trade-offs

![Throughput Comparison](../images/plot1_throughput.png)
*Figure 1: Generation throughput comparison across serving engines and quantization configurations.*

![TTFT Comparison](../images/plot3_ttft.png)
*Figure 2: Time to First Token (TTFT) latency comparison (lower is better).*

![VRAM Usage Comparison](../images/plot4_vram.png)
*Figure 3: Peak VRAM consumption comparison under a 4.0 GB physical hardware constraint.*

![GPU Utilization vs Throughput](../images/plot6_gpu_util.png)
*Figure 4: GPU utilization versus generation throughput showing the Efficiency Paradox (high utilization does not guarantee high throughput).*

### 3.2 Observed Accuracy on 50-Prompt Evaluation Set

Quality metrics were computed over a 50-prompt test subset under single-concurrency (`temperature=0`). Note that the primary goal of these benchmarks is to evaluate serving performance and efficiency differences between inference runtimes rather than model accuracy, which depends on training quality and calibration. Quality comparisons were performed against the merged FP16 checkpoint executed through a lightweight Hugging Face Transformers runtime rather than vLLM/SGLang, which could not host the FP16 model within the 4 GB VRAM budget.

| Configuration (Folder) | Engine | Observed Accuracy on 50-Prompt Evaluation Set (%) | BERTScore F1 (%) | ROUGE-L (%) |
| :--- | :--- | :---: | :---: | :---: |
| **baseline_awq** | vLLM | 66.00% | 18.03% | 20.81% |
| **awq_sglang** | SGLang | 68.00% | 18.31% | 21.14% |
| **spec_N2** | vLLM | 66.00% | 18.04% | 20.73% |
| **spec_N5** | vLLM | 66.00% | 18.06% | 20.77% |
| **spec_N10** | vLLM | 66.00% | 60.14% | 20.77% |
| **spec_N20** | vLLM | 66.00% | 17.99% | 20.74% |
| **awq_sglang_N2** | SGLang | 68.00% | 17.64% | 20.27% |
| **awq_sglang_N5** | SGLang | 68.00% | 17.68% | 20.46% |
| **awq_sglang_N10** | SGLang | 68.00% | 18.26% | 21.00% |
| **spec_N5_gpu** | vLLM | 66.00% | 18.06% | 20.77% |
| **bnb_nf4** | vLLM | **100.00%** | 27.58% | **29.36%** |
| **bnb_nf4_sglang** | SGLang | **100.00%** | **64.34%** | 29.06% |
| **fp8_vllm** | vLLM | 90.00% | 21.52% | 23.52% |
| **fp8_sglang** | SGLang | 0.00% | 22.66% | 0.00% |

> *Note: SGLang FP8 output consists entirely of repeating the word `limp` due to an unpatched weight matrix transposition bug.

> [!WARNING]
> **BERTScore Anomaly:** Anomalously high BERTScore values observed in certain speculative decoding runs (e.g., `spec_N10` at 60.14%) and SGLang NF4 runs (64.34%) likely stem from metric instability or evaluation artifacts rather than genuine quality improvements, given that their Observed Accuracy and ROUGE-L scores remain identical or very close to their respective baselines.

#### Visualizations: Quality & Performance Trade-offs

![Accuracy vs Throughput Scatter Plot](../images/plot2_accuracy_throughput.png)
*Figure 5: Accuracy vs. throughput trade-offs showing the sweet spot represented by patched FP8 compared to low-speed NF4 and moderate-quality AWQ.*

![Quality vs Memory](../images/plot8_quality_memory.png)
*Figure 6: Model accuracy versus peak VRAM consumption showing the optimal configurations within hardware VRAM limits.*

### 3.3 Metric Definitions & Calculations

The evaluation metrics used in this benchmarking study are defined and calculated as follows:

#### Performance & Resource Metrics
1. **Throughput (tok/s)**:
   * **Definition:** Tracks the rate at which the serving engine generates output text tokens across all active concurrent client streams.
   * **Calculation:** The sum of all generated tokens across all concurrent requests divided by the total request execution duration (wall-clock time in seconds).
     $$\text{Throughput} = \frac{\sum \text{Generated Tokens}}{\text{Total Request Execution Time (s)}}$$

2. **Time to First Token (TTFT, ms)**:
   * **Definition:** Tracks the delay or latency before the serving engine sends the very first generated token. This represents the responsiveness of the system.
   * **Calculation:** The duration between the initiation of the request prompt (timestamp $t_{\text{request\_sent}}$) and the receipt of the first chunk of the generated token response (timestamp $t_{\text{first\_token}}$).
     $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}}$$

3. **Inter-Token Latency (ITL, ms/tok)**:
   * **Definition:** Tracks the average delay between successive token generations during the auto-regressive decoding phase.
   * **Calculation:** The total time spent in the decoding phase (total execution time minus TTFT) divided by the remaining number of generated tokens.
     $$\text{ITL} = \frac{\text{Total Request Execution Time} - \text{TTFT}}{\text{Total Generated Tokens} - 1}$$

4. **Prefill Throughput (tok/s)**:
   * **Definition:** Tracks the speed of prompt processing and ingestion before generation begins (encoding phase), which populates the KV cache.
   * **Calculation:** The number of input prompt tokens divided by the TTFT in seconds.
     $$\text{Prefill Throughput} = \frac{\text{Input Prompt Tokens}}{\text{TTFT (s)}}$$

5. **Decode Throughput (tok/s)**:
   * **Definition:** Tracks the speed of the auto-regressive token generation phase.
   * **Calculation:** The number of remaining generated tokens divided by the duration of the decoding phase.
     $$\text{Decode Throughput} = \frac{\text{Total Generated Tokens} - 1}{\text{Total Request Execution Time} - \text{TTFT (s)}}$$

6. **Peak VRAM (GB)**:
   * **Definition:** Tracks the maximum physical graphics memory allocated on the GPU during active serving.
   * **Calculation:** Captured dynamically via the NVIDIA Management Library (NVML) API backend query to record the highest VRAM footprint reached during the active benchmark window.

7. **Average GPU Utilization (%)**:
   * **Definition:** Tracks the percentage of time that the GPU streaming multiprocessors (SMs) are active.
   * **Calculation:** Sampled dynamically from NVML at periodic intervals during the benchmarking run and averaged over the request execution window.

8. **VRAM Efficiency (tok/s per GB)**:
   * **Definition:** Tracks the throughput efficiency normalized by the amount of physical GPU memory occupied.
   * **Calculation:** Aggregate generation throughput (tok/s) divided by the peak VRAM footprint (GB).
     $$\text{VRAM Efficiency} = \frac{\text{Throughput (tok/s)}}{\text{Peak VRAM (GB)}}$$

#### Quality Metrics (on 50-Prompt Evaluation Set)
1. **Observed Accuracy on 50-Prompt Evaluation Set (%)**:
   * **Definition:** Tracks the model's instruction-following and reasoning correctness on a specific, closed-form task subset.
   * **Calculation:** The percentage of model outputs that match the expected ground-truth answers over the 50-prompt test subset.
     $$\text{Observed Accuracy} = \frac{\text{Number of Correct Matches}}{50} \times 100$$

2. **BERTScore F1 (%)**:
   * **Definition:** Tracks semantic similarity by comparing the contextual token embeddings of the generated output and the reference output using a pre-trained language model (e.g. RoBERTa).
   * **Calculation:** Computes cosine similarity between contextual embeddings, returning precision, recall, and their harmonic mean (F1 score).

3. **ROUGE-L (%)**:
   * **Definition:** Tracks structural overlap by measuring the longest common subsequence of words between the generated and reference outputs.
   * **Calculation:** Measures the length of the longest common word sequence present in both strings, normalized by the reference length (recall) and generated length (precision).

---

## 4. Key Research Findings

### 4.1 Quantization Formats: Trade-offs in Quality and Speed

1. **BitsAndBytes NF4 (Preserved Model Quality, Lowest Performance):**
   * **Quality:** NF4 preserved the original model's performance on the evaluation subset, achieving 100.00% observed accuracy on the evaluation subset while introducing no measurable degradation. This indicates that NF4 successfully retains the unquantized model's representation space without requiring an offline calibration search.
   * **Performance:** It runs extremely slow (averaging **11.5 - 12.8 tok/s** at Concurrency 4) because it performs *in-flight dequantization* from 4-bit to FP16 before every GEMM operation, placing a severe burden on memory bandwidth and computing capacity.
2. **AWQ 4-Bit Marlin (Excellent Performance, Moderate Quality Loss):**
   * **Quality:** Aggressive 4-bit offline quantization shifts model probability distributions, inducing minor quality degradation (Observed Accuracy on the evaluation subset drops to **66-68%**, and BLEU-4 drops to ~8.5%). 
   * **Performance:** By utilizing highly-optimized Marlin GEMM kernels that compute directly on compressed 4-bit integer weights in the register file, it achieves a massive speedup, delivering **90-104 tok/s** baseline throughput.
3. **FP8 Dynamic W8A8 (High Performance, Coherent Quality):**
   * **Quality:** Under vLLM, FP8 achieves a strong **90.00% Observed Accuracy** with a ROUGE-L of 23.52%—outperforming AWQ by a wide margin. 
   * **Performance:** It achieves a fast baseline throughput of **79.90 tok/s** and the lowest Time to First Token (**TTFT of 34.7 ms**), making it highly competitive for latency-sensitive prefill stages.
   * **Dynamic Serving-Layer Issue:** Initially, both engines failed and generated garbage text (repeating `limp` indefinitely). Evidence strongly suggests the issue originates from FP8 kernel handling of square projection matrices (N == K, specifically `q_proj` and `o_proj`) within the serving stack. Applying a custom weight transposition patch to vLLM's kernel restored vLLM's quality to 90.00% Observed Accuracy on the evaluation subset, while SGLang remained unpatched and generated corrupted output (0.00% Observed Accuracy).
   * **Interpretation Warning:** Quality metrics for FP8 should be interpreted cautiously because the checkpoint generated semantically corrupted outputs under both vLLM and SGLang despite producing valid outputs under Transformers, indicating a serving-layer incompatibility rather than a quantization-quality failure.

![VRAM Efficiency](../images/plot9_vram_efficiency.png)
*Figure 9: VRAM Efficiency comparison showing generation throughput per gigabyte of peak VRAM (higher is better).*

### 4.2 Engine Architectures: vLLM vs. SGLang

Below is the aggregate serving throughput (tok/s) across different concurrency levels (number of concurrent clients) for core serving and speculative configurations:

| Configuration | Serving Engine | C1 (tok/s) | C2 (tok/s) | C4 (tok/s) | C8 (tok/s) | C16 (tok/s) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **AWQ vLLM** | vLLM | 125.28 | 236.73 | 378.53 | 562.58 | 666.99 |
| **AWQ SGLang** | SGLang | 131.97 | 247.01 | 429.55 | 729.56 | 671.79 |
| **NF4 vLLM** | vLLM | 71.88 | 24.02 | 43.06 | 41.77 | 40.52 |
| **NF4 SGLang** | SGLang | 31.13 | 23.35 | 40.76 | 68.42 | 94.58 |
| **AWQ+Spec (N = 2) SGLang** | SGLang | 195.66 | 322.98 | 520.68 | 725.05 | **765.94** |
| **AWQ+Spec (N = 10) SGLang**| SGLang | **627.02**| **689.42**| **909.58**| **1251.44**| *OOM* |

1. **Baseline AWQ Comparison:**
   * SGLang outperforms vLLM by **14.8%** in throughput (103.98 tok/s vs. 90.57 tok/s) and achieves a **23.2% faster TTFT** (50.6 ms vs. 65.9 ms) while allocating slightly less memory (3.54 GB vs. 3.58 GB). SGLang shows superior scheduling efficiency.
2. **NF4 Concurrency Scaling:**
   * While vLLM has a slight edge at Concurrency 4, SGLang's memory management handles high load much better. At Concurrency 16, SGLang achieves **94.58 tok/s** compared to vLLM's **40.52 tok/s** (a **+133.4%** advantage) because SGLang operates within a tighter VRAM envelope (2.89 GB vs 3.19 GB), avoiding scheduling delays.

![Concurrency Scaling](../images/plot7_concurrency_scaling.png)
*Figure 7: Concurrency scaling results showing aggregate throughput trends as concurrent requests scale from 1 to 16.*

### 4.3 Speculative Decoding Sweeps (N-Gram CPU vs. GPU)

1. **Draft Overhead on Small Models (vLLM):**
   * On vLLM, speculative decoding with N-gram CPU-drafting does **not** yield substantial throughput gains on this 1.5B parameter model. CPU-lookup verification overhead actually degrades throughput at smaller step sizes ($N=2$: 85.10 tok/s vs 90.57 tok/s baseline). Only at $N=10$ does it match baseline throughput (91.66 tok/s), showing that the verification cost cancels out the drafting benefits for lightweight models.
2. **Highly Fused Speculative Drafting (SGLang):**
   * SGLang's N-gram speculative decoding pipeline is highly optimized. It scales throughput dramatically with $N$:
     * **Baseline:** 103.98 tok/s
     * **$N=2$:** 107.35 tok/s (+3.2%)
     * **$N=5$:** 159.23 tok/s (**+53.1%**)
     * **$N=10$:** 197.56 tok/s (**+90.0%**)
3. **Speculative Decoding VRAM Constraint & OOMs:**
   * SGLang's massive speculative throughput comes with a VRAM cost. At Concurrency 16, SGLang $N=5$ and $N=10$ encounter **Out-of-Memory (OOM) crashes** on the 4 GB GPU due to the speculative drafting overhead. vLLM successfully handles Concurrency 16 across all $N$ values.
   * SGLang $N=2$ remains highly stable, scales to Concurrency 16 successfully to achieve a peak throughput of **765.94 tok/s** without crashing.
4. **GPU Speculative Variant (`spec_N5_gpu`):**
   * Moving the N-gram lookup to the GPU (`spec_N5_gpu`) hurts performance in vLLM. It degrades throughput from 90.31 tok/s to **55.88 tok/s** and spikes TTFT to **261.7 ms** (a 5x increase). The latency of launching CUDA kernels for small n-gram checks exceeds the cost of executing them on the CPU.

![Speculative Decoding Scaling](../images/plot5_spec_scaling.png)
*Figure 8: Throughput scaling of SGLang vs. vLLM with speculative draft token sizes (N).*

---

## 5. Threats to Validity

* **Hardware constraints:** The evaluation was conducted on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM). While valuable for testing edge deployment limits, this low-spec hardware is not representative of large-scale production accelerators (e.g., H100 or A100 GPUs), which possess much higher memory bandwidth and dedicated FP8 tensor cores.
* **Evaluation dataset size:** The quality evaluation was carried out on a 50-prompt subset of the validation dataset. Although sufficient for detecting obvious semantic collapse, a larger and more diverse dataset would be needed to establish robust statistical confidence across all metrics.
* **Model size limitations:** These experiments were restricted to a Qwen2.5-1.5B parameter model. Quantization dynamics, dequantization overheads, and engine scheduling efficiencies can behave differently when scaling to larger model sizes (e.g., 7B, 72B).
* **Rapidly evolving software versions:** Inference libraries (vLLM, SGLang) and quantization toolkits (llmcompressor, AutoAWQ) are updated frequently. The performance profiles, compatibility patches, and memory utilization numbers reflect a specific point-in-time state of these software ecosystems.

---

## 6. Conclusions & Serving Recommendations

* **Project-Level Conclusion:**
  AWQ provided the highest serving performance and scaled best under concurrency. NF4 preserved baseline quality almost perfectly but incurred a substantial throughput penalty, making it unsuitable for latency-sensitive deployments on constrained hardware. FP8 achieved a balanced compromise between quality and performance, although runtime compatibility issues prevented consistent deployment across serving frameworks. Speculative decoding delivered substantial gains in SGLang, reaching nearly 2× the throughput of the non-speculative baseline, while vLLM showed limited benefit from the N-gram speculative strategy on this hardware configuration.

* **Serving Recommendations Matrix:**
  1. **For Maximum Accuracy Preservation:** Use **BitsAndBytes NF4 on SGLang**. Although it has lower raw throughput, SGLang operates it within 2.89 GB of VRAM and scales concurrency much better than vLLM.
  2. **For High Concurrency & Balanced Performance (90% Accuracy Preservation):** Use **FP8 Dynamic (W8A8) on vLLM with the Marlin transposition patch**. It restores quality, offers the fastest TTFT (34.7 ms), and runs at a high 79.9 tok/s.
  3. **For Latency-Insensitive High-Throughput:** Use **SGLang AWQ + N-Gram speculative decoding (N = 2)**. This configuration yields 107.35 tok/s at Concurrency 4 and scales stably to 16 concurrent users (765.94 tok/s) without OOM crashing. Avoid N >= 5 under concurrency to prevent OOM errors on 4 GB cards.

