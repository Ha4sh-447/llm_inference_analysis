# Model Configurations and Server Parameters Rationale

This report details the underlying model configurations, engine command-line parameters, and the technical rationale for their selection across all 13 experimental setups.

---

## 1. Parameters by Case Configuration

### Case 1: Baseline FP16
* **Model Config:** Unquantized full-precision weights (`./models/qwen-1.5b-merged`), model size: **2.89 GB**.
* **vLLM Parameters:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-merged \
      --dtype float16 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000
  ```
* **SGLang Parameters:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-merged \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```
* **Rationale:**
  * `--dtype float16` / standard loading: Runs the model in native half-precision mode.
  * `--gpu-memory-utilization 0.95` / `--mem-fraction-static 0.95`: Attempts to maximize the pre-allocated static pool size for high concurrency serving.
  * **OOM Behavior:** Due to the 4 GB local VRAM limit, reserving 95% of available space (3.8 GB) leaves only ~0.9 GB after loading the 2.89 GB model. Because both engines require substantial baseline allocations for runtime execution and CUDA context setup, loading the model in FP16 triggers an immediate Out-of-Memory (OOM) error.

---

### Case 2: AWQ Baseline (`baseline_awq` / `awq_sglang`)
* **Model Config:** Statically quantized 4-bit AWQ weights (`./models/qwen-1.5b-awq`), model size: **1.08 GB**.
* **vLLM Parameters:**
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
* **SGLang Parameters:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-awq \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```
* **Rationale:**
  * `--quantization awq_marlin`: Overrides the default AWQ kernel and forces vLLM to compile with Marlin kernels. Marlin is a highly optimized execution pathway that computes directly on 4-bit integer weights without unpacking them to FP16 in memory first, resolving bandwidth bottlenecks.
  * `--gpu-memory-utilization 0.95` / `--mem-fraction-static 0.95`: Safely allocates 95% of VRAM. Because the model weights only occupy 1.08 GB, this leaves over 2.7 GB for a large KV cache allocation and CUDA graphs, allowing stable serving on the 4 GB GPU.

---

### Case 3: AWQ + Speculative Decoding ($N=2, 5, 10, 20$)
* **Model Config:** Same as AWQ Baseline.
* **vLLM Parameters (`spec_N<N>`):**
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
      --spec-tokens <N>
  ```
* **SGLang Parameters (`awq_sglang_N<N>`):**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-awq \
      --speculative-algorithm NGRAM \
      --speculative-num-draft-tokens <N> \
      --host 0.0.0.0 \
      --port 8000
  ```
* **Rationale:**
  * `--spec-method ngram` / `--speculative-algorithm NGRAM`: Enables *model-less* speculative decoding. Instead of using a secondary draft model (which would occupy VRAM and cause OOMs), the engine reads the active prompt's context window to identify recurring patterns (n-grams) and speculatively drafts the next tokens.
  * `--spec-tokens <N>` / `--speculative-num-draft-tokens <N>`: Varies the speculative window. A higher $N$ (e.g. 5, 10) allows the engine to skip more forward pass steps if draft sequences are accepted, but incurs a higher computation cost for verification.
  * **OOM Behavior (SGLang):** Running large draft steps ($N \ge 5$) at high concurrency (16) scales activation memory requirements significantly. In SGLang, this combined overhead exceeded the remaining memory pool, triggering OOM errors at concurrency 16. SGLang with $N=2$ was the optimal, stable balance.

---

### Case 4: BitsAndBytes NF4 (`bnb_nf4` / `bnb_nf4_sglang`)
* **Model Config:** 4-bit Normalized Float BitsAndBytes quantized weights (`./models/qwen-1.5b-bnb-nf4`), model size: **1.18 GB**.
* **vLLM Parameters:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-bnb-nf4 \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --port 8000
  ```
* **SGLang Parameters:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-bnb-nf4 \
      --quantization bitsandbytes \
      --load-format bitsandbytes \
      --mem-fraction-static 0.70 \
      --context-length 2048 \
      --attention-backend triton \
      --sampling-backend pytorch \
      --disable-cuda-graph \
      --disable-piecewise-cuda-graph \
      --host 0.0.0.0 \
      --port 8000
  ```
* **Rationale:**
  * `--quantization bitsandbytes --load-format bitsandbytes`: Configures the loader to read and run the dynamic BitsAndBytes 4-bit model structure.
  * `--mem-fraction-static 0.70` (SGLang): Dynamic BitsAndBytes execution requires substantial active GPU memory (activation space) during runtime to dequantize 4-bit weights to FP16 before executing GEMM operations. Lowering SGLang's static memory pool fraction to 0.70 was critical to prevent runtime OOM crashes.
  * `--attention-backend triton --sampling-backend pytorch`: Changes attention/sampling layers to rely on Triton and PyTorch implementations rather than default C++ backends, which are incompatible with the loaded BitsAndBytes weight structures.
  * `--disable-cuda-graph` / `--disable-piecewise-cuda-graph`: Disables SGLang's CUDA graphs. Dynamic on-chip dequantization kernels change execution graphs dynamically at runtime, which causes CUDA graph capture errors. Disabling graphs prevents startup compilation crashes.

---

### Case 5: FP8 Dynamic W8A8 (`fp8_vllm` / `fp8_sglang`)
* **Model Config:** Dynamically quantized FP8 weights and activations (`./models/qwen-1.5b-fp8-v2`), model size: **1.51 GB**.
* **vLLM Parameters:**
  ```bash
  python -m vllm.entrypoints.openai.api_server \
      --model ./models/qwen-1.5b-fp8-v2 \
      --gpu-memory-utilization 0.95 \
      --max-model-len 2048 \
      --max-num-seqs 16 \
      --port 8000
  ```
* **SGLang Parameters:**
  ```bash
  python -m sglang.launch_server \
      --model-path ./models/qwen-1.5b-fp8-v2 \
      --host 0.0.0.0 \
      --port 8000 \
      --mem-fraction-static 0.95 \
      --context-length 2048
  ```
* **Rationale:**
  * **Auto-quantization loading:** Both engines natively detect FP8 (float8_e4m3fn) scaling factors and model weight layers directly from the configuration files. No additional CLI arguments are required.
  * `--gpu-memory-utilization 0.95` / `--mem-fraction-static 0.95`: With a model footprint of 1.51 GB, 95% memory allocation leaves ~2.3 GB of VRAM, which comfortably hosts the execution context and KV cache on the 4 GB GPU.
  * **Marlin Kernel Patch (vLLM only):** Evidence strongly suggests that the default Marlin FP8 kernels suffer from a weight matrix transposition issue when mapping square matrices ($N==K$, such as `q_proj` and `o_proj`), causing the model to output corrupted text. Applying a custom weight transposition patch to vLLM's kernel loading logic restored generation quality. SGLang remained unpatched and generated corrupted outputs (continuous repetition of the token `limp`).

---

## 2. Threats to Validity

* **Hardware constraints:** The evaluation was conducted on an NVIDIA RTX 3050 Laptop GPU (4 GB VRAM). While valuable for testing edge deployment limits, this low-spec hardware is not representative of large-scale production accelerators (e.g., H100 or A100 GPUs), which possess much higher memory bandwidth and dedicated FP8 tensor cores.
* **Evaluation dataset size:** The quality evaluation was carried out on a 50-prompt subset of the validation dataset. Although sufficient for detecting obvious semantic collapse, a larger and more diverse dataset would be needed to establish robust statistical confidence across all metrics.
* **Model size limitations:** These experiments were restricted to a Qwen2.5-1.5B parameter model. Quantization dynamics, dequantization overheads, and engine scheduling efficiencies can behave differently when scaling to larger model sizes (e.g., 7B, 72B).
* **Rapidly evolving software versions:** Inference libraries (vLLM, SGLang) and quantization toolkits (llmcompressor, AutoAWQ) are updated frequently. The performance profiles, compatibility patches, and memory utilization numbers reflect a specific point-in-time state of these software ecosystems.

