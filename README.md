# Qwen2.5-1.5B — Fine-Tuning & Inference Benchmarking

Fine-tuning [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) for structured instructional generation, then benchmarking inference across six configurations spanning two runtimes (Hugging Face Transformers, vLLM), three quantization methods (NF4, AWQ, BitsAndBytes), and model-less speculative decoding — all under strict VRAM constraints.

## Performance Summary

Greedy decoding, 128 tokens, averaged over 5 measured runs:

| | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 | Case 6 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config** | HF + LoRA (NF4) | HF Merged (BF16) | HF AWQ (Marlin) | vLLM AWQ | vLLM BNB 4-bit | vLLM AWQ + Spec |
| **Throughput** | 8.68 tok/s | 24.97 tok/s | 19.15 tok/s | 132.93 tok/s | 58.85 tok/s | **199.72 tok/s** |
| **Latency** | 14.88s | 5.13s | 6.70s | 0.96s | 2.18s | **0.64s** |
| **TTFT** | 160ms | 50ms | 53ms | **14ms** | 21ms | 14ms |
| **ITL** | 115ms | 40ms | 52ms | 7.5ms | 17ms | **4.9ms** |
| **VRAM** | 1.21 GB | 3.11 GB | 1.17 GB | 2.46 GB | 13.50 GB | 2.46 GB |
| **GPU** | RTX 3050 | RTX 3050 | RTX 3050 | RTX 3050 | Tesla T4 | RTX 3050 |

> Case 5 runs on Google Colab (Tesla T4) due to the 4 GB local VRAM limit. All other cases run on the RTX 3050.

## Repository Structure

```
├── train.py                    # QLoRA fine-tuning script
├── merge_model.py              # Model weights merging script
├── quantize_awq.py             # AWQ 4-bit quantization script
├── inference.py                # Unified benchmarking suite
├── vllm_bnb_colab.ipynb        # Google Colab notebook
├── docs/                       # Research documentation
│   ├── research_inference_benchmarks.md
│   ├── vllm_ngram_speculative_decoding.md
│   └── colab_vllm_setup.md
├── dataset/                    # Training and validation dataset
```

## Quick Start

### Training

```bash
uv run python train.py           # Fine-tune with QLoRA
uv run python merge_model.py     # Merge LoRA adapter into base weights
uv run python quantize_awq.py    # Quantize to AWQ 4-bit
```

### Benchmarking

To run the benchmarking suite, use the unified `inference.py` script. You can run all test cases sequentially (with automatic subprocess VRAM isolation) or target a specific case.

```bash
# Run all 6 benchmark cases sequentially (highly recommended for clean VRAM management)
uv run python inference.py --case all

# Run a specific benchmark case (e.g. Case 6: vLLM AWQ + Speculative Decoding)
uv run python inference.py --case 6
```

## Documentation

| Document | Description |
|:---|:---|
| **[Inference Benchmarking Report](docs/research_inference_benchmarks.md)** | Full research report — methodology, hardware profiles, detailed per-case configuration & engine optimizations, performance metrics, per-case output quality analysis with prompt-level examples, cross-case quality comparison matrix, and conclusions |
| **[N-Gram Speculative Decoding](docs/vllm_ngram_speculative_decoding.md)** | Deep-dive into model-less speculative decoding — how it achieves +50% throughput with zero VRAM overhead and mathematically identical outputs |
| **[Colab Setup Guide](docs/colab_vllm_setup.md)** | Troubleshooting PyTorch 2.11 / CUDA 13 dynamic linker issues (`libnvJitLink.so.13`, `libcudart.so.13`) on Google Colab |

## Conclusion

This study demonstrates that **runtime engine selection and quantization strategy have a far greater impact on inference speed than on output quality** for small language models:

- **Best throughput:** AWQ + vLLM + N-Gram Speculative Decoding delivers **199.72 tok/s** — a **23x speedup** over the baseline — with sub-5ms inter-token latency and zero additional VRAM cost.
- **Best VRAM efficiency:** HF AWQ achieves **1.17 GB** peak VRAM, enabling deployment on GPUs with as little as 2 GB.
- **Fastest model iteration turnaround:** vLLM BitsAndBytes in-flight quantization completely bypasses the offline calibration and compilation pipelines, letting developers evaluate new checkpoints instantly. While its raw inference throughput (58.85 tok/s) is lower than pre-quantized configs, it eliminates all setup preparation time.
- **Quality is a training problem, not a compression problem.** All six configurations produce qualitatively equivalent outputs — the same formatting errors, the same zero-loops, the same source reference memorization dumps. 4-bit quantization (both AWQ and BitsAndBytes) faithfully preserves the model's representational space. The observed failures are properties of the fine-tuned 1.5B weights (catastrophic forgetting from a narrow training set), not artifacts of post-training compression.

For the full per-case analysis including prompt-level output quality breakdowns, see the [Inference Benchmarking Report](docs/research_inference_benchmarks.md).

## Requirements & Setup

This repository is managed using [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency resolution.

### System Prerequisites
- Linux
- Python 3.12
- NVIDIA GPU with CUDA runtime installed

### Environment Setup

Create a virtual environment and install all project dependencies from `pyproject.toml` and `uv.lock`:

```bash
# Create a virtual environment and synchronize dependencies
uv venv
uv sync
```

To run any benchmark or training script within the managed environment without manually activating it, use `uv run`:

```bash
# Example: Run the unified inference suite for Case 4
uv run python inference.py --case 4
```
