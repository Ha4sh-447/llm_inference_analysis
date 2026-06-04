# Research Documentation

This directory contains the detailed experimental analysis and setup guides for the Qwen2.5-1.5B inference optimization:

- **[Inference Benchmarking Report](research_inference_benchmarks.md)**: Full research report covering benchmark methodology, hardware profiles, per-case engine configuration/optimizations, throughput/latency metrics, and qualitative quality comparisons.
- **[N-Gram Speculative Decoding](vllm_ngram_speculative_decoding.md)**: Deep dive into the model-less speculative decoding mechanism that delivers a +50.2% throughput boost for free with zero extra VRAM.
- **[Colab Setup Guide](colab_vllm_setup.md)**: Diagnostic and setup walkthrough for managing CUDA 13 runtime path resolution on Google Colab with PyTorch 2.11+.

*For a high-level summary, repository structure, and reproduction commands, please refer to the root [README.md](../README.md).*
