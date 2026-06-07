# Research Documentation

This directory contains the detailed experimental analysis and setup guides for the Qwen2.5-1.5B inference optimization:

- **[Comparative Case Study](quantization_serving_case_study.md)**: Full research report covering benchmark methodology, hardware profiles, per-case engine configuration/optimizations, throughput/latency metrics, and qualitative quality comparisons.
- **[Metrics & Tradeoffs Analysis](metrics_comparison_and_tradeoffs.md)**: Deep dive into the trade-offs between speed, accuracy, and memory utilization for AWQ, NF4, FP8, and FP16.
- **[Server Params & Configs](model_configs_and_server_params.md)**: Detailed startup commands and technical rationales for vLLM and SGLang parameters.
- **[Experimental Challenges](experimental_challenges_and_mitigations.md)**: Detailed journal of hardware limitations, software bottlenecks, and mitigation workflows.

*For a high-level summary, repository structure, and reproduction commands, please refer to the root [README.md](../README.md).*
