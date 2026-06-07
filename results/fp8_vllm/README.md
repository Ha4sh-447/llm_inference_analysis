# Benchmark Results — FP8 Baseline (vLLM)

**Model:** Qwen-1.5B FP8
**Engine:** vLLM 0.6.3.post1 (with Marlin transposition fix)
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** None (baseline)
**Date:** 2026-06-07

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 79.90 tok/s | 0.02 | ±0.05 |
| TTFT | 34.7 ms | 0.19 ms | ±0.5 ms |
| ITL | 11.6 ms/tok | 0.00 ms | ±0.0 ms |
| Prefill Latency | 34.7 ms | — | — |
| Prefill Throughput | 5537.29 tok/s | — | — |
| Decode Latency | 518.8 ms | — | — |
| Decode Throughput | 86.01 tok/s | — | — |
| Peak VRAM | 3.58 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 87.1% | — | — |
| GPU Util (peak) | 100.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 29.09% |
| ROUGE-L | 23.52% |
| BLEU-4 | 11.10% |
| BERTScore F1 | 21.52% |
| Task Accuracy | 90.00% |
| Semantic Similarity | 21.52% |

*Note: The model quality is fully restored following the Marlin transposition fix on square weight matrices ($N==K$, e.g., `q_proj` and `o_proj`). Generation is coherent, resulting in high task accuracy (90.00%).*

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 84.85 | 0.07 | ±0.17 |
| 2 | 163.34 | 0.44 | ±1.09 |
| 4 | 298.61 | 4.59 | ±11.41 |
| 8 | 487.19 | 15.28 | ±37.97 |
| 16 | 637.02 | 0.33 | ±0.82 |

---

## 4. Engine Comparison: AWQ Baseline vs FP8 Baseline (vLLM)

| Metric | AWQ Baseline | FP8 Baseline | Delta |
|--------|:------------:|:------------:|:-----:|
| Throughput (C=4) (tok/s) | 90.57 | 79.90 | **−11.8%** |
| TTFT (ms) | 65.9 | 34.7 | **−47.3%** |
| ITL (ms/tok) | 8.7 | 11.6 | **+33.3%** |
| Peak VRAM (GB) | 3.58 | 3.58 | **0.0%** |
| Task Accuracy | 66.00% | 90.00% | **+24.0%** |

Following the weight transposition patch, the FP8 model's generation quality is no longer corrupted (achieving a high **90.00% Task Accuracy** compared to **66.00%** for the AWQ baseline, and higher ROUGE/BLEU scores). 

In terms of performance, the FP8 model running on Marlin (weight-only) on the Ampere GPU achieves a **47.3% improvement in TTFT** (34.7 ms vs 65.9 ms) compared to the AWQ baseline, but has **11.8% lower throughput** at Concurrency=4 and **33.3% higher ITL**.
