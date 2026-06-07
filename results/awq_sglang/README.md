# Benchmark Results — AWQ Baseline (SGLang)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** SGLang 0.5.12.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** None (baseline)
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 103.98 tok/s | 22.09 | ±54.89 |
| TTFT | 50.6 ms | 48.05 ms | ±119.4 ms |
| ITL | 8.7 ms/tok | 1.37 ms | ±3.4 ms |
| Prefill Latency | 50.6 ms | — | — |
| Prefill Throughput | 6284.43 tok/s | — | — |
| Decode Latency | 363.8 ms | — | — |
| Decode Throughput | 117.26 tok/s | — | — |
| Peak VRAM | 3.54 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 76.2% | — | — |
| GPU Util (peak) | 100.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 25.24% |
| ROUGE-L | 21.14% |
| BLEU-4 | 8.61% |
| BERTScore F1 | 18.31% |
| Task Accuracy | 68.00% |
| Semantic Similarity | 18.31% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 131.97 | 0.08 | ±0.21 |
| 2 | 247.01 | 0.06 | ±0.16 |
| 4 | 429.55 | 1.01 | ±2.51 |
| 8 | 729.56 | 5.86 | ±14.56 |
| 16 | 671.79 | 57.91 | ±143.88 |

---

## 4. Engine Comparison: vLLM vs SGLang (AWQ Baseline)

| Metric | vLLM Baseline | SGLang | Delta |
|--------|:-------------:|:------:|:-----:|
| Throughput (tok/s) | 90.57 | 103.98 | **+14.8%** |
| TTFT (ms) | 65.9 | 50.6 | **−23.2%** |
| ITL (ms/tok) | 8.7 | 8.7 | **0.0%** |
| Prefill Latency (ms) | 65.9 | 50.6 | **−23.2%** |
| Prefill Throughput (tok/s) | 4210.68 | 6284.43 | **+49.2%** |
| Decode Latency (ms) | 406.3 | 363.8 | **−10.5%** |
| Decode Throughput (tok/s) | 104.87 | 117.26 | **+11.8%** |
| Peak VRAM (GB) | 3.58 | 3.54 | **−1.1%** |

SGLang shows significant performance advantages over vLLM for the Qwen 1.5B AWQ model on this hardware configuration. It increases serving throughput by 14.8% while reducing TTFT by 23.2% and keeping ITL identical. Prefill throughput in SGLang is particularly fast, achieving a 49.2% speedup compared to vLLM.
