# Benchmark Results — AWQ + N-Gram Spec (N=5, GPU variant)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** vLLM 0.6.3.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram (GPU variant), `num_speculative_tokens=5`
**Date:** 2026-06-05

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 55.88 tok/s | 10.89 | ±27.05 |
| TTFT | 261.7 ms | 180.8 ms | ±449.1 ms |
| ITL | 19.9 ms/tok | 2.64 ms | ±6.56 ms |
| Peak VRAM | 3.27 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 31.6% | — | — |
| GPU Util (peak) | 62.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.65% |
| ROUGE-L | 20.77% |
| BLEU-4 | 8.62% |
| BERTScore F1 | 18.06% |
| Task Accuracy | 66.00% |
| Semantic Similarity | 18.06% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 68.58 | 1.22 | ±3.04 |
| 2 | 124.80 | 0.47 | ±1.17 |
| 4 | 250.92 | 17.85 | ±44.35 |
| 8 | 430.87 | 59.14 | ±146.93 |
| 16 | 545.04 | 13.35 | ±33.16 |

---

## 4. All Configs Comparison

| Config | Throughput (tok/s) | TTFT (ms) | ITL (ms/tok) |
|--------|:------------------:|:---------:|:------------:|
| Baseline (no spec) | 113.64 | 35.9 | 7.8 |
| Spec N=5 (CPU) | 66.01 | 70.9 | 18.6 |
| Spec N=5 (GPU variant) | 55.88 | 261.7 | 19.9 |

Quality metrics are identical across all configs (Task Accuracy: 66%, ROUGE-1: ~24.7%) — speculative decoding preserves output quality.
