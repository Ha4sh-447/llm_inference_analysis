# Benchmark Results — AWQ + N-Gram Spec (N=20)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** vLLM 0.6.3.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `num_speculative_tokens=20`, `ngram_prompt_lookup_max=4`, `ngram_prompt_lookup_min=1`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 85.88 tok/s | 12.17 | ±30.23 |
| TTFT | 56.9 ms | 44.44 ms | ±110.4 ms |
| ITL | 15.0 ms/tok | 1.10 ms | ±2.7 ms |
| Prefill Latency | 56.9 ms | — | — |
| Prefill Throughput | 4813.34 tok/s | — | — |
| Decode Latency | 500.2 ms | — | — |
| Decode Throughput | 96.66 tok/s | — | — |
| Peak VRAM | 3.67 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 61.4% | — | — |
| GPU Util (peak) | 89.3% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.60% |
| ROUGE-L | 20.74% |
| BLEU-4 | 8.54% |
| BERTScore F1 | 17.99% |
| Task Accuracy | 66.00% |
| Semantic Similarity | 17.99% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 98.15 | 4.43 | ±10.99 |
| 2 | 169.46 | 6.34 | ±15.75 |
| 4 | 327.24 | 6.81 | ±16.91 |
| 8 | 526.46 | 1.52 | ±3.77 |
| 16 | 632.56 | 20.87 | ±51.85 |

Throughput scales near-linearly up to C=8, with diminishing returns at C=16 (VRAM-limited on 4 GB GPU).
