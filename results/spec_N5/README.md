# Benchmark Results — AWQ + N-Gram Spec (N=5)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** vLLM 0.6.3.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `num_speculative_tokens=5`, `ngram_prompt_lookup_max=4`, `ngram_prompt_lookup_min=1`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 90.31 tok/s | 12.42 | ±30.85 |
| TTFT | 50.1 ms | 38.07 ms | ±94.6 ms |
| ITL | 13.5 ms/tok | 1.13 ms | ±2.8 ms |
| Prefill Latency | 50.1 ms | — | — |
| Prefill Throughput | 5376.19 tok/s | — | — |
| Decode Latency | 474.6 ms | — | — |
| Decode Throughput | 100.52 tok/s | — | — |
| Peak VRAM | 3.61 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 51.5% | — | — |
| GPU Util (peak) | 79.0% | — | — |

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
| 1 | 114.52 | 3.87 | ±9.62 |
| 2 | 209.73 | 11.71 | ±29.10 |
| 4 | 381.13 | 11.95 | ±29.70 |
| 8 | 563.94 | 43.72 | ±108.61 |
| 16 | 667.71 | 31.01 | ±77.03 |

Throughput scales near-linearly up to C=8, with diminishing returns at C=16 (VRAM-limited on 4 GB GPU).
