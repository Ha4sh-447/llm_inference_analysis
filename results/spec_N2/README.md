# Benchmark Results — AWQ + N-Gram Spec (N=2)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** vLLM 0.6.3.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `num_speculative_tokens=2`, `ngram_prompt_lookup_max=4`, `ngram_prompt_lookup_min=1`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 85.10 tok/s | 13.07 | ±32.46 |
| TTFT | 51.4 ms | 38.99 ms | ±96.9 ms |
| ITL | 13.4 ms/tok | 1.54 ms | ±3.8 ms |
| Prefill Latency | 51.4 ms | — | — |
| Prefill Throughput | 5316.03 tok/s | — | — |
| Decode Latency | 502.7 ms | — | — |
| Decode Throughput | 93.94 tok/s | — | — |
| Peak VRAM | 3.59 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 51.9% | — | — |
| GPU Util (peak) | 83.3% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.64% |
| ROUGE-L | 20.73% |
| BLEU-4 | 8.62% |
| BERTScore F1 | 18.04% |
| Task Accuracy | 66.00% |
| Semantic Similarity | 18.04% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 106.89 | 6.66 | ±16.54 |
| 2 | 176.41 | 5.31 | ±13.18 |
| 4 | 334.97 | 32.80 | ±81.48 |
| 8 | 516.18 | 18.06 | ±44.86 |
| 16 | 630.71 | 20.51 | ±50.96 |

---

## 4. All Configs Comparison

| Config | Throughput (tok/s) | TTFT (ms) | ITL (ms/tok) |
|--------|:------------------:|:---------:|:------------:|
| Baseline (no spec) | 90.57 | 65.9 | 8.7 |
| Spec N=2 | 85.10 | 51.4 | 13.4 |
| Spec N=5 | 90.31 | 50.1 | 13.5 |
| Spec N=10 | 91.66 | 52.0 | 14.0 |
| Spec N=20 | 85.88 | 56.9 | 15.0 |
| Spec N=5 (GPU variant) | 87.70 | 146.0 | 15.5 |

Quality remains identical across all configs. Speculative decoding generally increases ITL due to draft verification overhead on this small model, but N=10 achieves a slight throughput speedup (+1.2%) and TTFT reduction over the baseline.

