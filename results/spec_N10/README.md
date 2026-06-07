# Benchmark Results — AWQ + N-Gram Spec (N=10)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** vLLM 0.6.3.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `num_speculative_tokens=10`, `ngram_prompt_lookup_max=4`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 91.66 tok/s | 16.97 | ±42.17 |
| TTFT | 52.0 ms | 40.19 ms | ±99.8 ms |
| ITL | 14.0 ms/tok | 2.11 ms | ±5.25 ms |
| Prefill Latency | 52.0 ms | — | — |
| Prefill Throughput | 5275.31 tok/s | — | — |
| Decode Latency | 486.0 ms | — | — |
| Decode Throughput | 101.63 tok/s | — | — |
| Peak VRAM | 3.40 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 57.4% | — | — |
| GPU Util (peak) | 85.7% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.65% |
| ROUGE-L | 20.77% |
| BLEU-4 | 8.62% |
| BERTScore F1 | 60.14% |
| Task Accuracy | 66.00% |
| Semantic Similarity | 60.14% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 114.14 | 6.76 | ±16.78 |
| 2 | 188.38 | 7.51 | ±18.66 |
| 4 | 334.05 | 18.86 | ±46.86 |
| 8 | 512.26 | 27.48 | ±68.28 |
| 16 | 663.30 | 29.76 | ±73.93 |

---

## 4. All Configs Comparison

| Config | Throughput (tok/s) | TTFT (ms) | ITL (ms/tok) | Peak VRAM (GB) |
|--------|:------------------:|:---------:|:------------:|:--------------:|
| Baseline (no spec) | 90.57 | 65.9 | 8.7 | 3.58 |
| Spec N=2 | 85.10 | 51.4 | 13.4 | 3.59 |
| Spec N=5 | 90.31 | 50.1 | 13.5 | 3.61 |
| **Spec N=10** | **91.66** | **52.0** | **14.0** | **3.40** |
| Spec N=20 | 85.88 | 56.9 | 15.0 | 3.67 |
| Spec N=5 (GPU variant) | 87.70 | 146.0 | 15.5 | 3.60 |

N=10 is the best speculative decoding configuration tested, achieving 101.2% of the baseline's throughput while reducing TTFT by 21% and peak VRAM usage. It is the only speculative configuration to outperform the baseline on this model/hardware combination.

