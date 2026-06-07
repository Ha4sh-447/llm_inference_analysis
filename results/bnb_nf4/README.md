# Benchmark Results — BitsAndBytes NF4

**Model:** Qwen-1.5B BitsAndBytes NF4
**Engine:** vLLM 0.22.0
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** None (baseline)
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 12.84 tok/s | 0.32 | ±0.78 |
| TTFT | 224.4 ms | 5.02 ms | ±12.5 ms |
| ITL | 76.7 ms/tok | 1.87 ms | ±4.6 ms |
| Peak VRAM | 3.19 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 93.3% | — | — |
| GPU Util (peak) | 100.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 36.54% |
| ROUGE-L | 29.36% |
| BLEU-4 | 15.37% |
| BERTScore F1 | 27.58% |
| Task Accuracy | 100.00% |
| Semantic Similarity | 27.58% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 71.88 | 2.64 | ±6.57 |
| 2 | 24.02 | 0.34 | ±0.83 |
| 4 | 43.06 | 0.27 | ±0.66 |
| 8 | 41.77 | 0.43 | ±1.07 |
| 16 | 40.52 | 0.33 | ±0.82 |

---

## 4. Discussion

The BitsAndBytes NF4 4-bit quantized model shows stable memory consumption with peak VRAM remaining under 3.2 GB. The generation quality metrics indicate a high task accuracy (100.00%) on this evaluation subset, and a BERTScore F1 of 27.58%. Under scaling, the throughput peaks at concurrency 1 at 71.88 tok/s and remains stable around 40-43 tok/s at higher concurrency levels.
