# Benchmark Results — AWQ Baseline (No Speculative Decoding)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** vLLM 0.6.3.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** None (baseline)
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 90.57 tok/s | 18.89 | ±46.93 |
| TTFT | 65.9 ms | 53.95 ms | ±134.0 ms |
| ITL | 8.7 ms/tok | 1.37 ms | ±3.4 ms |
| Prefill Latency | 65.9 ms | — | — |
| Prefill Throughput | 4210.68 tok/s | — | — |
| Decode Latency | 406.3 ms | — | — |
| Decode Throughput | 104.87 tok/s | — | — |
| Peak VRAM | 3.58 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 73.4% | — | — |
| GPU Util (peak) | 100.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.66% |
| ROUGE-L | 20.81% |
| BLEU-4 | 8.53% |
| BERTScore F1 | 18.03% |
| Task Accuracy | 66.00% |
| Semantic Similarity | 18.03% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 125.28 | 0.71 | ±1.75 |
| 2 | 236.73 | 0.79 | ±1.95 |
| 4 | 378.53 | 1.56 | ±3.88 |
| 8 | 562.58 | 2.37 | ±5.90 |
| 16 | 666.99 | 8.79 | ±21.84 |

---

## 4. Baseline vs Spec N=5 Comparison

| Metric | Baseline (no spec) | Spec N=5 | Delta |
|--------|:------------------:|:--------:|:-----:|
| Throughput (tok/s) | 90.57 | 66.01 | **−27%** |
| TTFT (ms) | 65.9 | 70.9 | **+8%** |
| ITL (ms/tok) | 8.7 | 18.6 | **+114%** |
| Prefill Latency (ms) | 65.9 | 66.5 | **+1%** |
| Prefill Throughput (tok/s) | 4210.68 | 3872.58 | **−8%** |
| Decode Latency (ms) | 406.3 | 612.2 | **+51%** |
| Decode Throughput (tok/s) | 104.87 | 80.27 | **−23%** |
| Peak VRAM (GB) | 3.58 | 3.28 | **−8%** |

N-gram speculative decoding **hurts** throughput on this AWQ model — baseline is 1.4× faster. This is expected for small models on latency-sensitive workloads where the draft verification overhead outweighs the savings.
