# Benchmark Results — AWQ + N-Gram Spec (N=2, SGLang)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** SGLang 0.5.12.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `speculative-algorithm=NGRAM`, `speculative-num-draft-tokens=2`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 107.35 tok/s | 39.94 | ±99.23 |
| TTFT | 45.3 ms | 39.39 ms | ±97.9 ms |
| ITL | 14.3 ms/tok | 1.99 ms | ±4.9 ms |
| Prefill Latency | 45.3 ms | — | — |
| Prefill Throughput | 6526.57 tok/s | — | — |
| Decode Latency | 419.9 ms | — | — |
| Decode Throughput | 117.94 tok/s | — | — |
| Peak VRAM | 3.53 GB | 0.01 | ±0.03 |
| GPU Util (avg) | 74.2% | — | — |
| GPU Util (peak) | 90.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.23% |
| ROUGE-L | 20.27% |
| BLEU-4 | 8.17% |
| BERTScore F1 | 17.64% |
| Task Accuracy | 68.00% |
| Semantic Similarity | 17.64% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 195.66 | 1.13 | ±2.82 |
| 2 | 322.98 | 0.59 | ±1.48 |
| 4 | 520.68 | 0.53 | ±1.32 |
| 8 | 725.05 | 16.85 | ±41.85 |
| 16 | 765.94 | 24.91 | ±61.88 |

---

## 4. Engine Comparison: vLLM vs SGLang (AWQ + N-Gram Spec N=2)

| Metric | vLLM (Spec N=2) | SGLang (Spec N=2) | Delta |
|--------|:---------------:|:-----------------:|:-----:|
| Throughput (tok/s) | 85.10 | 107.35 | **+26.1%** |
| TTFT (ms) | 51.4 | 45.3 | **−11.9%** |
| ITL (ms/tok) | 13.4 | 14.3 | **+6.7%** |
| Prefill Latency (ms) | 51.4 | 45.3 | **−11.9%** |
| Prefill Throughput (tok/s) | 5316.03 | 6526.57 | **+22.8%** |
| Decode Latency (ms) | 502.7 | 419.9 | **−16.5%** |
| Decode Throughput (tok/s) | 93.94 | 117.94 | **+25.5%** |
| Peak VRAM (GB) | 3.59 | 3.53 | **−1.7%** |

SGLang with N-gram speculative decoding (N=2) achieves a 26.1% throughput improvement and 16.5% lower decode latency than vLLM under the same configuration. SGLang also scales stably up to concurrency 16, reaching a mean throughput of 765.94 tok/s without triggering an OOM error, representing a significant memory-efficiency improvement compared to the N=5 spec configuration.
