# Benchmark Results — AWQ + N-Gram Spec (N=5, SGLang)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** SGLang 0.5.12.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `speculative-algorithm=NGRAM`, `speculative-num-draft-tokens=5`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 159.23 tok/s | 86.05 | ±213.78 |
| TTFT | 44.8 ms | 38.41 ms | ±95.4 ms |
| ITL | 17.4 ms/tok | 1.22 ms | ±3.0 ms |
| Prefill Latency | 44.8 ms | — | — |
| Prefill Throughput | 6491.21 tok/s | — | — |
| Decode Latency | 376.0 ms | — | — |
| Decode Throughput | 183.67 tok/s | — | — |
| Peak VRAM | 3.57 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 57.3% | — | — |
| GPU Util (peak) | 88.7% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 24.14% |
| ROUGE-L | 20.46% |
| BLEU-4 | 8.34% |
| BERTScore F1 | 17.68% |
| Task Accuracy | 68.00% |
| Semantic Similarity | 17.68% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 399.47 | 17.75 | ±44.11 |
| 2 | 641.99 | 0.69 | ±1.71 |
| 4 | 782.94 | 93.94 | ±233.37 |
| 8 | 965.48 | 190.99 | ±474.49 |
| 16 | OOM | — | — |

*Note: At concurrency level 16, the engine reported an Out of Memory (OOM) error due to VRAM limits on the 4 GB GPU.*

---

## 4. Engine Comparison: vLLM vs SGLang (AWQ + N-Gram Spec N=5)

| Metric | vLLM (Spec N=5) | SGLang (Spec N=5) | Delta |
|--------|:---------------:|:-----------------:|:-----:|
| Throughput (tok/s) | 90.31 | 159.23 | **+76.3%** |
| TTFT (ms) | 50.1 | 44.8 | **−10.6%** |
| ITL (ms/tok) | 13.5 | 17.4 | **+28.9%** |
| Prefill Latency (ms) | 50.1 | 44.8 | **−10.6%** |
| Prefill Throughput (tok/s) | 5376.19 | 6491.21 | **+20.7%** |
| Decode Latency (ms) | 474.6 | 376.0 | **−20.8%** |
| Decode Throughput (tok/s) | 100.52 | 183.67 | **+82.7%** |
| Peak VRAM (GB) | 3.61 | 3.57 | **−1.1%** |

SGLang with N-gram speculative decoding (N=5) shows a significant throughput advantage (+76.3%) and a 20.8% lower decode latency over vLLM with the same speculative setup. However, it exhibits a higher inter-token latency (ITL) of 17.4 ms/tok compared to vLLM's 13.5 ms/tok. Under scaling, SGLang achieves extremely high throughput at lower concurrency levels, but experiences Out of Memory (OOM) at concurrency 16, whereas vLLM completes the concurrency 16 workload successfully (667.71 tok/s) within the 4 GB VRAM limit.
