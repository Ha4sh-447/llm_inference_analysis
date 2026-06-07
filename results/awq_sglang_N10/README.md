# Benchmark Results — AWQ + N-Gram Spec (N=10, SGLang)

**Model:** Qwen2.5-1.5B AWQ (4-bit, Marlin)
**Engine:** SGLang 0.5.12.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** N-gram, `speculative-algorithm=NGRAM`, `speculative-num-draft-tokens=10`
**Date:** 2026-06-06

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 197.56 tok/s | 127.11 | ±315.77 |
| TTFT | 46.3 ms | 37.13 ms | ±92.3 ms |
| ITL | 21.3 ms/tok | 1.85 ms | ±4.6 ms |
| Prefill Latency | 46.3 ms | — | — |
| Prefill Throughput | 5994.16 tok/s | — | — |
| Decode Latency | 401.8 ms | — | — |
| Decode Throughput | 248.72 tok/s | — | — |
| Peak VRAM | 3.64 GB | 0.02 | ±0.06 |
| GPU Util (avg) | 63.9% | — | — |
| GPU Util (peak) | 90.3% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 25.17% |
| ROUGE-L | 21.00% |
| BLEU-4 | 8.39% |
| BERTScore F1 | 18.26% |
| Task Accuracy | 68.00% |
| Semantic Similarity | 18.26% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 627.02 | 60.19 | ±149.53 |
| 2 | 689.42 | 105.98 | ±263.29 |
| 4 | 909.58 | 103.47 | ±257.06 |
| 8 | 1251.44 | 147.35 | ±366.06 |
| 16 | OOM | — | — |

*Note: At concurrency level 16, the engine reported an Out of Memory (OOM) error due to VRAM limits on the 4 GB GPU.*

---

## 4. Engine Comparison: vLLM vs SGLang (AWQ + N-Gram Spec N=10)

| Metric | vLLM (Spec N=10) | SGLang (Spec N=10) | Delta |
|--------|:----------------:|:------------------:|:-----:|
| Throughput (tok/s) | 91.66 | 197.56 | **+115.5%** |
| TTFT (ms) | 52.0 | 46.3 | **−11.0%** |
| ITL (ms/tok) | 14.0 | 21.3 | **+52.1%** |
| Prefill Latency (ms) | 52.0 | 46.3 | **−11.0%** |
| Prefill Throughput (tok/s) | 5275.31 | 5994.16 | **+13.6%** |
| Decode Latency (ms) | 486.0 | 401.8 | **−17.3%** |
| Decode Throughput (tok/s) | 101.63 | 248.72 | **+144.7%** |
| Peak VRAM (GB) | 3.40 | 3.64 | **+7.1%** |

SGLang with N-gram speculative decoding (N=10) provides a substantial throughput benefit (+115.5%) and a 144.7% higher decode throughput than vLLM under the same configuration. However, this comes at the expense of a 52.1% higher inter-token latency (21.3 ms/tok vs 14.0 ms/tok) and higher peak VRAM usage (+7.1%). In addition, while vLLM handles concurrency 16 successfully, SGLang encounters an Out of Memory (OOM) failure at this level.
