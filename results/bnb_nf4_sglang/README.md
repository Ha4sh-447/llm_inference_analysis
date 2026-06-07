# Benchmark Results — BitsAndBytes NF4 (SGLang)

**Model:** Qwen-1.5B BitsAndBytes NF4
**Engine:** SGLang 0.5.12.post1
**GPU:** RTX 3050 Laptop (4 GB VRAM)
**Spec Decode:** None (baseline)
**Date:** 2026-06-06

---

## Server Command

The SGLang server was launched using the following command:
```bash
python -m sglang.launch_server \
  --model-path ./models/qwen-1.5b-bnb-nf4 \
  --quantization bitsandbytes \
  --load-format bitsandbytes \
  --host 0.0.0.0 \
  --port 30000 \
  --mem-fraction-static 0.70 \
  --context-length 2048 \
  --attention-backend triton \
  --sampling-backend pytorch \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph
```

---

## 1. Performance (Concurrency=4, 3 reps)

| Metric | Mean | StdDev | 95% CI |
|--------|------|--------|--------|
| Throughput | 11.51 tok/s | 0.42 | ±1.05 |
| TTFT | 228.5 ms | 82.24 ms | ±204.3 ms |
| ITL | 82.5 ms/tok | 1.57 ms | ±3.9 ms |
| Prefill Latency | 228.5 ms | — | — |
| Prefill Throughput | 929.20 tok/s | — | — |
| Decode Latency | 4714.2 ms | — | — |
| Decode Throughput | 12.15 tok/s | — | — |
| Peak VRAM | 2.89 GB | 0.00 | ±0.00 |
| GPU Util (avg) | 89.0% | — | — |
| GPU Util (peak) | 100.0% | — | — |

---

## 2. Quality (50 prompts, concurrency=1)

| Metric | Score |
|--------|-------|
| ROUGE-1 | 36.20% |
| ROUGE-L | 29.06% |
| BLEU-4 | 15.79% |
| BERTScore F1 | 64.34% |
| Task Accuracy | 100.00% |
| Semantic Similarity | 64.34% |

---

## 3. Concurrency Scaling

| Concurrency | Throughput (tok/s) | StdDev | 95% CI |
|:-----------:|:------------------:|:------:|:-------:|
| 1 | 31.13 | 0.64 | ±1.59 |
| 2 | 23.35 | 0.06 | ±0.16 |
| 4 | 40.76 | 0.11 | ±0.27 |
| 8 | 68.42 | 0.02 | ±0.04 |
| 16 | 94.58 | 0.26 | ±0.65 |

---

## 4. Engine Comparison: vLLM vs SGLang (BitsAndBytes NF4)

| Metric | vLLM (NF4) | SGLang (NF4) | Delta |
|--------|:----------:|:------------:|:-----:|
| Throughput (C=4) (tok/s) | 12.84 | 11.51 | **−10.4%** |
| TTFT (ms) | 224.4 | 228.5 | **+1.8%** |
| ITL (ms/tok) | 76.7 | 82.5 | **+7.6%** |
| Peak VRAM (GB) | 3.19 | 2.89 | **−9.4%** |
| Scaling C=1 (tok/s) | 71.88 | 31.13 | **−56.7%** |
| Scaling C=16 (tok/s) | 40.52 | 94.58 | **+133.4%** |

While vLLM shows slightly better latency and single-concurrency throughput, SGLang operates with significantly lower peak VRAM usage (2.89 GB vs 3.19 GB). This memory efficiency allows SGLang to scale much better under load, outperforming vLLM at concurrency 16 by 133.4% (94.58 tok/s vs 40.52 tok/s).
