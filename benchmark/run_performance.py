#!/usr/bin/env python3
"""
benchmark/run_performance.py

Phase 2: Serving Performance Evaluation.
Measures TTFT, ITL, Throughput, Peak VRAM, and GPU utilization under Concurrency=4 over repetitions.
Computes Mean, Std Dev, and 95% Confidence Intervals, saving to results/performance.csv.
"""

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import time
import yaml
import threading
import subprocess
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

SYSTEM_PROMPT = "You are an educational assistant generating questions aligned with Bloom's Taxonomy."

def format_chatml(passage: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{passage}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

def compute_ci95(data):
    n = len(data)
    if n < 2:
        return 0.0
    mean = np.mean(data)
    std_err = np.std(data, ddof=1) / math.sqrt(n)
    t_table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571
    }
    t_val = t_table.get(n, 1.96)
    return t_val * std_err

# ── GPU Systems Tracker ───────────────────────────────────────────────────────

class GPUTracker:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.peak_vram = 0.0
        self.gpu_utils = []
        self.running = False
        self.thread = None

    def _track(self):
        while self.running:
            try:
                cmd = "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,nounits,noheader"
                output = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
                vram_str, util_str = output.split(",")
                vram = float(vram_str.strip()) / 1024.0
                util = float(util_str.strip())
                
                if vram > self.peak_vram:
                    self.peak_vram = vram
                self.gpu_utils.append(util)
            except Exception:
                pass
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.peak_vram = 0.0
        self.gpu_utils = []
        self.thread = threading.Thread(target=self._track, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        avg_util = sum(self.gpu_utils) / len(self.gpu_utils) if self.gpu_utils else 0.0
        peak_util = max(self.gpu_utils) if self.gpu_utils else 0.0
        # Fallback to torch if nvidia-smi failed entirely
        if self.peak_vram == 0.0 and torch.cuda.is_available():
            self.peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
        return self.peak_vram, avg_util, peak_util

# ── API Streaming Concurrency ─────────────────────────────────────────────────

def get_api_tokenizer(model_path):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_path)
    except Exception:
        return None

async def send_api_request_stream(session, url, model, prompt, max_tokens=128):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True
    }
    ttft = None
    start = time.perf_counter()
    timestamps = []
    text_pieces = []
    
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return None
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0]['delta']
                            if 'content' in delta and delta['content']:
                                text_pieces.append(delta['content'])
                                if ttft is None:
                                    ttft = (time.perf_counter() - start) * 1000
                                timestamps.append(time.perf_counter())
                    except Exception:
                        continue
            end = time.perf_counter()
            duration = end - start
            return {
                "ttft_ms": ttft or (duration * 1000),
                "duration_s": duration,
                "timestamps": timestamps,
                "generated_text": "".join(text_pieces),
                "prompt": prompt,
            }
    except Exception:
        return None

async def resolve_served_model_name(api_url, model_name, model_path):
    import aiohttp
    resolved_model_name = model_name
    try:
        if "/v1/chat/completions" in api_url:
            models_url = api_url.replace("/v1/chat/completions", "/v1/models")
        elif "/v1/completions" in api_url:
            models_url = api_url.replace("/v1/completions", "/v1/models")
        else:
            models_url = api_url.rsplit('/', 1)[0] + '/models'
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url) as resp:
                if resp.status == 200:
                    models_data = await resp.json()
                    if "data" in models_data and models_data["data"]:
                        available_models = [m["id"] for m in models_data["data"]]
                        if model_name in available_models:
                            resolved_model_name = model_name
                        elif model_path in available_models:
                            resolved_model_name = model_path
                        else:
                            matched = None
                            for am in available_models:
                                if model_name.lower() in am.lower() or am.lower() in model_name.lower():
                                    matched = am
                                    break
                            if matched:
                                resolved_model_name = matched
                            else:
                                resolved_model_name = available_models[0]
                        print(f"Auto-resolved served model name to: {resolved_model_name}")
    except Exception as e:
        print(f"Could not auto-resolve served model name, using configured '{model_name}': {e}")
    return resolved_model_name

async def run_api_perf(api_url, model_name, model_path, prompts, concurrency=4, max_tokens=128):
    import aiohttp
    
    # Auto-resolve served model name from the server's /v1/models endpoint
    resolved_model_name = await resolve_served_model_name(api_url, model_name, model_path)

    tracker = GPUTracker()
    tracker.start()

    sem = asyncio.Semaphore(concurrency)
    
    async def sem_worker(session, prompt):
        async with sem:
            return await send_api_request_stream(session, api_url, resolved_model_name, prompt, max_tokens)
            
    async with aiohttp.ClientSession() as session:
        # Warmup request to trigger JIT compilation/graph setup on the server
        print("Warming up API server...")
        _ = await send_api_request_stream(session, api_url, resolved_model_name, "Warmup prompt", max_tokens=max_tokens)
        
        tasks = [sem_worker(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        
    valid_res = [r for r in results if r is not None]
    if not valid_res:
        peak_vram, avg_util, peak_util = tracker.stop()
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, peak_vram, avg_util, peak_util
        
    tokenizer = get_api_tokenizer(model_path)
    
    tps = []
    ttfts = []
    itls = []
    prefill_latencies = []
    prefill_tps = []

    decode_latencies = []
    decode_tps = []
    
    for r in valid_res:
        text = r["generated_text"]
        if tokenizer:
            output_tokens = len(tokenizer.encode(text))
            input_tokens = len(tokenizer.encode(r["prompt"]))
        else:
            output_tokens = int(len(text.split()) * 1.3)
            input_tokens = int(len(r["prompt"].split()) * 1.3)

        n_tok = output_tokens        
        duration = r["duration_s"]
        ttft = r["ttft_ms"]
        duration_ms = duration * 1000

        prefill_latency = ttft

        decode_latency = max(
            duration_ms - ttft,
            1e-6
        )

        prefill_throughput = (
            input_tokens /
            (prefill_latency / 1000)
        )

        decode_throughput = (
            output_tokens /
            (decode_latency / 1000)
        )

        prefill_latencies.append(prefill_latency)
        prefill_tps.append(prefill_throughput)

        decode_latencies.append(decode_latency)
        decode_tps.append(decode_throughput)
        # Calculate ITL: avg inter-token latency between consecutive chunks
        # Using chunk timestamps is correct here since each SSE chunk ~ 1 token
        # from vLLM/SGLang (they emit one token per chunk by default).
        # Divide by (chunks-1) intervals, not (n_tok-1), to avoid mismatch.
        timestamps = r["timestamps"]
        n_chunks = len(timestamps)
        if n_chunks > 1:
            stream_duration_ms = (timestamps[-1] - timestamps[0]) * 1000
            itl = stream_duration_ms / (n_chunks - 1)
        else:
            itl = 0.0
            
        tp = n_tok / duration if duration > 0 else 0.0
        
        tps.append(tp)
        ttfts.append(ttft)
        itls.append(itl)
        
    peak_vram, avg_util, peak_util = tracker.stop()
    return (
        np.mean(tps),
        np.mean(ttfts),
        np.mean(itls),

        np.mean(prefill_latencies),
        np.mean(prefill_tps),

        np.mean(decode_latencies),
        np.mean(decode_tps),

        peak_vram,
        avg_util,
        peak_util,
    )

# ── Main Run ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Performance Evaluation.")
    parser.add_argument("--config", type=str, default="benchmark/config.yaml")
    parser.add_argument("--model", type=str, default="awq", choices=["fp16", "awq", "nf4", "aqlm", "fp8", "bnbint8"])
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to write output CSV (default: results)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    val_path = config["dataset"]["val_path"]
    num_perf_samples = config["dataset"]["num_samples_performance"]
    concurrency = config["workload"]["concurrency_performance"]
    max_tokens = config["workload"]["max_tokens"]
    mixed_length = config["workload"].get("mixed_length", False)
    repetitions = config["statistical"]["repetitions"]

    if not os.path.exists(val_path):
        print(f"Dataset not found at: {val_path}")
        sys.exit(1)
        
    with open(val_path, "r") as f:
        val_data = json.load(f)[:num_perf_samples]
    prompts = [x["input_text"] for x in val_data]

    model_conf = config["models"][args.model]
    
    throughputs = []
    ttfts = []
    itls = []
    vrams = []
    gpu_utils_avg = []
    gpu_utils_peak = []

    prefill_latency_runs = []
    prefill_tps_runs = []
    decode_latency_runs = []
    decode_tps_runs = []

    print(f"Starting performance testing for {args.model} on {args.engine}...")
    print(f"Workload: Concurrency={concurrency}, Mixed Length={mixed_length}, Repetitions={repetitions}")

    for r in range(repetitions):
        print(f"\n--- Repetition {r+1}/{repetitions} ---")
        engine_conf = config["engines"][args.engine]
        (
            tp,
            ttft,
            itl,

            prefill_latency,
            prefill_tps,

            decode_latency,
            decode_tps,

            vram,
            avg_u,
            peak_u,
        ) = asyncio.run(run_api_perf(
            engine_conf["api_url"], model_conf.get("served_model_name", model_conf["path"]), model_conf["path"], prompts, concurrency, max_tokens
        ))
        throughputs.append(tp)
        ttfts.append(ttft)
        itls.append(itl)
        vrams.append(vram)
        gpu_utils_avg.append(avg_u)
        gpu_utils_peak.append(peak_u)

        prefill_latency_runs.append(prefill_latency)
        prefill_tps_runs.append(prefill_tps)
        decode_latency_runs.append(decode_latency)
        decode_tps_runs.append(decode_tps)

        print(f"Rep {r+1} Complete | Throughput: {tp:.2f} tok/s | TTFT: {ttft:.1f}ms | ITL: {itl:.1f}ms | Peak VRAM: {vram:.2f}GB | GPU Util: {avg_u:.1f}%")

    # Statistical summaries
    mean_tp, std_tp, ci_tp = np.mean(throughputs), np.std(throughputs, ddof=1) if repetitions > 1 else 0.0, compute_ci95(throughputs)
    mean_ttft, std_ttft, ci_ttft = np.mean(ttfts), np.std(ttfts, ddof=1) if repetitions > 1 else 0.0, compute_ci95(ttfts)
    mean_itl, std_itl, ci_itl = np.mean(itls), np.std(itls, ddof=1) if repetitions > 1 else 0.0, compute_ci95(itls)
    mean_vram, std_vram, ci_vram = np.mean(vrams), np.std(vrams, ddof=1) if repetitions > 1 else 0.0, compute_ci95(vrams)
    mean_u_avg = np.mean(gpu_utils_avg)
    mean_u_peak = np.mean(gpu_utils_peak)

    mean_prefill_latency = np.mean(prefill_latency_runs)
    mean_prefill_tps = np.mean(prefill_tps_runs)
    mean_decode_latency = np.mean(decode_latency_runs)
    mean_decode_tps = np.mean(decode_tps_runs)

    # Save to CSV
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "performance.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if not file_exists:
            writer.writerow([
                "model", "engine", "concurrency", "mixed_length",
                "throughput_mean", "throughput_std", "throughput_ci95",
                "ttft_mean", "ttft_std", "ttft_ci95",
                "itl_mean", "itl_std", "itl_ci95",
                "prefill_latency_ms", "prefill_tps",
                "decode_latency_ms", "decode_tps",
                "vram_peak_mean", "vram_peak_std", "vram_peak_ci95",
                "gpu_util_avg", "gpu_util_peak"
            ])
        writer.writerow([
            args.model, args.engine, concurrency, mixed_length,
            mean_tp, std_tp, ci_tp,
            mean_ttft, std_ttft, ci_ttft,
            mean_itl, std_itl, ci_itl,
            mean_prefill_latency, mean_prefill_tps,
            mean_decode_latency, mean_decode_tps,
            mean_vram, std_vram, ci_vram,
            mean_u_avg, mean_u_peak
        ])

    print("\n" + "=" * 60)
    print(f"PERFORMANCE SUMMARY FOR Model: {args.model} | Engine: {args.engine}")
    print("=" * 60)
    print(f"Throughput (tok/s): {mean_tp:.2f} ± {ci_tp:.2f} (StdDev: {std_tp:.2f})")
    print(f"Avg TTFT (ms):      {mean_ttft:.1f} ± {ci_ttft:.1f} (StdDev: {std_ttft:.2f})")
    print(f"Avg ITL (ms):       {mean_itl:.1f} ± {ci_itl:.1f} (StdDev: {std_itl:.2f})")

    print(f"Prefill Latency:    {mean_prefill_latency:.1f} ms")
    print(f"Prefill Throughput: {mean_prefill_tps:.2f} tok/s")

    print(f"Decode Latency:     {mean_decode_latency:.1f} ms")
    print(f"Decode Throughput:  {mean_decode_tps:.2f} tok/s")

    print(f"Peak VRAM:          {mean_vram:.2f} ± {ci_vram:.2f} GB (StdDev: {std_vram:.2f})")
    print(f"GPU Core Util:      Avg: {mean_u_avg:.1f}%, Peak: {mean_u_peak:.1f}%")
    print(f"Results written to: {csv_path}")

if __name__ == "__main__":
    main()
