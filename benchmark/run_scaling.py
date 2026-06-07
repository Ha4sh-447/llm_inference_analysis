#!/usr/bin/env python3
"""
benchmark/run_scaling.py

Phase 3: Concurrency Scaling Evaluation.
Varies concurrency (1, 2, 4, 8, 16) and measures throughput (tokens/sec) over multiple repetitions.
Computes Mean, Std Dev, and 95% Confidence Intervals, saving to results/scaling.csv.
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
import numpy as np

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

# ── SGLang API Generation ─────────────────────────────────────────────────────

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
    text_pieces = []
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return ""
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    try:
                        data = json.loads(line[6:])
                        if 'choices' in data and data['choices']:
                            delta = data['choices'][0]['delta']
                            if 'content' in delta and delta['content']:
                                text_pieces.append(delta['content'])
                    except Exception:
                        continue
            return "".join(text_pieces)
    except Exception:
        return ""

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

async def run_api_scaling(api_url, model_name, model_path, prompts, concurrency, max_tokens=128):
    import aiohttp
    sem = asyncio.Semaphore(concurrency)
    
    async def sem_worker(session, prompt):
        async with sem:
            return await send_api_request_stream(session, api_url, model_name, prompt, max_tokens)
            
    async with aiohttp.ClientSession() as session:
        # Warmup request to trigger JIT compilation/graph setup on the server
        print("Warming up API server...")
        _ = await send_api_request_stream(session, api_url, model_name, "Warmup prompt", max_tokens=max_tokens)
        
        start_time = time.perf_counter()
        tasks = [sem_worker(session, p) for p in prompts]
        generated_texts = await asyncio.gather(*tasks)
        end_time = time.perf_counter()
        
    duration = end_time - start_time
    
    tokenizer = get_api_tokenizer(model_path)
    total_tokens = 0
    for text in generated_texts:
        if tokenizer:
            total_tokens += len(tokenizer.encode(text))
        else:
            total_tokens += int(len(text.split()) * 1.3)
            
    return total_tokens / duration if duration > 0 else 0.0

# ── Main Run ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Phase 3 Concurrency Scaling.")
    parser.add_argument("--config", type=str, default="benchmark/config.yaml")
    parser.add_argument("--model", type=str, default="awq", choices=["fp16", "awq", "nf4", "aqlm", "fp8"])
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to write output CSV (default: results)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    val_path = config["dataset"]["val_path"]
    num_perf_samples = config["dataset"]["num_samples_performance"]
    concurrency_levels = config["workload"]["concurrency_levels_scaling"]
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
    
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "scaling.csv")
    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if not file_exists:
            writer.writerow([
                "model", "engine", "concurrency", "mixed_length",
                "throughput_mean", "throughput_std", "throughput_ci95"
            ])

        print(f"Starting concurrency scaling testing for {args.model} on {args.engine}...")
        print(f"Concurrency Levels: {concurrency_levels} | Repetitions: {repetitions}")

        engine_conf = config["engines"][args.engine]
        model_name = model_conf.get("served_model_name", model_conf["path"])
        model_path = model_conf["path"]
        resolved_model_name = asyncio.run(resolve_served_model_name(engine_conf["api_url"], model_name, model_path))

        for c in concurrency_levels:
            print(f"\n>>> Concurrency Level: {c} <<<")
            throughputs = []
            
            for r in range(repetitions):
                tp = asyncio.run(run_api_scaling(
                    engine_conf["api_url"], resolved_model_name, model_conf["path"], prompts, c, max_tokens
                ))
                    
                throughputs.append(tp)
                print(f"  Rep {r+1}/{repetitions} Throughput: {tp:.2f} tok/s")
                
            mean_tp = np.mean(throughputs)
            std_tp = np.std(throughputs, ddof=1) if repetitions > 1 else 0.0
            ci_tp = compute_ci95(throughputs)
            
            writer.writerow([args.model, args.engine, c, mixed_length, mean_tp, std_tp, ci_tp])
            print(f"Summary for C={c}: Mean={mean_tp:.2f} ± {ci_tp:.2f} tok/s (StdDev: {std_tp:.2f})")

    print(f"\nAll scaling data written to: {csv_path}")

if __name__ == "__main__":
    main()
