#!/usr/bin/env python3
"""
benchmark/run_quality.py

Phase 1: Model Quality Evaluation.
Loads prompts from the validation dataset, generates outputs at concurrency=1,
computes ROUGE, BLEU, BERTScore, and Task Accuracy, and writes results to results/quality.csv.
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
import yaml
import numpy as np
import torch

# Add parent and benchmark dir to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from benchmark.metrics import (
    calculate_rouge_1,
    calculate_rouge_l,
    calculate_bleu_1,
    calculate_bleu_4,
    calculate_bertscore,
    calculate_task_accuracy,
    calculate_gpt_judge_score,
    calculate_semantic_similarity
)

SYSTEM_PROMPT = "You are an educational assistant generating questions aligned with Bloom's Taxonomy."

def format_chatml(passage: str) -> str:
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{passage}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ── Inference Methods ─────────────────────────────────────────────────────────

async def send_api_request(session, url, model, prompt, max_tokens=128):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens
    }
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                return ""
            data = await response.json()
            return data['choices'][0]['message']['content']
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

async def run_api_suite(api_url, model_name, prompts, max_tokens=128):
    import aiohttp
    print(f"Sending API requests to {api_url}...")
    async with aiohttp.ClientSession() as session:
        generations = []
        for idx, p in enumerate(prompts):
            gen = await send_api_request(session, api_url, model_name, p, max_tokens)
            generations.append(gen)
            print(f"Generated {idx+1}/{len(prompts)}")
        return generations

# ── Main Run ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Quality Evaluation.")
    parser.add_argument("--config", type=str, default="benchmark/config.yaml")
    parser.add_argument("--model", type=str, default="awq", choices=["fp16", "awq", "nf4", "aqlm", "fp8"])
    parser.add_argument("--engine", type=str, default="vllm", choices=["vllm", "sglang"])
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Directory to write output CSV (default: results)")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Load dataset
    val_path = config["dataset"]["val_path"]
    num_samples = config["dataset"]["num_samples_quality"]
    
    if not os.path.exists(val_path):
        print(f"Dataset not found at: {val_path}")
        sys.exit(1)
        
    with open(val_path, "r") as f:
        val_data = json.load(f)[:num_samples]
        
    prompts = [x["input_text"] for x in val_data]
    references = [x["target_text"] for x in val_data]
    
    model_conf = config["models"][args.model]
    max_tokens = config["workload"]["max_tokens"]
    
    # Run completions
    engine_conf = config["engines"][args.engine]
    model_name = model_conf.get("served_model_name", model_conf["path"])
    model_path = model_conf["path"]
    resolved_model_name = asyncio.run(resolve_served_model_name(engine_conf["api_url"], model_name, model_path))
    generations = asyncio.run(run_api_suite(engine_conf["api_url"], resolved_model_name, prompts, max_tokens))

    # Evaluate metrics
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "quality.csv")
    
    file_exists = os.path.exists(csv_path)
    
    judge_conf = config.get("judge", {})
    judge_enabled = judge_conf.get("enabled", False)
    
    rouge1_scores, rougel_scores = [], []
    bleu1_scores, bleu4_scores = [], []
    bert_scores, task_acc_scores = [], []
    gpt_scores = []
    semantic_scores = []
    
    with open(csv_path, "a", newline="", encoding="utf-8") as f_csv:
        writer = csv.writer(f_csv)
        if not file_exists:
            headers = [
                "model", "engine", "prompt_idx", "prompt", "reference", "generated",
                "rouge_1", "rouge_l", "bleu_1", "bleu_4", "bert_score", "task_accuracy", "semantic_similarity"
            ]
            if judge_enabled:
                headers.append("gpt_judge_score")
            writer.writerow(headers)
            
        for idx, (cand, ref) in enumerate(zip(generations, references)):
            r1 = calculate_rouge_1(cand, ref)
            rl = calculate_rouge_l(cand, ref)
            b1 = calculate_bleu_1(cand, ref)
            b4 = calculate_bleu_4(cand, ref)
            bs = calculate_bertscore(cand, ref)
            ta = calculate_task_accuracy(cand, ref)
            ss = calculate_semantic_similarity(cand, ref)
            
            gpt_score = 0.0
            if judge_enabled:
                gpt_score = calculate_gpt_judge_score(
                    cand, ref, prompts[idx],
                    judge_conf["api_url"], judge_conf["model_name"], judge_conf.get("api_key")
                )
                gpt_scores.append(gpt_score)
            
            rouge1_scores.append(r1)
            rougel_scores.append(rl)
            bleu1_scores.append(b1)
            bleu4_scores.append(b4)
            bert_scores.append(bs)
            task_acc_scores.append(ta)
            semantic_scores.append(ss)

            
            row = [
                args.model, args.engine, idx, prompts[idx], ref, cand,
                r1, rl, b1, b4, bs, ta, ss
            ]
            if judge_enabled:
                row.append(gpt_score)
            writer.writerow(row)
            
    print("\n" + "=" * 60)
    print(f"QUALITY RESULTS FOR Model: {args.model} | Engine: {args.engine}")
    print("=" * 60)
    print(f"  ROUGE-1:      {np.mean(rouge1_scores)*100:.2f}%")
    print(f"  ROUGE-L:      {np.mean(rougel_scores)*100:.2f}%")
    print(f"  BLEU-4:       {np.mean(bleu4_scores)*100:.2f}%")
    print(f"  BERTScore F1: {np.mean(bert_scores)*100:.2f}%")
    print(f"  Task Accuracy: {np.mean(task_acc_scores)*100:.2f}%")
    print(f"  Semantic Similarity: {np.mean(semantic_scores)*100:.2f}%")
    if judge_enabled:
        print(f"  GPT Judge Score (1-5): {np.mean(gpt_scores):.2f}")
    print(f"Detailed scores written to: {csv_path}")

if __name__ == "__main__":
    main()
