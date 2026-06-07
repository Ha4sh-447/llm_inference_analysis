#!/usr/bin/env python3
"""
benchmark/plotting.py

Visualization module for the benchmarking suite.
Generates research-grade charts:
1. Quality vs Throughput (scatter plot)
2. Memory vs Quality (scatter plot)
3. Throughput vs Concurrency (scaling curves per engine)

Outputs saved to results/figures/
"""

import os
import csv
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

def load_quality_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Quality results not found at: {csv_path}")
        return {}
    
    # Group by (model, engine)
    scores = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["engine"])
            # We use BERTScore as the primary quality metric
            scores[key].append(float(row["bert_score"]))
            
    # Calculate average
    avg_scores = {}
    for key, vals in scores.items():
        avg_scores[key] = sum(vals) / len(vals)
    return avg_scores

def load_performance_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Performance results not found at: {csv_path}")
        return {}
        
    perf = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["engine"])
            perf[key] = {
                "throughput": float(row["throughput_mean"]),
                "ttft": float(row["ttft_mean"]),
                "itl": float(row["itl_mean"]),
                "vram": float(row.get("vram_peak_mean", row.get("vram_mean", 0.0)))
            }
    return perf

def load_scaling_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"Scaling results not found at: {csv_path}")
        return {}
        
    scaling = defaultdict(list)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["model"], row["engine"])
            scaling[key].append((int(row["concurrency"]), float(row["throughput_mean"])))
            
    # Sort scaling data by concurrency level
    for key in scaling:
        scaling[key].sort(key=lambda x: x[0])
    return scaling

def plot_quality_vs_throughput(quality_data, perf_data, out_dir):
    plt.figure(figsize=(8, 6))
    
    # We find keys present in both datasets
    keys = set(quality_data.keys()).intersection(set(perf_data.keys()))
    if not keys:
        print("No overlapping data found for Quality vs Throughput plot.")
        plt.close()
        return
        
    for model, engine in keys:
        throughput = perf_data[(model, engine)]["throughput"]
        quality = quality_data[(model, engine)] * 100  # Convert to %
        
        label = f"{model.upper()} ({engine.upper()})"
        plt.scatter(throughput, quality, s=120, alpha=0.8, label=label, edgecolors='black')
        plt.text(throughput + 2, quality, label, fontsize=9, va='center')
        
    plt.xlabel("Throughput (tokens/sec)", fontsize=11, fontweight='bold')
    plt.ylabel("Quality (BERTScore %)", fontsize=11, fontweight='bold')
    plt.title("Quality vs. Throughput Tradeoff", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "quality_vs_throughput.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Quality vs Throughput plot to: {out_path}")

def plot_memory_vs_quality(quality_data, perf_data, out_dir):
    plt.figure(figsize=(8, 6))
    
    keys = set(quality_data.keys()).intersection(set(perf_data.keys()))
    if not keys:
        print("No overlapping data found for Memory vs Quality plot.")
        plt.close()
        return
        
    for model, engine in keys:
        vram = perf_data[(model, engine)]["vram"]
        quality = quality_data[(model, engine)] * 100
        
        label = f"{model.upper()} ({engine.upper()})"
        plt.scatter(vram, quality, s=120, alpha=0.8, label=label, edgecolors='black')
        plt.text(vram + 0.1, quality, label, fontsize=9, va='center')
        
    plt.xlabel("VRAM Usage (GB)", fontsize=11, fontweight='bold')
    plt.ylabel("Quality (BERTScore %)", fontsize=11, fontweight='bold')
    plt.title("Memory vs. Quality Tradeoff", fontsize=13, fontweight='bold', pad=15)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, "memory_vs_quality.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Memory vs Quality plot to: {out_path}")

def plot_throughput_vs_concurrency(scaling_data, out_dir):
    # Group by engine to plot one graph per engine
    engine_data = defaultdict(dict)
    for (model, engine), curve in scaling_data.items():
        engine_data[engine][model] = curve
        
    if not engine_data:
        print("No scaling data found.")
        return
        
    for engine, models_curve in engine_data.items():
        plt.figure(figsize=(8, 6))
        
        for model, curve in models_curve.items():
            concurrencies = [x[0] for x in curve]
            throughputs = [x[1] for x in curve]
            
            plt.plot(concurrencies, throughputs, marker='o', linewidth=2.5, label=model.upper())
            
        plt.xlabel("Concurrency (Parallel Requests)", fontsize=11, fontweight='bold')
        plt.ylabel("System Throughput (tokens/sec)", fontsize=11, fontweight='bold')
        plt.title(f"Throughput Scaling: {engine.upper()} Engine", fontsize=13, fontweight='bold', pad=15)
        plt.xticks(concurrencies)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        out_path = os.path.join(out_dir, f"throughput_vs_concurrency_{engine}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved Throughput vs Concurrency scaling plot to: {out_path}")

def main():
    results_dir = "results"
    figures_dir = os.path.join(results_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    quality_path = os.path.join(results_dir, "quality.csv")
    perf_path = os.path.join(results_dir, "performance.csv")
    scaling_path = os.path.join(results_dir, "scaling.csv")
    
    quality_data = load_quality_data(quality_path)
    perf_data = load_performance_data(perf_path)
    scaling_data = load_scaling_data(scaling_path)
    
    plot_quality_vs_throughput(quality_data, perf_data, figures_dir)
    plot_memory_vs_quality(quality_data, perf_data, figures_dir)
    plot_throughput_vs_concurrency(scaling_data, figures_dir)

if __name__ == "__main__":
    main()
