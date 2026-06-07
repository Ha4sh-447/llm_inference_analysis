#!/usr/bin/env python3
import subprocess
import time
import urllib.request
import json
import os
import shutil
import sys

# Define configs to run
configs = [
    {
        "name": "baseline_awq",
        "results_dir": "results/baseline_awq",
        "args": [
            "--model", "./models/qwen-1.5b-awq",
            "--served-model-name", "qwen-awq",
            "--quantization", "awq_marlin",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "2048",
            "--port", "8000"
        ]
    },
    {
        "name": "spec_N2",
        "results_dir": "results/spec_N2",
        "args": [
            "--model", "./models/qwen-1.5b-awq",
            "--served-model-name", "qwen-awq",
            "--quantization", "awq_marlin",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "2048",
            "--port", "8000",
            "--spec-method", "ngram",
            "--spec-tokens", "2",
            "--speculative-config", '{"prompt_lookup_max": 4, "prompt_lookup_min": 1}'
        ]
    },
    {
        "name": "spec_N5",
        "results_dir": "results/spec_N5",
        "args": [
            "--model", "./models/qwen-1.5b-awq",
            "--served-model-name", "qwen-awq",
            "--quantization", "awq_marlin",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "2048",
            "--port", "8000",
            "--spec-method", "ngram",
            "--spec-tokens", "5",
            "--speculative-config", '{"prompt_lookup_max": 4, "prompt_lookup_min": 1}'
        ]
    },
    {
        "name": "spec_N10",
        "results_dir": "results/spec_N10",
        "args": [
            "--model", "./models/qwen-1.5b-awq",
            "--served-model-name", "qwen-awq",
            "--quantization", "awq_marlin",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "2048",
            "--port", "8000",
            "--spec-method", "ngram",
            "--spec-tokens", "10",
            "--speculative-config", '{"prompt_lookup_max": 4, "prompt_lookup_min": 1}'
        ]
    },
    {
        "name": "spec_N20",
        "results_dir": "results/spec_N20",
        "args": [
            "--model", "./models/qwen-1.5b-awq",
            "--served-model-name", "qwen-awq",
            "--quantization", "awq_marlin",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "2048",
            "--port", "8000",
            "--spec-method", "ngram",
            "--spec-tokens", "20",
            "--speculative-config", '{"prompt_lookup_max": 4, "prompt_lookup_min": 1}'
        ]
    },
    {
        "name": "spec_N5_gpu",
        "results_dir": "results/spec_N5_gpu",
        "args": [
            "--model", "./models/qwen-1.5b-awq",
            "--served-model-name", "qwen-awq",
            "--quantization", "awq_marlin",
            "--dtype", "float16",
            "--gpu-memory-utilization", "0.95",
            "--max-model-len", "2048",
            "--port", "8000",
            "--spec-method", "ngram_gpu",
            "--spec-tokens", "5",
            "--speculative-config", '{"prompt_lookup_max": 4, "prompt_lookup_min": 1}'
        ]
    }
]

# Ensure we use venv python
venv_python = ".venv/bin/python"

def wait_for_server(url="http://localhost:8000/v1/models", timeout=300):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=5) as response:
                if response.status == 200:
                    print(f"Server is ready! (Took {time.time() - start_time:.1f}s)")
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False

def main():
    print("Starting automated execution of all AWQ & speculative decoding benchmarks...")
    
    for config in configs:
        print(f"\n=============================================================")
        print(f" RUNNING BENCHMARK FOR CONFIG: {config['name']}")
        print(f"=============================================================")
        
        # 1. Back up or delete existing performance.csv
        csv_path = os.path.join(config['results_dir'], "performance.csv")
        if os.path.exists(csv_path):
            backup_path = csv_path + ".bak"
            print(f"Backing up existing performance.csv to {backup_path}")
            shutil.copyfile(csv_path, backup_path)
            os.remove(csv_path)
        else:
            os.makedirs(config['results_dir'], exist_ok=True)

        # 2. Start vLLM server
        cmd = [venv_python, "-m", "vllm.entrypoints.openai.api_server"] + config['args']
        print(f"Starting vLLM server with command: {' '.join(cmd)}")
        log_file_path = os.path.join(config['results_dir'], "vllm_server.log")
        
        with open(log_file_path, "w") as log_file:
            server_process = subprocess.Popen(cmd, stdout=log_file, stderr=log_file)
            
            try:
                # 3. Wait for server to start
                print("Waiting for server to become healthy (polling port 8000)...")
                if not wait_for_server():
                    print(f"ERROR: Server failed to start within timeout. Check logs at {log_file_path}")
                    # Dump the last 20 lines of server log to stdout for quick debugging
                    try:
                        with open(log_file_path, "r") as f:
                            lines = f.readlines()
                            print(f"\nLast 20 lines of {log_file_path}:")
                            for line in lines[-20:]:
                                print(line, end="")
                    except Exception:
                        pass
                    server_process.terminate()
                    server_process.wait()
                    sys.exit(1)
                    
                # 4. Run benchmarking script
                bench_cmd = [
                    venv_python, "benchmark/run_performance.py",
                    "--model", "awq",
                    "--engine", "vllm",
                    "--results-dir", config['results_dir']
                ]
                print(f"Running benchmark command: {' '.join(bench_cmd)}")
                subprocess.run(bench_cmd, check=True)
                print(f"Successfully finished benchmark for config {config['name']}")
                
            except Exception as e:
                print(f"ERROR during benchmarking config {config['name']}: {e}")
                # Dump server log if error
                try:
                    with open(log_file_path, "r") as f:
                        lines = f.readlines()
                        print(f"\nLast 20 lines of {log_file_path}:")
                        for line in lines[-20:]:
                            print(line, end="")
                except Exception:
                    pass
                server_process.terminate()
                server_process.wait()
                sys.exit(1)
                
            finally:
                # 5. Clean up server process
                print("Stopping vLLM server...")
                server_process.terminate()
                try:
                    server_process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    print("Force killing server process...")
                    server_process.kill()
                    server_process.wait()
                
                # Give some time to completely free the GPU memory and clean up sockets
                print("Waiting 5s for GPU memory cleanup...")
                time.sleep(5)

    print("\nAll AWQ and speculative decoding benchmarks finished successfully!")

if __name__ == "__main__":
    main()
