"""
inference.py — Unified Inference Benchmarking Suite.

Supports 6 benchmark configurations:
  Case 1: HF Transformers + LoRA Adapter (NF4 dynamic loading)
  Case 2: HF Transformers Fully Merged Model (BF16 native)
  Case 3: HF Transformers AWQ 4-bit (Marlin kernels)
  Case 4: vLLM AWQ 4-bit (Marlin + CUDA Graphs)
  Case 5: vLLM BitsAndBytes 4-bit (In-flight quantization on Colab)
  Case 6: vLLM AWQ 4-bit + N-Gram Speculative Decoding (zero VRAM overhead)

Usage:
  python inference.py --case all
  python inference.py --case 1
"""

import argparse
import glob
import os
import subprocess
import sys
import time

# ── Shared Constants ─────────────────────────────────────────────────────────

BENCHMARK_PASSAGE = (
    "Photosynthesis is a system of biological processes by which photosynthetic "
    "organisms, such as most plants, algae, and cyanobacteria, convert light energy, "
    "typically from sunlight, into the chemical energy necessary to fuel their activities."
)

QUALITY_PROMPTS = [
    "Generate a Level 1 (Remember) Bloom's Taxonomy question based on the following passage:\n"
    "Photosynthesis is the process by which green plants convert sunlight, carbon dioxide, "
    "and water into glucose and oxygen. It primarily occurs in the chloroplasts of plant cells.",

    "Generate a Level 2 (Understand) Bloom's Taxonomy question based on the following passage:\n"
    "Mitochondria are known as the powerhouses of the cell because they generate ATP through "
    "cellular respiration. ATP serves as the primary energy currency for cellular activities.",

    "Generate a Level 2 (Understand) Bloom's Taxonomy question based on the following passage:\n"
    "The water cycle involves evaporation, condensation, precipitation, and collection. It "
    "continuously moves water through Earth's atmosphere, land, and oceans.",

    "Generate a Level 3 (Apply) Bloom's Taxonomy question based on the following passage:\n"
    "Newton's First Law states that an object remains at rest or in uniform motion unless "
    "acted upon by an external force. This principle is also called the law of inertia.",

    "Generate a Level 3 (Apply) Bloom's Taxonomy question based on the following passage:\n"
    "DNA contains genetic instructions used in the growth, development, functioning, and "
    "reproduction of living organisms. DNA is organized into chromosomes within the nucleus "
    "of eukaryotic cells.",

    "Generate a Level 4 (Analyze) Bloom's Taxonomy question based on the following passage:\n"
    "Cellular respiration and photosynthesis are complementary biological processes. "
    "Photosynthesis stores energy in glucose, while cellular respiration releases energy "
    "from glucose to produce ATP.",

    "Generate a Level 4 (Analyze) Bloom's Taxonomy question based on the following passage:\n"
    "An ecosystem consists of living organisms interacting with one another and with their "
    "physical environment. Energy flows through food chains, while nutrients cycle through "
    "ecosystems.",

    "Generate a Level 4 (Analyze) Bloom's Taxonomy question based on the following passage:\n"
    "Plate tectonics explains the movement of Earth's lithospheric plates. Interactions between "
    "plates can result in earthquakes, volcanic activity, and mountain formation.",

    "Generate a Level 3 (Apply) Bloom's Taxonomy question based on the following passage:\n"
    "Chemical reactions involve the transformation of reactants into products. According to the "
    "law of conservation of mass, matter is neither created nor destroyed during a chemical reaction.",

    "Generate a Level 4 (Analyze) Bloom's Taxonomy question based on the following passage:\n"
    "Natural selection is the process by which organisms with advantageous traits are more likely "
    "to survive and reproduce. Over many generations, this process can lead to evolutionary change "
    "within populations.",
]


def format_chatml(passage: str) -> str:
    """Wrap a passage into the Qwen ChatML prompt template."""
    return (
        "<|im_start|>system\n"
        "You are an educational assistant generating questions aligned with Bloom's Taxonomy.<|im_end|>\n"
        f"<|im_start|>user\n{passage}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def fix_cuda_paths():
    """Auto-discover CUDA library paths and re-exec with updated LD_LIBRARY_PATH.

    Needed for PyTorch 2.11+ CUDA 13 package resolution in Colab vLLM runs.
    """
    discovered = []
    site_dirs = {p for p in sys.path if os.path.isdir(p)}
    site_dirs.add("/usr/local/lib/python3.12/dist-packages")
    for sp in site_dirs:
        for lib_dir in sorted(glob.glob(os.path.join(sp, "nvidia", "*", "lib"))):
            if lib_dir not in discovered:
                discovered.append(lib_dir)

    for d in ["/usr/local/cuda/lib64"] + sorted(glob.glob("/usr/local/cuda-*/lib64")):
        if os.path.isdir(d) and d not in discovered:
            discovered.append(d)

    for d in ["/usr/lib/x86_64-linux-gnu", "/usr/lib64-nvidia",
              "/usr/local/nvidia/lib64", "/usr/local/nvidia/lib"]:
        if os.path.isdir(d) and d not in discovered:
            discovered.append(d)

    if not discovered:
        return

    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    if all(p in current_ld for p in discovered):
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(discovered) + (":" + current_ld if current_ld else "")
    print(f"[fix_cuda_paths] LD_LIBRARY_PATH set to:\n  {os.environ['LD_LIBRARY_PATH']}")
    print("[fix_cuda_paths] Re-executing script with updated library paths...")
    os.execv(sys.executable, [sys.executable] + sys.argv)


def get_model_size_gb(model_path: str) -> str:
    """Return the total size of all files in a model directory."""
    if not os.path.exists(model_path):
        return "N/A"
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(model_path)
        for f in fns
        if not os.path.islink(os.path.join(dp, f))
    )
    return f"{total / (1024 ** 3):.2f} GB"


class TimingStreamer:
    """Minimal streamer that captures the timestamp of the first generated token."""

    def __init__(self):
        self.first_token_time = None
        self._skip_first = True

    def put(self, value):
        if self._skip_first:
            self._skip_first = False
            return
        if self.first_token_time is None:
            self.first_token_time = time.time()

    def end(self):
        pass


# ── HF Transformers Benchmarking (Cases 1, 2, 3) ─────────────────────────────

def run_hf_case(model_path, use_4bit, peft_adapter_path, label, case_id):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    print(f"Model Disk Footprint: {get_model_size_gb(model_path)}")

    # Check existence
    if not os.path.exists(model_path) and not model_path.startswith("Qwen/"):
        print(f"[SKIPPED] Model path '{model_path}' not found.")
        return

    if peft_adapter_path and not os.path.exists(peft_adapter_path):
        print(f"[SKIPPED] Adapter path '{peft_adapter_path}' not found.")
        return

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"device_map": "cuda:0"}
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if peft_adapter_path:
        model = PeftModel.from_pretrained(model, peft_adapter_path)

    # 1 Warmup + 5 Measured Runs
    prompt = format_chatml(f"generate a level 2 question testing comprehension: {BENCHMARK_PASSAGE}")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

    total_throughput = 0.0
    total_duration = 0.0
    total_ttft = 0.0
    total_itl = 0.0
    generated_text = ""
    tokens_generated = 0
    num_runs = 6

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print(f"\nRunning deterministic benchmark (1 warmup + 5 measured)...")
    for i in range(num_runs):
        streamer = TimingStreamer()
        start = time.time()

        outputs = model.generate(
            **inputs,
            min_new_tokens=128,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

        end = time.time()
        duration = end - start
        input_len = inputs["input_ids"].shape[1]
        gen_tokens = outputs[0][input_len:]
        throughput = len(gen_tokens) / duration
        ttft = (streamer.first_token_time - start) if streamer.first_token_time else duration
        gen_time = (end - streamer.first_token_time) if streamer.first_token_time else 0
        itl = gen_time / (len(gen_tokens) - 1) if len(gen_tokens) > 1 else 0

        if i == 0:
            generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            tokens_generated = len(gen_tokens)
            print(f"  Run {i + 1} (Warmup):   {throughput:.2f} tok/s, TTFT: {ttft:.4f}s, ITL: {itl:.4f}s")
        else:
            total_duration += duration
            total_throughput += throughput
            total_ttft += ttft
            total_itl += itl
            print(f"  Run {i + 1} (Measured): {throughput:.2f} tok/s, TTFT: {ttft:.4f}s, ITL: {itl:.4f}s")

    n = num_runs - 1
    avg_tp = total_throughput / n
    avg_dur = total_duration / n
    avg_ttft = total_ttft / n
    avg_itl = total_itl / n
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"\n  Output Preview: {generated_text[:100]}...")
    print(f"  Avg Throughput: {avg_tp:.2f} tok/s")
    print(f"  Avg Latency:    {avg_dur:.2f}s")
    print(f"  Avg TTFT:       {avg_ttft:.4f}s")
    print(f"  Avg ITL:        {avg_itl:.4f}s/token")
    print(f"  Peak VRAM:      {peak_vram:.2f} GB")

    # Quality comparison generation
    output_file = f"outputs/quality_case{case_id}.txt"
    print(f"\nGenerating quality evaluation prompts → {output_file}...")
    with open(output_file, "w") as f:
        for idx, passage in enumerate(QUALITY_PROMPTS):
            inputs_q = tokenizer(format_chatml(passage), return_tensors="pt").to("cuda:0")
            outputs_q = model.generate(
                **inputs_q,
                min_new_tokens=128,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            text = tokenizer.decode(outputs_q[0][inputs_q["input_ids"].shape[1]:], skip_special_tokens=True)
            f.write(f"Prompt {idx + 1}:\n{passage}\nGenerated Output:\n{text}\n{'-' * 50}\n\n")
    print(f"  Saved to {output_file}")


# ── vLLM Benchmarking (Cases 4, 5, 6) ────────────────────────────────────────

def run_vllm_case(model_path, use_speculative, use_bnb, label, case_id):
    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}\n  {label}\n{'=' * 60}")
    print(f"Model Disk Footprint: {get_model_size_gb(model_path)}")

    # Check existence
    if not os.path.exists(model_path):
        print(f"[SKIPPED] Model path '{model_path}' not found.")
        return

    import torch
    major, minor = torch.cuda.get_device_capability()
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Detected GPU Compute Capability: {major}.{minor}")
    print(f"Detected GPU Memory: {total_mem_gb:.2f} GB")

    if use_bnb and total_mem_gb < 8.0:
        print(f"Warning: Case 5 (vLLM BitsAndBytes) typically requires at least 8-12 GB VRAM (allocated pool is ~13 GB on Colab) and may fail with an Out-of-Memory (OOM) error on this {total_mem_gb:.2f} GB GPU.")

    llm_kwargs = {
        "model": model_path,
        "gpu_memory_utilization": 0.85 if not use_bnb else 0.90,
        "max_model_len": 512,
        "enforce_eager": False,
        "disable_log_stats": False,
    }

    if use_bnb:
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["dtype"] = "float16" if major < 8 else "bfloat16"
        if major < 8:
            llm_kwargs["attention_backend"] = "TRITON_ATTN"
    else:
        llm_kwargs["quantization"] = "awq_marlin"
        llm_kwargs["dtype"] = "float16"

    if use_speculative:
        llm_kwargs["speculative_config"] = {
            "method": "ngram",
            "num_speculative_tokens": 5,
            "prompt_lookup_max": 4,
            "prompt_lookup_min": 1,
        }

    llm = LLM(**llm_kwargs)
    print("vLLM engine ready.")

    sampling_params = SamplingParams(temperature=0, max_tokens=128, min_tokens=128)
    benchmark_prompt = format_chatml(
        f"generate a level 2 question testing comprehension: {BENCHMARK_PASSAGE}"
    )

    print(f"\nRunning deterministic benchmark (1 warmup + 5 measured)...")
    total_tp, total_dur, total_ttft, total_itl = 0.0, 0.0, 0.0, 0.0
    generated_text = ""
    tokens_generated = 0
    num_runs = 6

    for i in range(num_runs):
        start = time.time()
        outputs = llm.generate([benchmark_prompt], sampling_params)
        end = time.time()

        output = outputs[0]
        n_tok = len(output.outputs[0].token_ids)
        duration = end - start
        throughput = n_tok / duration

        metrics = output.metrics
        try:
            ttft = metrics.first_token_latency
            itl = (metrics.last_token_ts - metrics.first_token_ts) / (n_tok - 1) if n_tok > 1 else 0
        except (AttributeError, TypeError):
            ttft, itl = 0.0, duration / n_tok if n_tok > 0 else 0

        if i == 0:
            generated_text = output.outputs[0].text
            tokens_generated = n_tok
            print(f"  Run {i + 1} (Warmup):   {throughput:.2f} tok/s, TTFT: {ttft:.4f}s, ITL: {itl:.4f}s")
        else:
            total_dur += duration
            total_tp += throughput
            total_ttft += ttft
            total_itl += itl
            print(f"  Run {i + 1} (Measured): {throughput:.2f} tok/s, TTFT: {ttft:.4f}s, ITL: {itl:.4f}s")

    n = num_runs - 1
    avg_tp, avg_dur = total_tp / n, total_dur / n
    avg_ttft, avg_itl = total_ttft / n, total_itl / n

    print(f"\n  Output Preview: {generated_text[:100]}...")
    print(f"  Avg Throughput: {avg_tp:.2f} tok/s")
    print(f"  Avg Latency:    {avg_dur:.2f}s")
    print(f"  Avg TTFT:       {avg_ttft:.4f}s")
    print(f"  Avg ITL:        {avg_itl:.4f}s/token")

    # Generate Quality Output File
    output_file = f"outputs/quality_case{case_id}.txt"
    print(f"\nGenerating quality evaluation prompts (batched) → {output_file}...")
    formatted = [format_chatml(p) for p in QUALITY_PROMPTS]
    outputs_q = llm.generate(formatted, sampling_params)

    with open(output_file, "w") as f:
        for idx, out in enumerate(outputs_q):
            f.write(f"Prompt {idx + 1}:\n{QUALITY_PROMPTS[idx]}\n")
            f.write(f"Generated Output:\n{out.outputs[0].text}\n{'-' * 50}\n\n")
    print(f"  Saved to {output_file}")


# ── CLI & Subprocess Orchestration ───────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Qwen2.5-1.5B Inference Benchmark Suite")
    parser.add_argument(
        "--case",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="Select a specific case (1-6) or run all cases sequentially",
    )
    args = parser.parse_args()

    # If "all", orchestrate execution using subprocesses to guarantee clean memory isolation
    if args.case == "all":
        print(f"\n=== Orchestrating All 6 Benchmark Cases Sequentially (Subprocess Isolation) ===")
        cases = ["1", "2", "3", "4", "5", "6"]
        for case in cases:
            print(f"\n--- Spawning Subprocess for Case {case} ---")
            proc_args = [sys.executable, __file__, "--case", case]
            try:
                subprocess.run(proc_args, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[ERROR] Case {case} execution failed: {e}\nContinuing to next case...")
        print("\n=== All Orchestrated Benchmarks Finished ===")
        sys.exit(0)

    # Individual Case execution
    case = args.case
    if case == "1":
        run_hf_case(
            model_path="Qwen/Qwen2.5-1.5B",
            use_4bit=True,
            peft_adapter_path="./final_models/qwen_blooms_lora_final",
            label="Case 1: Base + LoRA (NF4)",
            case_id=1,
        )

    elif case == "2":
        run_hf_case(
            model_path="./models/qwen-1.5b-merged",
            use_4bit=False,
            peft_adapter_path=None,
            label="Case 2: Fully Merged Model (BF16 native)",
            case_id=2,
        )

    elif case == "3":
        run_hf_case(
            model_path="./models/qwen-1.5b-awq",
            use_4bit=False,
            peft_adapter_path=None,
            label="Case 3: AWQ 4-bit (HF Transformers Marlin)",
            case_id=3,
        )

    elif case == "4":
        run_vllm_case(
            model_path="./models/qwen-1.5b-awq",
            use_speculative=False,
            use_bnb=False,
            label="Case 4: vLLM AWQ 4-bit (Marlin + CUDA Graphs)",
            case_id=4,
        )

    elif case == "5":
        # Safe CUDA path fixing before vLLM/BNB load
        fix_cuda_paths()
        run_vllm_case(
            model_path="./models/qwen-1.5b-merged",
            use_speculative=False,
            use_bnb=True,
            label="Case 5: vLLM BitsAndBytes 4-bit (In-flight Quantization)",
            case_id=5,
        )

    elif case == "6":
        run_vllm_case(
            model_path="./models/qwen-1.5b-awq",
            use_speculative=True,
            use_bnb=False,
            label="Case 6: vLLM AWQ + N-Gram Speculative Decoding",
            case_id=6,
        )
