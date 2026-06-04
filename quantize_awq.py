"""
quantize_awq.py — 4-bit AWQ quantization of the merged Qwen2.5-1.5B checkpoint.

Uses calibration data from the training set to determine activation-aware
quantization scales. Produces a compact AWQ model (~1.08 GB on disk) that
can be served with Marlin kernels in vLLM.
"""

import json
import os

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_PATH = "./models/qwen-1.5b-merged"
OUTPUT_PATH = "./models/qwen-1.5b-awq"
DATASET_PATH = "./dataset/data.json"
NUM_CALIBRATION_SAMPLES = 32

#  Calibration Data 
with open(DATASET_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

calib_data = []
for sample in dataset[:NUM_CALIBRATION_SAMPLES]:
    text = (
        f"<|im_start|>system\nYou are an educational assistant generating questions "
        f"aligned with Bloom's Taxonomy.<|im_end|>\n"
        f"<|im_start|>user\n{sample['input_text']}<|im_end|>\n"
        f"<|im_start|>assistant\n{sample['target_text']}<|im_end|>"
    )
    calib_data.append(text)
print(f"Loaded {len(calib_data)} calibration samples")

# Model & Tokenizer
model = AutoAWQForCausalLM.from_pretrained(MODEL_PATH, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# Quantize
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM",
}

torch.cuda.empty_cache()
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calib_data,
    max_calib_seq_len=384,
    max_calib_samples=NUM_CALIBRATION_SAMPLES,
)

model.save_quantized(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
print(f"AWQ quantized model saved to {OUTPUT_PATH}")
