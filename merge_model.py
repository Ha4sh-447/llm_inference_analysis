"""
merge_model.py — Merge QLoRA adapter weights into the base Qwen2.5-1.5B model.

Loads the fine-tuned LoRA adapter, merges it into the base model on CPU,
and saves the resulting standalone BF16 checkpoint for downstream quantization
and inference benchmarking.
"""

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

ADAPTER_PATH = "./final_models/qwen_blooms_lora_final"
TOKENIZER_PATH = "./final_tokenizers/qwen_blooms_lora_final"
OUTPUT_PATH = "./models/qwen-1.5b-merged"

model = AutoPeftModelForCausalLM.from_pretrained(
    ADAPTER_PATH,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

merged_model = model.merge_and_unload()
merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Merged model saved to {OUTPUT_PATH}")
