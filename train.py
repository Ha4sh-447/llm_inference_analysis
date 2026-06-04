"""
train.py — QLoRA fine-tuning of Qwen2.5-1.5B for Bloom's Taxonomy question generation.

Uses 4-bit NormalFloat quantization (NF4) + LoRA adapters to fit training within
4 GB VRAM on an RTX 3050. Produces loss curves and saves the final adapter weights.
"""

import os
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen2.5-1.5B"
DATASET_TRAIN = "./dataset/data.json"
DATASET_VAL = "./dataset/val.json"
OUTPUT_DIR = "./models/qwen-1.5b-blooms-lora"
ADAPTER_SAVE_DIR = "./final_models/qwen_blooms_lora_final"
TOKENIZER_SAVE_DIR = "./final_tokenizers/qwen_blooms_lora_final"

# ── Tokenizer ────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ── 4-bit Quantized Base Model ───────────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="cuda:0",
)
print(f"Base model memory footprint: {model.get_memory_footprint() / 1e6:.1f} MB")

# ── LoRA Adapter ─────────────────────────────────────────────────────────────
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── Dataset ──────────────────────────────────────────────────────────────────
dataset = load_dataset("json", data_files=DATASET_TRAIN, split="train")
eval_dataset = load_dataset("json", data_files=DATASET_VAL, split="train")


def format_prompt(example):
    """Format a dataset row into the Qwen ChatML template."""
    return {
        "text": (
            f"<|im_start|>system\nYou are an educational assistant generating questions "
            f"aligned with Bloom's Taxonomy.<|im_end|>\n"
            f"<|im_start|>user\n{example['input_text']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['target_text']}{tokenizer.eos_token}"
        )
    }


formatted_dataset = dataset.map(format_prompt)
formatted_eval_dataset = eval_dataset.map(format_prompt)

# ── Training ─────────────────────────────────────────────────────────────────
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    num_train_epochs=3,
    save_steps=500,
    optim="paged_adamw_8bit",
    bf16=True,
    remove_unused_columns=True,
    gradient_checkpointing=True,
    dataset_text_field="text",
    max_length=384,
    eval_strategy="steps",
    eval_steps=100,
    per_device_eval_batch_size=1,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    eval_dataset=formatted_eval_dataset,
    args=training_args,
)

torch.cuda.empty_cache()
print("Starting training...")
trainer.train()

# ── Loss Curves ──────────────────────────────────────────────────────────────
print("Generating training loss graph...")
log_history = trainer.state.log_history
train_loss = [(log["step"], log["loss"]) for log in log_history if "loss" in log]
eval_loss = [(log["step"], log["eval_loss"]) for log in log_history if "eval_loss" in log]

plt.figure(figsize=(10, 5))
if train_loss:
    steps, losses = zip(*train_loss)
    plt.plot(steps, losses, label="Training Loss", color="blue")
if eval_loss:
    steps, losses = zip(*eval_loss)
    plt.plot(steps, losses, label="Eval Loss", color="orange", marker="o")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Qwen2.5-1.5B Bloom's Taxonomy QLoRA — Training Loss")
plt.legend()
plt.grid(True)
plt.savefig("./outputs/training_loss_graph.png")
print("Loss graph saved to outputs/training_loss_graph.png")

# ── Save Adapter ─────────────────────────────────────────────────────────────
trainer.model.save_pretrained(ADAPTER_SAVE_DIR)
tokenizer.save_pretrained(TOKENIZER_SAVE_DIR)
print(f"Training complete. Adapter saved to {ADAPTER_SAVE_DIR}")
