from transformers import TrainingArguments
import os
import torch
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer
import matplotlib.pyplot as plt

model_id = "Qwen/Qwen2.5-1.5B"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="cuda:0",
)


print(f"Model memory: {model.get_memory_footprint()/1e6}")


# Lora training
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable()

# Lora config
lora_config= LoraConfig(
    r = 16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
print(f"New training params after lora: {model.print_trainable_parameters()}")

# Load Dataset
dataset = load_dataset("json", data_files="./dataset/blooms_4level_train.json", split="train")
eval_dataset = load_dataset("json", data_files="./dataset/blooms_4level_val.json", split="train")

# format of training data
def format_prompt(example):
    formatted_text = (
        f"<|im_start|>system\nYou are an educational assistant generating questions aligned with Bloom's Taxonomy.<|im_end|>\n"
        f"<|im_start|>user\n{example['input_text']}<|im_end|>\n"
        f"<|im_start|>assistant\n{example['target_text']}{tokenizer.eos_token}"
    )
    return {"text": formatted_text}

formatted_dataset = dataset.map(format_prompt)
formatted_eval_dataset = eval_dataset.map(format_prompt)

# Training arguments
training_args = SFTConfig(
    output_dir="./models/qwen-1.5b-blooms-lora",
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

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    eval_dataset=formatted_eval_dataset,
    args=training_args,
)

# Empty the cache
torch.cuda.empty_cache()
print("Starting training")
trainer.train()
# --- Generating Training and Eval Loss Graphs ---
print("Generating training graphs...")
log_history = trainer.state.log_history
train_loss_data = [(log['step'], log['loss']) for log in log_history if 'loss' in log]
eval_loss_data = [(log['step'], log['eval_loss']) for log in log_history if 'eval_loss' in log]

plt.figure(figsize=(10, 5))
if train_loss_data:
    t_steps, t_losses = zip(*train_loss_data)
    plt.plot(t_steps, t_losses, label="Training Loss", color='blue')
if eval_loss_data:
    e_steps, e_losses = zip(*eval_loss_data)
    plt.plot(e_steps, e_losses, label="Eval Loss", color='orange', marker='o')

plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Qwen2.5-1.5B Bloom's Taxonomy QLoRA Loss")
plt.legend()
plt.grid(True)
plt.savefig("./training_loss_graph.png")
print("Training graph saved as training_loss_graph.png")
# Save the final adapter weights
trainer.model.save_pretrained("./final_models/qwen_blooms_lora_final")
tokenizer.save_pretrained("./final_tokenizers/qwen_blooms_lora_final")
print("Training complete and adapter saved!")
