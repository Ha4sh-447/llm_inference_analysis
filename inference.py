import torch
import time
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

base_model_id = "Qwen/Qwen2.5-1.5B"
adapter_path = "./final_models/qwen_blooms_lora_final"

print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Configuring 4-bit loading")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

print("Loading base model")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="cuda:0",
)

print("Applying your trained LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# Prepare a test prompt
passage = "Photosynthesis is a system of biological proces  ses by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert light energy, typically from sunlight, into the chemical energy necessary to fuel their activities."
prompt = (
    f"<|im_start|>system\nYou are an educational assistant generating questions aligned with Bloom's Taxonomy.<|im_end|>\n"
    f"<|im_start|>user\ngenerate a level 2 question testing comprehension: {passage}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

print("\n--- Generating Question ---")
inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")

start_time = time.time()

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,
    top_p=0.95,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)

end_time = time.time()
duration = end_time - start_time

input_length = inputs['input_ids'].shape[1]
generated_tokens = outputs[0][input_length:]
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

tokens_generated = len(generated_tokens)
throughput = tokens_generated / duration

print("\n--- Generated Output ---")
print(generated_text)
print("\n--- Baseline Metrics (NF4 QLoRA) ---")
print(f"Time taken: {duration:.2f} seconds")
print(f"Tokens generated: {tokens_generated}")
print(f"Throughput: {throughput:.2f} tokens/second")
print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
