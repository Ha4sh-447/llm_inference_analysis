from torch.cuda import temperature
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

print("4 bit config")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
)

print("Load model")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
        quantization_config=bnb_config,
        device_map="cuda:0",
)

print("Loading model using Lora adapter")
model = PeftModel.from_pretrained(model, adapter_path)

# test prompt
passage = "Photosynthesis is a system of biological processes by which photosynthetic organisms, such as most plants, algae, and cyanobacteria, convert light energy, typically from sunlight, into the chemical energy necessary to fuel their activities."
prompt = (
    f"<|im_start|>system\nYou are an educational assistant generating questions aligned with Bloom's Taxonomy.<|im_end|>\n"
    f"<|im_start|>user\ngenerate a level 2 question testing comprehension: {passage}<|im_end|>\n"
    f"<|im_start|>assistant\n"
)

print("\n Generating question")
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

print(outputs)
