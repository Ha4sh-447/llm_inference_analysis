import os, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

SOURCE = "./models/qwen-1.5b-merged"
OUTPUT = "./models/qwen-1.5b-fp8"

tokenizer = AutoTokenizer.from_pretrained(SOURCE)

model = AutoModelForCausalLM.from_pretrained(SOURCE, dtype=torch.float32, device_map="cpu")

# Load your calibration data
with open("./dataset/calibration.json") as f:
    raw = json.load(f)

texts = [s["text"] for s in raw[:512]]

def tokenize(sample):
    return tokenizer(sample["text"], truncation=True, max_length=2048, padding=False)

dataset = Dataset.from_dict({"text": texts})
dataset = dataset.map(tokenize, remove_columns=["text"])

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8",
    ignore=["lm_head"],
)


# Pass dataset to oneshot for better weight scales
compressed_model = oneshot(
    model=model,
    recipe=recipe,
    dataset=dataset,
    num_calibration_samples=512,
    processor=tokenizer,
)



os.makedirs(OUTPUT, exist_ok=True)
compressed_model.save_pretrained(OUTPUT)
tokenizer.save_pretrained(OUTPUT)
print("Done")
