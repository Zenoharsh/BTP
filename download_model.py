# Download Qwen/Qwen1.5-0.5B baseline model from HuggingFace

from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen1.5-0.5B"

print(f"Downloading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print(f"Downloading model {model_id}...")
model = AutoModelForCausalLM.from_pretrained(model_id)

print(f"{model_id} successfully downloaded and cached!")
