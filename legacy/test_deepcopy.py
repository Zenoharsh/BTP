import torch
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
import copy

print("Loading 4-bit model...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", quantization_config=bnb_config, device_map={"": 0})

print("Testing deepcopy...")
try:
    mlp_copy = copy.deepcopy(model.model.layers[0].mlp)
    print("SUCCESS: deepcopy works on 4-bit modules.")
except Exception as e:
    print("ERROR:", str(e))
