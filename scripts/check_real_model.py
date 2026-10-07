import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import torch
from transformers import AutoProcessor
from config import load
from data import VQADataset
from model import build_model
from utils import get_answer_labels

def main():
    if not torch.cuda.is_available():
        print("CRITICAL: CUDA is not available. 4-bit quantized model loading requires a GPU.")
        print("Please run this script on your Kaggle instance or a machine with CUDA.")
        return

    cfg = load("configs/base.yaml")
    print(f"Loaded config for {cfg.model_id}")
    
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(cfg.model_id)
    
    print("Building model (this may take a bit)...")
    model, token_ids = build_model(cfg, processor, device_index=0)
    
    print("Loading one sample from smoke.jsonl...")
    dataset = VQADataset("data/smoke.jsonl", processor, limit=1)
    item = dataset[0]
    
    print(f"Sample task: {item['task']}")
    print(f"Text snippet: {item['text'][:100]}...\n")
    
    # Process inputs
    inputs = processor(
        text=[item["text"]],
        images=[item["image"]],
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    
    # Generate labels
    labels = get_answer_labels(
        inputs.input_ids, 
        token_ids["im_start"], 
        token_ids["assistant"], 
        token_ids["im_end"]
    )
    
    unmasked = (labels != -100).sum().item()
    print(f"Unmasked answer tokens: {unmasked}")
    
    unmasked_ids = inputs.input_ids[labels != -100]
    print(f"Decoded answer tokens: {processor.tokenizer.decode(unmasked_ids)}")
    
    assert unmasked > 0, "No answer tokens were unmasked! get_answer_labels is broken."
    
    print("\nRunning forward pass...")
    outputs = model(
        input_ids=inputs.input_ids,
        image_grid_thw=inputs.image_grid_thw,
        pixel_values=inputs.pixel_values,
        labels=labels
    )
    
    print(f"Logits shape: {outputs.logits.shape}")
    print(f"Loss: {outputs.loss.item():.4f}")
    print("Sanity check passed!")

if __name__ == "__main__":
    main()
