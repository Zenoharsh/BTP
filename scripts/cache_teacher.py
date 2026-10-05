import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from train import LazyMultimodalDataset, collate_fn
from utils import get_answer_labels

def cache_teacher_targets(model_id="Qwen/Qwen2-VL-2B-Instruct", data_path="train.json", cache_dir="teacher_cache"):
    """
    P0-8: Precomputes and caches teacher KD targets for answer tokens only.
    Iterates the ENTIRE dataset and saves targets keyed by sample_id to perfectly 
    align with the shuffled dataloader in train.py.
    """
    os.makedirs(cache_dir, exist_ok=True)
    print("Initializing Teacher Model for full dataset caching...")
    
    try:
        teacher = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
    except Exception:
        print("Falling back to CPU/FP32 for teacher...")
        teacher = Qwen2VLForConditionalGeneration.from_pretrained(model_id)
        
    teacher.eval()
    processor = AutoProcessor.from_pretrained(model_id)
    dataset = LazyMultimodalDataset(processor, json_path=data_path, base_image_dir="./drive_mount/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda b: collate_fn(b, processor))
    
    temperature = 2.0
    
    print(f"Starting teacher inference over {len(dataloader)} samples...")
    for step, batch in enumerate(dataloader):
        sample_ids = batch.pop("sample_ids")
        sample_id = sample_ids[0]
        
        inputs = {k: v.to(teacher.device) for k, v in batch.items()}
        
        # Centralized answer masking
        labels = get_answer_labels(inputs["input_ids"])
        valid_mask = (labels != -100)[0]
        
        if not valid_mask.any():
            print(f"Sample {sample_id}: No answer tokens found. Skipping.")
            continue
            
        with torch.no_grad():
            outputs = teacher(**inputs)
            
        # Shift logits for causal LM
        shift_logits = outputs.logits[0, :-1, :]
        shift_mask = valid_mask[1:]
        
        valid_logits = shift_logits[shift_mask]
        
        if valid_logits.size(0) == 0:
            continue
            
        # Compute Top-K probabilities
        probs = F.softmax(valid_logits / temperature, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        cache_file = os.path.join(cache_dir, f"{sample_id}.pt")
        torch.save({
            "probs": topk_probs.cpu(),
            "indices": topk_indices.cpu(),
            "mask": shift_mask.cpu()
        }, cache_file)
        
        print(f"Cached {sample_id} -> {cache_file} (Masked tokens: {valid_logits.size(0)})")
        
    print("Teacher caching complete.")

if __name__ == "__main__":
    cache_teacher_targets()
