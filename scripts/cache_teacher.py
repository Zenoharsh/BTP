import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from train import LazyMultimodalDataset, collate_fn

def cache_teacher_targets(model_id="Qwen/Qwen2-VL-2B-Instruct", data_path="train.json", cache_dir="teacher_cache"):
    """
    P0-8: Precomputes and caches teacher KD targets for answer tokens only.
    This avoids running the teacher during the MoE student training loop.
    """
    os.makedirs(cache_dir, exist_ok=True)
    print("Initializing Teacher Model for caching...")
    
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
    
    im_start_id = 151644
    assistant_id = 77091
    im_end_id = 151645
    temperature = 2.0
    
    print(f"Starting teacher inference over {len(dataloader)} batches...")
    for step, batch in enumerate(dataloader):
        # We only cache a few steps for the smoke test demonstration
        if step >= 5:
            break
            
        inputs = {k: v.to(teacher.device) for k, v in batch.items()}
        seq = inputs["input_ids"][0]
        valid_mask = torch.zeros_like(seq, dtype=torch.bool)
        
        start_indices = (seq == im_start_id).nonzero(as_tuple=True)[0]
        for start_idx in start_indices:
            if start_idx + 1 < len(seq) and seq[start_idx + 1] == assistant_id:
                end_idx_candidates = (seq[start_idx:] == im_end_id).nonzero(as_tuple=True)[0]
                if len(end_idx_candidates) > 0:
                    end_idx = start_idx + end_idx_candidates[0]
                    valid_mask[start_idx+3 : end_idx+1] = True
                    
        if not valid_mask.any():
            print(f"Step {step}: No answer tokens found. Skipping.")
            continue
            
        with torch.no_grad():
            outputs = teacher(**inputs)
            
        # Shift logits for causal LM
        shift_logits = outputs.logits[0, :-1, :]
        shift_mask = valid_mask[1:]
        
        valid_logits = shift_logits[shift_mask]
        
        if valid_logits.size(0) == 0:
            continue
            
        # Compute and extract Top-K probabilities to save disk space
        probs = F.softmax(valid_logits / temperature, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        
        cache_file = os.path.join(cache_dir, f"batch_{step}.pt")
        torch.save({
            "probs": topk_probs.cpu(),
            "indices": topk_indices.cpu(),
            "mask": shift_mask.cpu()
        }, cache_file)
        
        print(f"Cached step {step} -> {cache_file} (Masked tokens: {valid_logits.size(0)})")
        
    print("Teacher caching complete.")

if __name__ == "__main__":
    cache_teacher_targets()
