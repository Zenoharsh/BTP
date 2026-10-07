import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from layers import MoELayer
from utils import get_answer_labels
import json
import os
from PIL import Image
import time

from model import build_model
class LazyMultimodalDataset(Dataset):
    def __init__(self, processor, json_path="train.json", base_image_dir="./drive_mount/"):
        self.processor = processor
        self.base_image_dir = base_image_dir
        if not os.path.exists(json_path):
            self.data = [{"image": "dummy.jpg", "question": "What is this?", "answer": "A test."}]
        else:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.base_image_dir, item["image"].replace("images/", ""))
        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((512, 512))
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), color='white')
        
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": item.get("question", "")}]},
            {"role": "assistant", "content": [{"type": "text", "text": item.get("answer", "")}]}
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "image": image, "sample_id": f"sample_{idx}"}

def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    sample_ids = [item["sample_id"] for item in batch]
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs["sample_ids"] = sample_ids
    return inputs

def train_step(batch, student_model, optimizer, accumulation_steps, temperature=2.0, cached_targets=None):
    inputs = {k: v.to(student_model.device) for k, v in batch.items()}
    
    labels = get_answer_labels(inputs["input_ids"]).to(student_model.device)
    
    student_outputs = student_model(**inputs)
    student_logits = student_outputs.logits
    
    shift_logits = student_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    kd_loss = torch.tensor(0.0, device=student_model.device)
    if cached_targets is not None:
        c_probs = cached_targets["probs"].to(student_model.device)
        c_indices = cached_targets["indices"].to(student_model.device)
        c_mask = cached_targets["mask"].to(student_model.device)
        
        student_valid = shift_logits[0][c_mask]
        if student_valid.size(0) > 0:
            student_valid_scaled = student_valid / temperature
            
            # P0: True Memory-Efficient Sparse KD
            # Compute selected student log-probabilities as: logit_topk/T - logsumexp(all_student_logits/T)
            student_topk_logits = student_valid_scaled.gather(dim=1, index=c_indices)
            logsumexp = torch.logsumexp(student_valid_scaled, dim=-1, keepdim=True)
            student_topk_log_probs = student_topk_logits - logsumexp
            
            # Top-K teacher distillation with renormalized probabilities
            c_probs_norm = c_probs / c_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            kd_loss = (c_probs_norm * (torch.log(c_probs_norm.clamp(min=1e-8)) - student_topk_log_probs)).sum(dim=-1).mean()
            kd_loss = kd_loss * (temperature ** 2)
            
    # Extract router metrics
    student_aux_loss = 0.0
    layers = get_mlp_layers(student_model)
    routing_stats = {}
    for layer in layers:
        if hasattr(layer.mlp, 'metrics'):
            metrics = layer.mlp.metrics
            student_aux_loss += metrics.get("aux_loss", 0.0)
            for k, v in metrics.items():
                if k == "aux_loss":
                    continue
                if isinstance(v, list) and len(v) > 0:
                    if k not in routing_stats:
                        routing_stats[k] = [0.0] * len(v)
                    for idx in range(len(v)):
                        routing_stats[k][idx] += v[idx]
                elif not isinstance(v, list):
                    routing_stats[k] = routing_stats.get(k, 0.0) + v

    num_layers = len(layers)
    for k in routing_stats:
        if isinstance(routing_stats[k], list):
            routing_stats[k] = [val / num_layers for val in routing_stats[k]]
        else:
            routing_stats[k] /= num_layers
            
    # Add explicit Layer 0 diagnostic for Step 1 telemetry tracking
    if not hasattr(student_model, "_printed_routing_telemetry") and len(layers) > 0:
        l0_metrics = layers[0].mlp.metrics
        num_tokens = sum(l0_metrics["expert_counts"])
        capacity = int(math.ceil((num_tokens / layers[0].mlp.num_experts) * layers[0].mlp.capacity_factor))
        print(f"\n--- Step 1 Layer 0 Routing Math Verification ---")
        print(f"Raw token count before dispatch: {int(num_tokens)}")
        print(f"Per-expert assignment counts before capacity: {l0_metrics['expert_counts']}")
        print(f"Per-expert capacity: {capacity}")
        print(f"Reported drop_rate: {l0_metrics['drop_rate']:.4f}")
        print(f"Expected dropped assignments: {int(l0_metrics['drop_rate'] * num_tokens)}")
        print(f"Fallback token count: {int(l0_metrics['drop_rate'] * num_tokens)}")
        print(f"Final expert counts in telemetry: {l0_metrics['expert_counts']}")
        print(f"------------------------------------------------\n")
        student_model._printed_routing_telemetry = True
    
    total_loss = ce_loss + kd_loss + (student_aux_loss * 0.01)
    total_loss = total_loss / accumulation_steps
    total_loss.backward()
    
    return total_loss.item() * accumulation_steps, ce_loss.item(), kd_loss.item(), routing_stats

def main():
    print("Initializing Validated Cached MoE Training Pipeline...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    # P0: Teacher is explicitly NOT loaded in the student training path
    student = setup_student(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    dataset = LazyMultimodalDataset(processor, json_path="train.json", base_image_dir="./drive_mount/")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=2e-4)
    accumulation_steps = 16
    
    print("Running Training Loop (Cache Integration Mode)...")
    student.train()
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        if step >= 10:
            break
            
        start_time = time.time()
        
        sample_ids = batch.pop("sample_ids")
        sample_id = sample_ids[0]
        
        # Load cached teacher targets matching the exact sample ID
        cached_targets = None
        cache_file = os.path.join("teacher_cache", f"{sample_id}.pt")
        if os.path.exists(cache_file):
            cached_targets = torch.load(cache_file, weights_only=False)
        else:
            print(f"Warning: Missing teacher cache for {sample_id}. Run scripts/cache_teacher.py first.")
            
        total_loss, ce_loss, kd_loss, stats = train_step(batch, student, optimizer, accumulation_steps, cached_targets=cached_targets)
        
        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader) or step == 9:
            optimizer.step()
            optimizer.zero_grad()
        
        step_time = time.time() - start_time
        peak_vram = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        
        print(f"Step {step+1} [{sample_id}] | Loss: {total_loss:.4f} (CE: {ce_loss:.4f}, Sparse KD: {kd_loss:.4f})")
        print(f"  -> Time: {step_time:.2f}s | Peak VRAM: {peak_vram:.1f}MB")
        print(f"  -> Routing: Entropy={stats.get('routing_entropy',0):.3f}, DropRate={stats.get('drop_rate',0):.3f}")
        
        # Format lists
        exp_counts = stats.get('expert_counts', [])
        mean_probs = stats.get('mean_routing_prob', [])
        print(f"  -> Expert counts: [{', '.join([f'{c:.1f}' for c in exp_counts])}]")
        print(f"  -> mean routing probability: [{', '.join([f'{p:.4f}' for p in mean_probs])}]")

    print("\nTraining Pipeline Test Complete.")

if __name__ == "__main__":
    main()
