import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from layers import MoELayer
import json
import os
from PIL import Image

def setup_models(model_id="Qwen/Qwen2-VL-2B-Instruct"):
    print("Loading Teacher Model...")
    # 1. Load Teacher (Frozen)
    teacher = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading Student Model in 4-bit (Anti-OOM optimizations)...")
    # 2. Load Student (with 4-bit QLoRA config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    student = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    
    # Explicitly enable gradient checkpointing to save massive VRAM activations
    student.gradient_checkpointing_enable()

    print("Performing Architecture Surgery (Sparse Upcycling)...")
    hidden_size = student.config.text_config.hidden_size
    for i, layer in enumerate(student.model.language_model.layers):
        original_mlp = layer.mlp
        moe_layer = MoELayer(original_mlp, hidden_size=hidden_size, num_experts=4)
        layer.mlp = moe_layer

    print("Applying QLoRA to Attention & Experts...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", 
            "v_proj",
            ".*experts.*gate_proj.*", 
            ".*experts.*up_proj.*", 
            ".*experts.*down_proj.*"
        ],
        modules_to_save=["router"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(student, lora_config)

    print("Unfreezing Routers...")
    for name, param in student.named_parameters():
        if "router" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)

    return teacher, student

class LazyMultimodalDataset(Dataset):
    def __init__(self, processor, json_path="train.json", base_image_dir="./drive_mount/"):
        self.processor = processor
        self.base_image_dir = base_image_dir
        
        # Parse train.json lazily. DO NOT load images into RAM here.
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Creating a dummy file.")
            self.data = [{"image": "dummy.jpg", "question": "What is in the image?", "answer": "A document."}]
        else:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        filename = item["image"].replace("images/", "")
        image_path = os.path.join(self.base_image_dir, filename)
        
        # LAZY LOAD: Fetch from disk / GDrive mount exactly when requested by the dataloader
        try:
            image = Image.open(image_path).convert("RGB")
            # Downscale high-res documents to prevent thousands of visual tokens (and PCIe swapping)
            image.thumbnail((512, 512))
        except FileNotFoundError:
            # Fallback for structural testing
            image = Image.new('RGB', (224, 224), color='white')
        
        # Flawlessly map to Qwen2-VL Processor requirements
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": item.get("question", "")},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": item.get("answer", "")},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return {"text": text, "image": image}

def collate_fn(batch, processor):
    texts = [item["text"] for item in batch]
    images = [item["image"] for item in batch]
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    return inputs

def train_step(batch, teacher_model, student_model, temperature=2.0):
    inputs = {k: v.to(student_model.device) for k, v in batch.items()}
    
    # 1. Teacher Forward (Strictly no gradients to save VRAM)
    with torch.no_grad():
        teacher_outputs = teacher_model(**inputs)
        teacher_logits = teacher_outputs.logits
        
    # 2. Student Forward (Mixed precision handled by QLoRA automatically)
    student_outputs = student_model(**inputs)
    student_logits = student_outputs.logits
    
    # 3. Extract and aggregate the Aux Losses from all custom MoE Layers
    student_aux_loss = 0.0
    for layer in student_model.model.model.language_model.layers:
        if hasattr(layer.mlp, 'latest_aux_loss'):
            student_aux_loss += layer.mlp.latest_aux_loss
            
    # 4. Knowledge Distillation (KD) Loss using KL Divergence
    scaled_student_logits = student_logits / temperature
    scaled_teacher_logits = teacher_logits / temperature
    
    student_log_probs = F.log_softmax(scaled_student_logits.view(-1, scaled_student_logits.size(-1)).float(), dim=-1)
    teacher_probs = F.softmax(scaled_teacher_logits.view(-1, scaled_teacher_logits.size(-1)).float(), dim=-1)
    
    kd_loss = F.kl_div(
        student_log_probs, 
        teacher_probs, 
        reduction="batchmean"
    ) * (temperature ** 2) 
    
    # 5. Total Loss Calculation (Downscale aux_loss so it doesn't overpower KD loss)
    aux_loss_coef = 0.01
    total_loss = kd_loss + (student_aux_loss * aux_loss_coef)
    
    return total_loss, kd_loss.item(), (student_aux_loss.item() if isinstance(student_aux_loss, torch.Tensor) else student_aux_loss)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train Qwen2-VL Sparse MoE")
    parser.add_argument("--data", type=str, default="train.json", help="Path to training JSON")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (1 for RTX 3050)")
    parser.add_argument("--accum_steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    print("Initializing Qwen2-VL Sparse MoE Edge-Constrained Pipeline...")
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    teacher, student = setup_models(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    
    dataset = LazyMultimodalDataset(processor, json_path=args.data, base_image_dir="./drive_mount/")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, processor)
    )
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=args.lr)
    
    print(f"Beginning training for {args.epochs} epochs with accum_steps={args.accum_steps}, batch_size={args.batch_size}, lr={args.lr}...")
    
    global_step = 0
    student.train()
    optimizer.zero_grad()
    
    for epoch in range(args.epochs):
        for step, batch in enumerate(dataloader):
            total_loss, kd_loss, aux_loss = train_step(batch, teacher, student)
            
            # Normalize loss for gradient accumulation
            total_loss = total_loss / args.accum_steps
            total_loss.backward()
            
            # Perform optimization step only after accumulating N gradients
            if (step + 1) % args.accum_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                print(f"Epoch {epoch+1} | Step {global_step} | Total Loss: {total_loss.item() * args.accum_steps:.4f} | KD Loss: {kd_loss:.4f} | Aux Loss: {aux_loss:.4f}")
            
    print("\nTraining Complete! Saving QLoRA Checkpoint...")
    student.save_pretrained("./qwen2vl_sparse_moe_checkpoint")
    processor.save_pretrained("./qwen2vl_sparse_moe_checkpoint")
    print("Saved to ./qwen2vl_sparse_moe_checkpoint successfully.")

if __name__ == "__main__":
    main()
