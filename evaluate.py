import argparse
import json
import time
import os
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from config import load
from metrics import compute_metric
from model import apply_moe_surgery, resolve_token_ids
from layers import moe_layers
from prompts import build_prompt
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--split", default="data/v3/dev.jsonl")
    parser.add_argument("--out", default="preds.jsonl")
    parser.add_argument("--max_pixels", type=int, default=401408)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base", action="store_true", help="Evaluate untouched base model")
    parser.add_argument("--record_routing", action="store_true")
    args = parser.parse_args()

    cfg = load(args.config)
    
    print("Loading processor & model...")
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=256*28*28, max_pixels=args.max_pixels)
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
                             
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, quantization_config=bnb,
                                                            torch_dtype=torch.float16, device_map={"": 0})
    
    token_ids = resolve_token_ids(processor)
    
    if not args.base:
        apply_moe_surgery(model, cfg, token_ids["image_pad"])
        if args.record_routing:
            for l in moe_layers(model):
                l.record = True
                
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    
    eos_ids = [processor.tokenizer.convert_tokens_to_ids('<|im_end|>')]
    nl_ids = processor.tokenizer.encode('\n', add_special_tokens=False)
    if len(nl_ids) == 1:
        eos_ids.append(nl_ids[0])
        
    samples = []
    with open(args.split, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
            if args.limit and len(samples) >= args.limit:
                break
                
    root = os.path.dirname(os.path.abspath(args.split))
    all_routing = []
    results = []
    
    task_scores = {}
    task_counts = {}
    
    for s in tqdm(samples, desc="Evaluating"):
        msgs = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": build_prompt(s["task"], s["question"])}
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_path = os.path.join(root, s["image"])
        image = Image.open(img_path).convert("RGB")
        
        inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(model.device)
        
        img_tokens = (inputs.input_ids == token_ids["image_pad"]).sum().item()
        
        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=eos_ids
            )
        latency = time.time() - t0
        
        input_len = inputs.input_ids.shape[1]
        new_tokens = outputs[0, input_len:]
        pred = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
        pred = pred.split('\n')[0].strip()
        
        score = compute_metric(s["task"], pred, s["answers"])
        
        task_scores[s["task"]] = task_scores.get(s["task"], 0) + score
        task_counts[s["task"]] = task_counts.get(s["task"], 0) + 1
        
        results.append({
            "uid": s["uid"],
            "task": s["task"],
            "pred": pred,
            "answers": s["answers"],
            "score": score,
            "latency": latency,
            "img_tokens": img_tokens
        })
        
        if args.record_routing and not args.base:
            routing_data = []
            for l in moe_layers(model):
                ids = l.last_expert_ids.numpy()
                text_ids = ids[ids != -1]
                hist, _ = np.histogram(text_ids, bins=np.arange(l.num_experts + 1))
                routing_data.append(hist)
            all_routing.append(np.array(routing_data))
            
    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
            
    if args.record_routing and not args.base:
        np.savez("routing_dev.npz", routing=np.array(all_routing))
        
    print("\n--- Evaluation Results ---")
    macro_sum = 0
    for t in task_counts:
        avg = task_scores[t] / task_counts[t]
        macro_sum += avg
        print(f"Task: {t} | Score: {avg:.4f}")
    print(f"Macro Average: {macro_sum / len(task_counts):.4f}")

if __name__ == "__main__":
    main()
