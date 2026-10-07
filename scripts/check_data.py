import json
import os
import random
from PIL import Image

def check():
    splits = ["data/v3/train.jsonl", "data/v3/dev.jsonl", "data/v3/test.jsonl", "data/smoke.jsonl"]
    
    for split in splits:
        if not os.path.exists(split):
            print(f"Skipping {split}")
            continue
            
        print(f"=== Split: {split} ===")
        data_by_task = {}
        
        with open(split, 'r') as f:
            for line in f:
                item = json.loads(line)
                task = item["task"]
                if task not in data_by_task:
                    data_by_task[task] = []
                data_by_task[task].append(item)
                
                # Check image opening
                img_path = os.path.join(os.path.dirname(split), item["image"])
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                except Exception as e:
                    print(f"FAILED TO OPEN IMAGE: {img_path} ({e})")
                    return
                    
        for task, items in data_by_task.items():
            print(f"Task: {task} | Count: {len(items)}")
            
            if task == "spatial_reasoning":
                t_count = sum(1 for x in items if x["answers"] == ["True"])
                f_count = sum(1 for x in items if x["answers"] == ["False"])
                print(f"  Distribution - True: {t_count}, False: {f_count}")
                
            print("  Random Samples:")
            samples = random.sample(items, min(3, len(items)))
            for s in samples:
                print(f"    Q: {s['question']} | A: {s['answers']} | Img: {s['image']}")
        print("All images verified.\n")

if __name__ == "__main__":
    check()
