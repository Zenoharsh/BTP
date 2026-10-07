import os
import json
import hashlib
import requests
import io
import time
import argparse
from collections import defaultdict
from datasets import load_dataset
from PIL import Image

def get_image_hash(image):
    if not isinstance(image, Image.Image):
        image = Image.open(io.BytesIO(image)).convert("RGB")
    else:
        image = image.convert("RGB")
    hasher = hashlib.md5()
    hasher.update(image.tobytes())
    return hasher.hexdigest(), image

def download_image(url, max_retries=1):
    for i in range(max_retries):
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return Image.open(io.BytesIO(r.content)).convert("RGB")
        except:
            pass
    return None

def build_task(task_id):
    os.makedirs("data/v3/images", exist_ok=True)
    os.makedirs("data/v3/temp", exist_ok=True)
    
    seen_hashes = set()
    splits_data = {"train": [], "dev": [], "test": []}
    
    targets = {
        "train": 400,
        "dev": 50,
        "test": 200
    }
    
    tasks = {
        "document_ocr": {
            "id": "document_ocr",
            "repo": "pixparse/docvqa-single-page-questions",
            "train_src": "train",
            "test_src": "validation",
            "q_field": "question",
            "a_field": "answers"
        },
        "chart_qa": {
            "id": "chart_qa",
            "repo": "HuggingFaceM4/ChartQA",
            "train_src": "train",
            "test_src": "test",
            "q_field": "query",
            "a_field": "label"
        },
        "spatial_reasoning": {
            "id": "spatial_reasoning",
            "repo": "cambridgeltl/vsr_random",
            "train_src": "train",
            "test_src": "test",
            "q_field": "caption",
            "a_field": "label"
        }
    }
    
    task = tasks[task_id]
    
    task_idx = list(tasks.keys()).index(task_id)
    uid_counter = task_idx * 1000000 
    
    print(f"Processing {task['id']} - Train/Dev", flush=True)
    ds_train = load_dataset(task["repo"], split=task["train_src"], streaming=True)
    ds_train = ds_train.shuffle(seed=0, buffer_size=5000)
    
    train_collected = 0
    dev_collected = 0
    
    for item in ds_train:
        if train_collected >= targets["train"] and dev_collected >= targets["dev"]:
            break
            
        q = item[task["q_field"]]
        a = item[task["a_field"]]
        if task["id"] == "spatial_reasoning":
            a = ["True"] if a == 1 else ["False"]
            img_url = item["image_link"]
            img = download_image(img_url)
            if img is None: continue
        else:
            img = item["image"]
            
        img_hash, img_pil = get_image_hash(img)
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)
        
        if train_collected < targets["train"]:
            target_split = "train"
            train_collected += 1
        else:
            target_split = "dev"
            dev_collected += 1
            
        uid = f"v3_{uid_counter:05d}"
        uid_counter += 1
        
        img_path = f"data/v3/images/{uid}.jpg"
        img_pil.save(img_path, format="JPEG")
        
        sample = {
            "uid": uid,
            "task": task["id"],
            "image": img_path,
            "question": q,
            "answers": a,
            "hash": img_hash
        }
        splits_data[target_split].append(sample)
        if uid_counter % 10 == 0:
            print(f"[{task_id}] Collected {train_collected}/{targets['train']} train, {dev_collected}/{targets['dev']} dev...", flush=True)
            
    print(f"Processing {task['id']} - Test", flush=True)
    ds_test = load_dataset(task["repo"], split=task["test_src"], streaming=True)
    ds_test = ds_test.shuffle(seed=0, buffer_size=5000)
    
    test_collected = 0
    
    for item in ds_test:
        if test_collected >= targets["test"]:
            break
            
        q = item[task["q_field"]]
        a = item[task["a_field"]]
        if task["id"] == "spatial_reasoning":
            a = ["True"] if a == 1 else ["False"]
            img_url = item["image_link"]
            img = download_image(img_url)
            if img is None: continue
        else:
            img = item["image"]
            
        img_hash, img_pil = get_image_hash(img)
        if img_hash in seen_hashes:
            continue
        seen_hashes.add(img_hash)
        
        uid = f"v3_{uid_counter:05d}"
        uid_counter += 1
        
        img_path = f"data/v3/images/{uid}.jpg"
        img_pil.save(img_path, format="JPEG")
        
        sample = {
            "uid": uid,
            "task": task["id"],
            "image": img_path,
            "question": q,
            "answers": a,
            "hash": img_hash
        }
        splits_data["test"].append(sample)
        test_collected += 1
        if test_collected % 10 == 0:
            print(f"[{task_id}] Collected {test_collected}/{targets['test']} test...", flush=True)

    for split in ["train", "dev", "test"]:
        path = f"data/v3/temp/{task_id}_{split}.jsonl"
        with open(path, "w") as f:
            for s in splits_data[split]:
                f.write(json.dumps(s) + "\n")
    print(f"Finished {task_id}")

def merge():
    tasks = ["document_ocr", "chart_qa", "spatial_reasoning"]
    splits = ["train", "dev", "test"]
    
    merged_data = {"train": [], "dev": [], "test": []}
    counts = defaultdict(lambda: defaultdict(int))
    split_hashes = {"train": set(), "dev": set(), "test": set()}
    all_hashes = set()
    
    for split in splits:
        for task_id in tasks:
            path = f"data/v3/temp/{task_id}_{split}.jsonl"
            if not os.path.exists(path):
                print(f"Warning: Missing {path}")
                continue
            with open(path, "r") as f:
                for line in f:
                    item = json.loads(line)
                    h = item.pop("hash")
                    
                    if h in all_hashes:
                        print(f"CRITICAL OVERLAP DETECTED FOR HASH {h}")
                        assert False, f"Image {h} appears in multiple splits or tasks!"
                        
                    all_hashes.add(h)
                    split_hashes[split].add(h)
                    
                    merged_data[split].append(item)
                    counts[split][task_id] += 1
                    
    assert len(split_hashes["train"].intersection(split_hashes["dev"])) == 0
    assert len(split_hashes["train"].intersection(split_hashes["test"])) == 0
    assert len(split_hashes["dev"].intersection(split_hashes["test"])) == 0
    print("Overlap check: PASS (No images appear in two splits)", flush=True)
    
    manifest = {
        "counts": {k: dict(v) for k, v in counts.items()},
        "sources": {
            "document_ocr": {"repo": "pixparse/docvqa-single-page-questions", "train_src": "train", "test_src": "validation"},
            "chart_qa": {"repo": "HuggingFaceM4/ChartQA", "train_src": "train", "test_src": "test"},
            "spatial_reasoning": {"repo": "cambridgeltl/vsr_random", "train_src": "train", "test_src": "test"}
        },
        "sha256": {}
    }
    
    for split in splits:
        path = f"data/v3/{split}.jsonl"
        with open(path, "w") as f:
            for s in merged_data[split]:
                f.write(json.dumps(s) + "\n")
        
        with open(path, "rb") as f:
            manifest["sha256"][split] = hashlib.sha256(f.read()).hexdigest()
            
    with open("data/v3/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("Counts:", flush=True)
    for split, task_counts in counts.items():
        print(f"  {split}: {dict(task_counts)}", flush=True)
        
    try:
        with open("data/smoke_raw/train.json", "r") as f:
            old_train = json.load(f)
        with open("data/smoke.jsonl", "w") as f:
            for i, item in enumerate(old_train):
                smoke_item = {
                    "uid": f"smoke_{i:03d}",
                    "task": "spatial_reasoning",
                    "image": item.get("image", ""),
                    "question": item.get("question", ""),
                    "answers": [item.get("answer", "")]
                }
                if "docvqa" in item.get("image", ""): smoke_item["task"] = "document_ocr"
                elif "chartqa" in item.get("image", ""): smoke_item["task"] = "chart_qa"
                f.write(json.dumps(smoke_item) + "\n")
        print("Smoke data built from old train.json.", flush=True)
    except Exception as e:
        print(f"Could not build smoke data from train.json (likely missing): {e}", flush=True)
        print("Building smoke.jsonl from new samples instead.", flush=True)
        with open("data/smoke.jsonl", "w") as f:
            idx = 0
            for split in ["train", "dev", "test"]:
                for s in merged_data[split]:
                    if idx >= 300: break
                    item = dict(s)
                    item["uid"] = f"smoke_{idx:03d}"
                    f.write(json.dumps(item) + "\n")
                    idx += 1
        print("Smoke data built from fallback.", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["document_ocr", "chart_qa", "spatial_reasoning"], help="Build specific task")
    parser.add_argument("--merge", action="store_true", help="Merge all tasks")
    args = parser.parse_args()
    
    if args.task:
        build_task(args.task)
    elif args.merge:
        merge()
    else:
        print("Specify --task or --merge")
