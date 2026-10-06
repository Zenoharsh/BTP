import json
from PIL import Image
from torch.utils.data import Dataset
from prompts import build_prompt

class VQADataset(Dataset):
    def __init__(self, jsonl_path, processor, limit=None):
        self.processor = processor
        self.data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                self.data.append(json.loads(line))
        if limit:
            self.data = self.data[:limit]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        uid = item["uid"]
        task = item["task"]
        question = item["question"]
        answer = str(item["answers"][0])
        image_path = item["image"]
        
        prompt = build_prompt(task, question)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}]
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224))
            
        return {
            "text": text,
            "image": image,
            "uid": uid,
            "task": task
        }

def collate_fn(batch, processor):
    texts = [b["text"] for b in batch]
    images = [b["image"] for b in batch]
    uids = [b["uid"] for b in batch]
    tasks = [b["task"] for b in batch]
    
    inputs = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt"
    )
    
    inputs["uids"] = uids
    inputs["tasks"] = tasks
    return inputs
