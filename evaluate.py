"""
T9 — evaluation.

    # untouched base model (T8 resolution sweep)
    python evaluate.py --base --split data/v3/dev.jsonl --max_pixels 401408 --out runs/b0_512/dev_preds.jsonl
    # trained adapters (config must be the one the run was trained with)
    python evaluate.py --config configs/moe.yaml --adapters runs/moe/best --split data/v3/dev.jsonl \
                       --out runs/moe/dev_preds.jsonl --record_routing

train.py calls evaluate(...) directly every epoch, with the model it already has in memory.

Protocol: batch 1, greedy, max_new_tokens=32, stop on <|im_end|> or newline, prediction = text before
the first newline. Resolution is set only through the processor (min_pixels from the config,
max_pixels from the config unless --max_pixels is given). All processor outputs go to the model.
"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

from metrics import compute_metric, exact_match
from prompts import build_prompt


# --------------------------------------------------------------------------- data
def load_samples(path, limit_per_task=None):
    """Samples from a jsonl. limit_per_task keeps the first N of EACH task (dev/test are grouped by
    task, so a plain head-N would evaluate only one task)."""
    out, seen = [], defaultdict(int)
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if limit_per_task and seen[s["task"]] >= limit_per_task:
                continue
            seen[s["task"]] += 1
            out.append(s)
    return out


def encode_prompt(processor, sample, root):
    from PIL import Image
    msgs = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": build_prompt(sample["task"], sample["question"])}]}]
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    image = Image.open(os.path.join(root, sample["image"])).convert("RGB")
    return dict(processor(text=[text], images=[image], return_tensors="pt"))


def stop_token_ids(processor, token_ids):
    ids = [token_ids["im_end"]]
    for s in ("\n", "\n\n"):
        try:
            t = processor.tokenizer.encode(s, add_special_tokens=False)
        except Exception:
            continue
        if len(t) == 1 and t[0] not in ids:
            ids.append(t[0])
    return ids


# --------------------------------------------------------------------------- routing
class RoutingRecorder:
    """Per-sample, per-layer expert counts and summed gate values over TEXT tokens of the prompt
    (prefill pass). gate of the chosen expert = prob * E / k, as in MoELayer (A1).
    Hooks the routers directly, so it works for route_tokens="all" and "text"."""

    def __init__(self, model, image_token_id):
        from layers import moe_layers
        self.layers = [l for l in moe_layers(model) if l.router is not None]
        self.E = self.layers[0].num_experts if self.layers else 0
        self.image_token_id = image_token_id
        self.active, self.text_mask = False, None
        self._cur = None
        self.counts, self.gate_sum = [], []
        self.handles = [l.router.register_forward_hook(self._hook(i, l)) for i, l in enumerate(self.layers)]

    def _hook(self, i, layer):
        def fn(mod, args, out):
            if not self.active:
                return
            _, probs, top_idx = out
            m = self.text_mask.to(probs.device)
            if getattr(layer, "route_level", "token") == "sequence":
                if probs.shape[0] != 1:
                    raise ValueError("routing recording supports batch size 1 only")
                n_text = int(m.sum())                     # one decision for the whole sample, weighted
                e = int(top_idx[0, 0])                    # by its number of text tokens
                g = float(probs[0, e]) * (self.E / layer.top_k)
                self._cur[0][i, e] += n_text
                self._cur[1][i, e] += g * n_text
                return
            if m.numel() != probs.shape[0]:
                return
            e = top_idx[m, 0]
            g = probs[m].gather(-1, top_idx[m, :1]).squeeze(-1).float() * (self.E / layer.top_k)
            self._cur[0][i] += torch.bincount(e, minlength=self.E).cpu().numpy()
            self._cur[1][i] += torch.zeros(self.E, device=g.device).index_add_(0, e, g).cpu().numpy()
        return fn

    @torch.no_grad()
    def record(self, model, inputs):
        L = len(self.layers)
        self._cur = (np.zeros((L, self.E), np.int64), np.zeros((L, self.E), np.float64))
        self.text_mask = (inputs["input_ids"] != self.image_token_id).reshape(-1)
        self.active = True
        try:
            model(**inputs, use_cache=False)
        finally:
            self.active = False
        self.counts.append(self._cur[0])
        self.gate_sum.append(self._cur[1])

    def save(self, path, samples):
        np.savez(path, counts=np.stack(self.counts), gate_sum=np.stack(self.gate_sum),
                 tasks=np.array([s["task"] for s in samples]), uids=np.array([str(s["uid"]) for s in samples]))

    def remove(self):
        for h in self.handles:
            h.remove()


# --------------------------------------------------------------------------- evaluation
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def evaluate(model, processor, samples, root, token_ids, out_path=None, max_new_tokens=32,
             routing_path=None, encode_fn=encode_prompt, warmup=2, progress=True):
    """Greedy batch-1 evaluation. Returns {"per_task": {...}, "macro": float, "macro_em": float, ...}.
    Writes preds.jsonl (uid, task, pred, answers, score, em, latency, img_tokens) if out_path is given,
    and the routing .npz if routing_path is given."""
    from layers import set_telemetry
    was_training = model.training
    model.eval()
    prev_telemetry = set_telemetry(model, False)     # no routing telemetry/GPU syncs while timing
    device = next(model.parameters()).device
    eos = stop_token_ids(processor, token_ids)
    # use_cache=True explicitly: build_model sets config.use_cache=False for training
    gen_kw = dict(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=eos,
                  pad_token_id=token_ids["im_end"], use_cache=True)
    rec = RoutingRecorder(model, token_ids["image_pad"]) if routing_path else None
    if rec is not None and not rec.layers:
        raise ValueError("--record_routing needs a routed MoE model (num_experts > 1, not --base)")

    def gen(inputs):
        return model.generate(**inputs, **gen_kw)

    # warm-up (CUDA kernels, allocator) so the first timed sample is not an outlier
    for s in samples[:warmup]:
        gen({k: v.to(device) for k, v in encode_fn(processor, s, root).items()})

    rows = []
    try:
        for s in tqdm(samples, desc="eval", disable=not progress):
            inputs = {k: v.to(device) for k, v in encode_fn(processor, s, root).items()}
            n_img = int((inputs["input_ids"] == token_ids["image_pad"]).sum())
            if rec is not None:
                rec.record(model, inputs)
            _sync(device)
            t0 = time.perf_counter()
            out = gen(inputs)
            _sync(device)
            latency = time.perf_counter() - t0
            new = out[0, inputs["input_ids"].shape[1]:]
            raw = processor.tokenizer.decode(new, skip_special_tokens=True)
            pred = raw.split("\n")[0].strip()
            rows.append({"uid": s["uid"], "task": s["task"], "pred": pred, "raw": raw,
                         "answers": s["answers"], "score": compute_metric(s["task"], pred, s["answers"]),
                         "em": exact_match(pred, s["answers"]), "latency": latency,
                         "img_tokens": n_img, "new_tokens": int(new.numel())})
    finally:
        if rec is not None:
            rec.remove()
        model.train(was_training)
        set_telemetry(model, prev_telemetry)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if rec is not None:
        rec.save(routing_path, samples)
    return summarize(rows)


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    per_task = {t: {"n": len(rs),
                    "score": float(np.mean([r["score"] for r in rs])),
                    "em": float(np.mean([r["em"] for r in rs])),
                    "latency_mean": float(np.mean([r["latency"] for r in rs])),
                    "img_tokens_mean": float(np.mean([r["img_tokens"] for r in rs]))}
                for t, rs in sorted(by.items())}
    return {"per_task": per_task,
            "macro": float(np.mean([v["score"] for v in per_task.values()])) if per_task else 0.0,
            "macro_em": float(np.mean([v["em"] for v in per_task.values()])) if per_task else 0.0,
            "latency_mean": float(np.mean([r["latency"] for r in rows])) if rows else 0.0,
            "img_tokens_mean": float(np.mean([r["img_tokens"] for r in rows])) if rows else 0.0}


def print_summary(res):
    print(f"\n{'task':<20}{'n':>5}{'score':>9}{'EM':>8}{'img_tok':>9}{'lat(s)':>8}")
    for t, v in res["per_task"].items():
        print(f"{t:<20}{v['n']:>5}{v['score']:>9.4f}{v['em']:>8.4f}{v['img_tokens_mean']:>9.1f}{v['latency_mean']:>8.3f}")
    print(f"{'macro':<20}{'':>5}{res['macro']:>9.4f}{res['macro_em']:>8.4f}"
          f"{res['img_tokens_mean']:>9.1f}{res['latency_mean']:>8.3f}")


# --------------------------------------------------------------------------- CLI
def load_for_eval(cfg, max_pixels, base=False, adapters=None, device_index=0):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    from model import apply_moe_surgery, load_adapters, resolve_token_ids
    min_pixels = min(cfg.data.min_pixels, max_pixels)
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        cfg.model_id, quantization_config=bnb, torch_dtype=torch.float16, device_map={"": device_index})
    token_ids = resolve_token_ids(processor)
    if not base:
        apply_moe_surgery(model, cfg, token_ids["image_pad"], token_ids["im_start"])
        load_adapters(model, adapters)
    return model, processor, token_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--split", default="data/v3/dev.jsonl")
    ap.add_argument("--out", default=None, help="preds jsonl (default: runs/eval/<split>_preds.jsonl)")
    ap.add_argument("--max_pixels", type=int, default=None, help="default: data.max_pixels from the config")
    ap.add_argument("--limit", type=int, default=None, help="first N samples PER TASK")
    ap.add_argument("--base", action="store_true", help="untouched base model, no surgery")
    ap.add_argument("--adapters", default=None, help="folder with adapters.pt (required unless --base)")
    ap.add_argument("--record_routing", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    if args.base == bool(args.adapters):
        ap.error("give exactly one of --base or --adapters")
    if args.base and args.record_routing:
        ap.error("--record_routing needs --adapters (the base model has no router)")

    from config import load
    cfg = load(args.config)
    max_pixels = args.max_pixels or cfg.data.max_pixels
    split_name = os.path.splitext(os.path.basename(args.split))[0]
    out = args.out or os.path.join("runs", "eval", f"{split_name}_preds.jsonl")
    routing = os.path.join(os.path.dirname(os.path.abspath(out)), f"routing_{split_name}.npz") \
        if args.record_routing else None

    model, processor, token_ids = load_for_eval(cfg, max_pixels, args.base, args.adapters)
    samples = load_samples(args.split, args.limit)
    print(f"{len(samples)} samples | max_pixels={max_pixels} | {'BASE' if args.base else args.adapters}")
    res = evaluate(model, processor, samples, os.path.dirname(os.path.abspath(args.split)), token_ids,
                   out_path=out, max_new_tokens=args.max_new_tokens, routing_path=routing)
    print_summary(res)
    res.update({"split": args.split, "max_pixels": max_pixels, "base": args.base,
                "adapters": args.adapters, "config": args.config})
    with open(os.path.splitext(out)[0] + "_summary.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"preds -> {out}" + (f"\nrouting -> {routing}" if routing else ""))


if __name__ == "__main__":
    main()
