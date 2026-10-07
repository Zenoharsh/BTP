"""
T5 — real-model sanity check. Run on Kaggle (GPU) BEFORE any caching or training.

    python scripts/check_real_model.py --data data/smoke.jsonl            # real Qwen2-VL-2B, NF4, GPU
    python scripts/check_real_model.py --tiny                             # CPU self-test of this script

Checks (all must PASS):
  1. Identity  : logits of the unmodified NF4 model == logits after MoE surgery with zero-init LoRA
  2. Grad flow : CE backward reaches LoRA B; after one optimizer step it also reaches LoRA A and the router
  3. Generate  : greedy generation works for route_tokens="all" and "text" and returns non-empty text
  4. Report    : image tokens / sequence length per sample, peak VRAM, NaN/inf check, trainable %
"""
import argparse
import json
import os
import sys
import types

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn.functional as F

from layers import moe_layers
from model import apply_moe_surgery, resolve_token_ids
from utils import get_answer_labels

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}", flush=True)


# --------------------------------------------------------------------------- inputs
def load_samples(path, n_per_task=1):
    """First sample of each task from a jsonl; image paths relative to the jsonl's folder."""
    from PIL import Image
    root = os.path.dirname(os.path.abspath(path))
    seen, out = {}, []
    with open(path) as f:
        for line in f:
            s = json.loads(line)
            if seen.get(s["task"], 0) >= n_per_task:
                continue
            seen[s["task"]] = seen.get(s["task"], 0) + 1
            s["image_obj"] = Image.open(os.path.join(root, s["image"])).convert("RGB")
            out.append(s)
    return out


def encode(processor, sample, with_answer):
    from prompts import build_prompt
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": build_prompt(sample["task"], sample["question"])}]}]
    if with_answer:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": str(sample["answers"][0])}]})
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=not with_answer)
    return processor(text=[text], images=[sample["image_obj"]], return_tensors="pt")


# --------------------------------------------------------------------------- loaders
def load_real(args, cfg):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    dtype = {"fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.data.min_pixels,
                                              max_pixels=cfg.data.max_pixels)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, quantization_config=bnb,
                                                            torch_dtype=dtype, device_map={"": 0})
    samples = load_samples(args.data)
    batches = [(s["task"], encode(processor, s, True), encode(processor, s, False)) for s in samples]
    return model, processor, resolve_token_ids(processor), batches


def load_tiny(args, cfg):
    """Tiny random Qwen2-VL (real HF class) + synthetic inputs, for a CPU test of this script."""
    from transformers import Qwen2VLConfig, Qwen2VLForConditionalGeneration
    torch.manual_seed(0)
    IMG, VS, VE, IMS, IME, ASSIST = 90, 91, 92, 93, 94, 95
    conf = Qwen2VLConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]},
        image_token_id=IMG, video_token_id=96, vision_start_token_id=VS, vision_end_token_id=VE,
        vision_config={"depth": 1, "embed_dim": 32, "hidden_size": 64, "num_heads": 2,
                       "in_chans": 3, "patch_size": 14, "spatial_merge_size": 2, "temporal_patch_size": 2})
    model = Qwen2VLForConditionalGeneration(conf).float().eval()
    grid = torch.tensor([[1, 4, 4]])                      # 16 patches -> 4 image tokens after 2x2 merge
    pv = torch.randn(16, 3 * 2 * 14 * 14)
    prompt = [IMS, 5, 6, VS] + [IMG] * 4 + [VE, 7, 8, IME, 9, IMS, ASSIST, 10]
    full = prompt + [11, 12, IME, 9]
    mk = lambda ids: types.SimpleNamespace(**{"input_ids": torch.tensor([ids]),
                                              "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                                              "pixel_values": pv, "image_grid_thw": grid,
                                              "mm_token_type_ids": (torch.tensor([ids]) == IMG).long()})
    as_dict = lambda ns: dict(vars(ns))
    token_ids = {"image_pad": IMG, "im_start": IMS, "im_end": IME, "assistant": ASSIST}
    tok = types.SimpleNamespace(decode=lambda ids, **k: " ".join(map(str, ids.tolist())), eos_token_id=IME)
    processor = types.SimpleNamespace(tokenizer=tok)
    return model, processor, token_ids, [("tiny", as_dict(mk(full)), as_dict(mk(prompt)))]


# --------------------------------------------------------------------------- checks
def to_dev(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def logits_of(model, batch):
    return model(**batch, use_cache=False).logits.float()


def answer_ce(model, batch, token_ids):
    labels = get_answer_labels(batch["input_ids"], token_ids["im_start"], token_ids["assistant"], token_ids["im_end"])
    logits = model(**batch, use_cache=False).logits[:, :-1].float()
    lab = labels[:, 1:]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), lab.reshape(-1), ignore_index=-100), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/moe.yaml")
    ap.add_argument("--data", default="data/smoke.jsonl")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()

    from config import load
    cfg = load(args.config)
    if args.tiny:
        cfg.moe.rank, cfg.moe.alpha = 4, 8

    if not args.tiny and not torch.cuda.is_available():
        sys.exit("CUDA not available: run on Kaggle (or use --tiny for a CPU self-test).")

    model, processor, token_ids, batches = (load_tiny if args.tiny else load_real)(args, cfg)
    device = next(model.parameters()).device
    batches = [(t, to_dev(full, device), to_dev(prm, device)) for t, full, prm in batches]
    print(f"token ids: {token_ids}")

    # ---- report inputs
    for task, full, _ in batches:
        n_img = int((full["input_ids"] == token_ids["image_pad"]).sum())
        labels = get_answer_labels(full["input_ids"], token_ids["im_start"], token_ids["assistant"], token_ids["im_end"])
        ans = processor.tokenizer.decode(full["input_ids"][labels != -100])
        print(f"  [{task}] seq_len={full['input_ids'].shape[1]} image_tokens={n_img} supervised={ans!r}")
        check(f"answer mask non-empty and ends with im_end [{task}]",
              (labels != -100).any() and int(full["input_ids"][labels != -100][-1]) == token_ids["im_end"])

    # ---- 1. identity
    model.eval()
    ref = [logits_of(model, full) for _, full, _ in batches]
    for r, (task, _, _) in zip(ref, batches):
        check(f"reference logits finite [{task}]", torch.isfinite(r).all())

    apply_moe_surgery(model, cfg, token_ids["image_pad"])
    print(f"MoE layers: {len(moe_layers(model))}")
    for r, (task, full, _) in zip(ref, batches):
        new = logits_of(model, full)
        d = (new - r).abs().max().item()
        check(f"identity at init [{task}]", d < 1e-2 and torch.equal(new.argmax(-1), r.argmax(-1)), f"max|diff|={d:.2e}")

    # ---- 2. gradient flow
    model.train()
    if not args.tiny:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    _, full, _ = batches[0]
    ce, _ = answer_ce(model, full, token_ids)
    check("CE finite", torch.isfinite(ce), f"CE={ce.item():.4f}")
    ce.backward()
    L = moe_layers(model)
    gB = sum(l.lora_down.B.grad.abs().sum().item() for l in L)
    check("grad reaches LoRA B (step 0)", gB > 0, f"sum|g|={gB:.3e}")
    opt.step(); opt.zero_grad(set_to_none=True)
    ce2, _ = answer_ce(model, full, token_ids)
    ce2.backward()
    gA = sum(l.lora_gate.A.grad.abs().sum().item() for l in L)
    check("grad reaches LoRA A (after 1 step)", gA > 0, f"sum|g|={gA:.3e}")
    if L[0].router is not None:
        gR = sum(l.router.gate.weight.grad.abs().sum().item() for l in L)
        check("grad reaches router from task loss", gR > 0, f"sum|g|={gR:.3e}")
    finite = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
    check("all grads finite", finite)
    opt.zero_grad(set_to_none=True)
    n_tr = sum(p.numel() for p in params)
    print(f"trainable params: {n_tr:,}")

    # ---- 3. generation, both routing modes
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable") and not args.tiny:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    for mode in ("all", "text"):
        for l in L:
            l.route_tokens = mode
        for task, _, prm in batches:
            with torch.no_grad():
                out = model.generate(**prm, max_new_tokens=16, do_sample=False,
                                     eos_token_id=processor.tokenizer.eos_token_id if args.tiny else None)
            new_tokens = out[0, prm["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
            check(f"generate route={mode} [{task}]", new_tokens.numel() > 0, f"-> {text!r}")
        if mode == "text":
            n_img = int((batches[-1][2]["input_ids"] == token_ids["image_pad"]).sum())
            check("text-routing hook saw image tokens", L[0].token_is_image is not None or n_img == 0)

    if torch.cuda.is_available():
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")

    failed = [n for n, ok in RESULTS if not ok]
    print("\n" + ("ALL T5 CHECKS PASSED" if not failed else f"FAILED: {failed}"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()