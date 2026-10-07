"""
T6 — teacher cache (Kaggle GPU).

Teacher = the same Qwen2-VL-2B, UNQUANTIZED (fp16 by default), with exactly the training prompts,
processor and max_pixels (from the config). One file per sample, keyed by uid:
    <cache_dir>/<uid>.pt = {probs [n,k] fp16, indices [n,k] int32, mask [L-1] bool, seq_len L,
                            topk_mass [n] fp32, teacher_top1 [n], temperature, k}

    python scripts/cache_teacher.py                                 # train + dev, resumes if interrupted
    CUDA_VISIBLE_DEVICES=1 python scripts/cache_teacher.py --shard 1 --num_shards 2   # second T4
    python scripts/cache_teacher.py --check                         # integrity report only
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from tqdm import tqdm

from kd import answer_shift_mask, teacher_targets


def safe_uid(uid):
    return str(uid).replace("/", "_").replace("\\", "_")


def cache_path(cache_dir, uid):
    return os.path.join(cache_dir, f"{safe_uid(uid)}.pt")


def load_uids(jsonl):
    with open(jsonl) as f:
        return [json.loads(l)["uid"] for l in f if l.strip()]


@torch.no_grad()
def cache_one(teacher, inputs, token_ids, temperature, k):
    """inputs: processor output for ONE sample (prompt + answer, no padding)."""
    ids = inputs["input_ids"][0]
    mask = answer_shift_mask(ids.cpu(), token_ids)
    if not mask.any():
        raise RuntimeError("no answer tokens in sample")
    logits = teacher(**inputs, use_cache=False).logits[0]
    entry = teacher_targets(logits, mask.to(logits.device), temperature, k)
    entry["mask"] = mask
    return entry


def check(cache_dir, splits, token_ids=None):
    """Integrity: every uid has a file; shapes agree; no NaN/inf; probability sanity."""
    ok, n_files, n_tok, masses, top1_frac, problems = True, 0, 0, [], [], []
    for split in splits:
        for uid in load_uids(split):
            p = cache_path(cache_dir, uid)
            if not os.path.exists(p):
                problems.append(f"missing {uid}")
                continue
            c = torch.load(p, map_location="cpu")
            n_files += 1
            n = int(c["mask"].sum())
            if c["probs"].shape[0] != n or c["indices"].shape[0] != n or c["mask"].numel() != c["seq_len"] - 1:
                problems.append(f"shape mismatch {uid}")
            if not torch.isfinite(c["probs"].float()).all() or not torch.isfinite(c["topk_mass"]).all():
                problems.append(f"nan/inf {uid}")
            if (c["probs"].float().sum(-1) > 1.001).any():
                problems.append(f"probs sum > 1 {uid}")
            n_tok += n
            masses.append(c["topk_mass"].numpy())
    if problems:
        ok = False
        print(f"{len(problems)} problems, first 10:", *problems[:10], sep="\n  ")
    m = np.concatenate(masses) if masses else np.zeros(1)
    print(f"files: {n_files} | answer tokens: {n_tok} | topk_mass mean={m.mean():.4f} "
          f"p5={np.percentile(m, 5):.4f} min={m.min():.4f}")
    print("INTEGRITY", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--splits", nargs="+", default=None, help="default: <data.dir>/train.jsonl dev.jsonl")
    ap.add_argument("--cache_dir", default=None, help="default: data.teacher_cache from the config")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--check", action="store_true", help="only run the integrity check")
    args = ap.parse_args()

    from config import load
    cfg = load(args.config)
    splits = args.splits or [os.path.join(cfg.data.dir, f"{s}.jsonl") for s in ("train", "dev")]
    cache_dir = args.cache_dir or cfg.data.teacher_cache
    os.makedirs(cache_dir, exist_ok=True)
    if args.check:
        sys.exit(0 if check(cache_dir, splits) else 1)

    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from data import VQADataset
    from model import resolve_token_ids
    dtype = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.data.min_pixels,
                                              max_pixels=cfg.data.max_pixels)
    teacher = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, torch_dtype=dtype,
                                                              device_map={"": 0}).eval()
    token_ids = resolve_token_ids(processor)
    T = cfg.loss.kd_temperature
    meta = {"model_id": cfg.model_id, "dtype": args.dtype, "temperature": T, "k": args.topk,
            "min_pixels": cfg.data.min_pixels, "max_pixels": cfg.data.max_pixels}
    with open(os.path.join(cache_dir, "cache_meta.yaml"), "w") as f:
        f.write("".join(f"{k}: {v}\n" for k, v in meta.items()))
    print("teacher:", meta)

    agree, total = 0, 0
    for split in splits:
        ds = VQADataset(split, processor)
        idx = [i for i in range(len(ds)) if i % args.num_shards == args.shard]
        for i in tqdm(idx, desc=os.path.basename(split)):
            uid = ds.data[i]["uid"]
            p = cache_path(cache_dir, uid)
            if os.path.exists(p) and not args.overwrite:
                continue
            item = ds[i]
            inputs = processor(text=[item["text"]], images=[item["image"]], return_tensors="pt")
            inputs = {k: v.to(teacher.device) for k, v in inputs.items()}
            entry = cache_one(teacher, inputs, token_ids, T, args.topk)
            if not torch.isfinite(entry["probs"].float()).all():
                sys.exit(f"non-finite teacher probs for {uid}: rerun with --dtype fp32")
            target = inputs["input_ids"][0, 1:][entry["mask"].to(teacher.device)].cpu()
            agree += int((entry["teacher_top1"] == target).sum()); total += target.numel()
            torch.save(entry, p + ".tmp")
            os.replace(p + ".tmp", p)                     # atomic: no half-written files on timeout
    if total:
        print(f"teacher top-1 == gold answer token on {agree}/{total} = {agree / total:.3f} of answer tokens")
    if args.num_shards == 1:
        check(cache_dir, splits)
    else:
        print("shard done; run --check after all shards finish")


if __name__ == "__main__":
    main()
