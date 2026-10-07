"""
T7 — training (v3).

    python train.py --config configs/moe.yaml --seed 0 [--set train.lr=1e-4 ...]
    CUDA_VISIBLE_DEVICES=1 python train.py --config configs/dense16.yaml --seed 0     # second T4

Loss per micro-batch:  ce * CE(answer tokens)  +  kd * sparse-KD(teacher cache)  +  router aux (MoE).
AdamW with two param groups (LoRA: train.lr, router: train.router_lr), cosine schedule with warmup,
grad clipping; an optimizer step every grad_accum micro-batches and at the end of every epoch.
Outputs in runs/<run_id>/: train_log.jsonl (one line per optimizer step + one per dev eval),
best/ and last/ adapters, dev predictions per evaluated epoch, config.json, summary.json.
"""
import argparse
import dataclasses
import functools
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from kd import answer_shift_mask, sparse_kd
from layers import moe_layers, total_aux_loss
from model import save_adapters
from utils import get_answer_labels

ROUTING_KEYS = ("entropy_token_mean", "overflow_rate", "load_cv")


# --------------------------------------------------------------------------- setup
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32           # differs per worker, derived from the loader generator
    np.random.seed(s)
    random.seed(s)


def param_groups(model, cfg):
    lora, router = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (router if ".router." in n else lora).append(p)
    groups = [{"params": lora, "lr": cfg.train.lr, "name": "lora"}]
    if router:
        groups.append({"params": router, "lr": cfg.train.router_lr, "name": "router"})
    return groups


def make_scheduler(opt, cfg, total_steps):
    from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup
    warm = int(round(cfg.train.warmup_ratio * total_steps))
    if cfg.train.scheduler == "cosine":
        return get_cosine_schedule_with_warmup(opt, warm, total_steps)
    if cfg.train.scheduler == "constant":
        return get_constant_schedule_with_warmup(opt, warm)
    raise ValueError(f"unknown scheduler {cfg.train.scheduler}")


class TeacherCache:
    """All cache entries for the given uids, loaded once into RAM (they are small)."""

    def __init__(self, cache_dir, uids):
        from scripts.cache_teacher import cache_path
        missing = [u for u in uids if not os.path.exists(cache_path(cache_dir, u))]
        if missing:
            raise FileNotFoundError(f"{len(missing)} teacher-cache files missing in {cache_dir} "
                                    f"(e.g. {missing[:3]}). Run scripts/cache_teacher.py first, "
                                    f"or train with --set loss.kd=0.")
        self.entries = {str(u): torch.load(cache_path(cache_dir, u), map_location="cpu") for u in uids}

    def __getitem__(self, uid):
        return self.entries[str(uid)]


# --------------------------------------------------------------------------- loss
def micro_batch_loss(model, batch, token_ids, cfg, cache=None):
    """Returns (loss tensor, dict of floats). batch = collate_fn output (+ 'uids', 'tasks')."""
    batch = dict(batch)
    uids = batch.pop("uids")
    batch.pop("tasks", None)
    device = next(p for p in model.parameters() if p.requires_grad).device
    inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    logits = model(**inputs, use_cache=False).logits                   # do NOT pass labels

    ids = inputs["input_ids"]
    labels = get_answer_labels(ids, token_ids["im_start"], token_ids["assistant"], token_ids["im_end"])
    lab = labels[:, 1:]
    sup = lab != -100
    if not sup.any():
        raise RuntimeError(f"no supervised tokens in batch {uids}")
    ce = F.cross_entropy(logits[:, :-1][sup].float(), lab[sup])

    kd = torch.zeros((), device=logits.device)
    if cfg.loss.kd > 0:
        am = inputs.get("attention_mask", torch.ones_like(ids)).bool()
        per = []
        for i, uid in enumerate(uids):
            c = cache[uid]
            ids_i = ids[i][am[i]]                                          # works for left or right padding
            if c["seq_len"] != ids_i.numel():
                raise RuntimeError(f"teacher/student length mismatch for {uid}: cache {c['seq_len']} vs "
                                   f"{ids_i.numel()} (different max_pixels or prompt?)")
            mask = answer_shift_mask(ids_i.cpu(), token_ids)
            if not torch.equal(mask, c["mask"]):
                raise RuntimeError(f"teacher/student answer mask mismatch for {uid}")
            s = logits[i][am[i]][:-1][mask.to(logits.device)]
            per.append(sparse_kd(s, c["indices"].to(s.device), c["probs"].to(s.device), cfg.loss.kd_temperature))
        kd = torch.stack(per).mean()

    aux = total_aux_loss(model)
    loss = cfg.loss.ce * ce + cfg.loss.kd * kd + (aux if aux is not None else 0.0)
    return loss, {"ce": ce.item(), "kd": kd.item(), "aux": aux.item() if aux is not None else 0.0,
                  "loss": loss.item()}


# --------------------------------------------------------------------------- routing telemetry
class RoutingWindow:
    """Per-layer router telemetry accumulated over the micro-batches of one optimizer step (A8)."""

    def __init__(self, model):
        self.layers = [l for l in moe_layers(model) if l.router is not None]
        L = len(self.layers)
        self.report_layers = sorted({0, L // 2 - 1, L - 1}) if L else []
        self.reset()

    def reset(self):
        self.n = 0
        self.sums = {k: np.zeros(len(self.layers)) for k in ROUTING_KEYS}
        self.counts = None

    def add(self):
        if not self.layers:
            return
        ms = [l.metrics for l in self.layers]
        if any(not m for m in ms):
            return
        for k in ROUTING_KEYS:
            self.sums[k] += np.array([m[k] for m in ms], dtype=float)
        pc = np.array([m["post_counts"] for m in ms], dtype=float)
        self.counts = pc if self.counts is None else self.counts + pc
        self.n += 1

    def summary(self):
        if not self.layers or self.n == 0:
            return None
        out = {}
        for k in ROUTING_KEYS:
            per_layer = self.sums[k] / self.n
            out[f"{k}_mean"] = float(per_layer.mean())
            out[f"{k}_max"] = float(per_layer.max())
            out[f"{k}_min"] = float(per_layer.min())
        out["post_counts"] = {str(i): self.counts[i].tolist() for i in self.report_layers}
        return out


# --------------------------------------------------------------------------- loop
def train(cfg, model, token_ids, loader, run_dir, cache=None, dev_fn=None, max_steps=None,
          cfg_dict=None, log_every=10):
    """Generic loop (also used by the CPU tests). dev_fn(model, epoch) -> evaluate() summary."""
    os.makedirs(run_dir, exist_ok=True)
    ga, epochs = cfg.train.grad_accum, cfg.train.epochs
    n_micro = len(loader)
    steps_per_epoch = math.ceil(n_micro / ga)
    total_steps = steps_per_epoch * epochs if max_steps is None else min(max_steps, steps_per_epoch * epochs)

    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg.train.weight_decay)
    sched = make_scheduler(opt, cfg, total_steps)
    trainable = [p for g in opt.param_groups for p in g["params"]]
    routing = RoutingWindow(model)
    log_f = open(os.path.join(run_dir, "train_log.jsonl"), "a")
    on_cuda = torch.cuda.is_available() and trainable[0].is_cuda

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    step, best, best_epoch, history = 0, -1.0, None, []
    t_start = time.time()
    model.train()
    opt.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        win = {"ce": [], "kd": [], "aux": [], "loss": []}
        t0 = time.time()
        if on_cuda:
            torch.cuda.reset_peak_memory_stats()
        for j, batch in enumerate(loader):
            win_size = min(ga, n_micro - (j // ga) * ga)                   # last window may be short
            loss, parts = micro_batch_loss(model, batch, token_ids, cfg, cache)
            if not math.isfinite(parts["loss"]):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} micro-batch {j}: {parts}")
            (loss / win_size).backward()
            routing.add()
            for k in win:
                win[k].append(parts[k])

            if (j + 1) % ga == 0 or j + 1 == n_micro:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable, cfg.train.max_grad_norm)
                lrs = {g["name"]: g["lr"] for g in opt.param_groups}       # lr used for THIS step
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                rec = {"type": "step", "step": step, "epoch": epoch,
                       "lr": lrs,
                       **{k: float(np.mean(v)) for k, v in win.items()},
                       "grad_norm": float(gnorm), "sec_per_step": time.time() - t0,
                       "peak_vram_mib": torch.cuda.max_memory_allocated() / 2 ** 20 if on_cuda else 0.0}
                r = routing.summary()
                if r is not None:
                    rec["routing"] = r
                log(rec)
                if step % log_every == 0 or step == 1:
                    ent = f" ent={r['entropy_token_mean_mean']:.3f}/{r['entropy_token_mean_min']:.3f}" if r else ""
                    print(f"step {step}/{total_steps} ep {epoch} lr {rec['lr']['lora']:.2e} "
                          f"ce {rec['ce']:.4f} kd {rec['kd']:.4f} aux {rec['aux']:.4f}{ent} "
                          f"{rec['sec_per_step']:.1f}s/step vram {rec['peak_vram_mib']:.0f}MiB", flush=True)
                win = {k: [] for k in win}
                routing.reset()
                t0 = time.time()
                if on_cuda:
                    torch.cuda.reset_peak_memory_stats()
                if max_steps is not None and step >= max_steps:
                    break

        last_epoch = epoch == epochs or (max_steps is not None and step >= max_steps)
        if dev_fn is not None and (epoch % cfg.train.eval_every == 0 or last_epoch):
            res = dev_fn(model, epoch)
            history.append({"epoch": epoch, **res})
            log({"type": "eval", "epoch": epoch, "step": step, **res})
            print(f"[dev] epoch {epoch}: macro {res['macro']:.4f} "
                  + " ".join(f"{t}={v['score']:.3f}" for t, v in res["per_task"].items()), flush=True)
            if res["macro"] > best:
                best, best_epoch = res["macro"], epoch
                save_adapters(model, os.path.join(run_dir, "best"), cfg_dict,
                              {"epoch": epoch, "step": step, "dev": res})
        if last_epoch:
            break

    save_adapters(model, os.path.join(run_dir, "last"), cfg_dict, {"epoch": epoch, "step": step})
    log_f.close()
    summary = {"best_dev_macro": best if best_epoch else None, "best_epoch": best_epoch,
               "steps": step, "train_minutes": (time.time() - t_start) / 60, "history": history}
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/moe.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--set", action="append", default=[], help="override, e.g. --set train.lr=1e-4")
    ap.add_argument("--run_id", default=None, help="default: <config name>_s<seed>")
    ap.add_argument("--max_steps", type=int, default=None, help="stop early (smoke tests)")
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()

    from torch.utils.data import DataLoader
    from transformers import AutoProcessor
    from config import load
    from data import VQADataset, collate_fn
    from evaluate import evaluate, load_samples
    from model import build_model

    cfg = load(args.config)                       # also applies --set overrides from sys.argv
    set_seed(args.seed)
    run_id = args.run_id or f"{os.path.splitext(os.path.basename(args.config))[0]}_s{args.seed}"
    run_dir = os.path.join("runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    cfg_dict = {**dataclasses.asdict(cfg), "seed": args.seed, "config_file": args.config,
                "overrides": args.set}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)
    print(f"run {run_id}: {json.dumps(cfg_dict)}")

    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.data.min_pixels,
                                              max_pixels=cfg.data.max_pixels)
    model, token_ids = build_model(cfg, processor, device_index=args.device)

    train_path = os.path.join(cfg.data.dir, "train.jsonl")
    dev_path = os.path.join(cfg.data.dir, "dev.jsonl")
    ds = VQADataset(train_path, processor, limit=cfg.data.limit)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True, generator=g,
                        num_workers=cfg.train.num_workers,
                        worker_init_fn=seed_worker,
                        collate_fn=functools.partial(collate_fn, processor=processor))
    cache = TeacherCache(cfg.data.teacher_cache, [d["uid"] for d in ds.data]) if cfg.loss.kd > 0 else None

    dev_samples = load_samples(dev_path, cfg.data.dev_limit)
    dev_root = os.path.dirname(os.path.abspath(dev_path))

    def dev_fn(m, epoch):
        return evaluate(m, processor, dev_samples, dev_root, token_ids,
                        out_path=os.path.join(run_dir, f"dev_preds_ep{epoch}.jsonl"), warmup=1)

    print(f"train samples {len(ds)} | dev samples {len(dev_samples)} | "
          f"{math.ceil(len(loader) / cfg.train.grad_accum) * cfg.train.epochs} optimizer steps")
    summary = train(cfg, model, token_ids, loader, run_dir, cache=cache, dev_fn=dev_fn,
                    max_steps=args.max_steps, cfg_dict=cfg_dict)
    print(f"done: best dev macro {summary['best_dev_macro']} at epoch {summary['best_epoch']} "
          f"({summary['train_minutes']:.1f} min) -> {run_dir}")


if __name__ == "__main__":
    main()
