"""
T10 — routing statistics and deployment converters.

Inputs: trained MoE adapters + the routing file written by
    python evaluate.py --adapters runs/moe_s0/best --record_routing --out runs/moe_s0/dev_eval.jsonl
which holds, per sample and layer, expert counts and summed gate values over the prompt's TEXT tokens.

    python convert.py --routing runs/moe_s0/routing_dev.npz --mode stats
    python convert.py --adapters runs/moe_s0/best --routing runs/moe_s0/routing_dev.npz --mode full_merge --eval
    python convert.py ... --mode selective --k 0 4 8 14 28 --eval
    python convert.py ... --mode profile --eval
    python convert.py ... --mode full_merge --export exports/moe_s0_merged          # plain Qwen2-VL checkpoint
    python convert.py ... --mode profile --task document_ocr --export exports/moe_s0_doc
    python convert.py --config configs/dense16.yaml --adapters runs/dense16_s0/best --mode dense --export exports/dense16_s0

Gate vectors used in fixed mode (A7):
    full merge / non-selected layers : g[l] = E[g_e] over text tokens = f[l, e] * gbar[l, e]
    profile for task t               : g[l] = onehot(argmax_e f_task[t, l, e]) * gbar[l, e]
where f = expert frequency and gbar = mean gate value of tokens routed to that expert (prob * E / k).
"""
import argparse
import json
import os

import numpy as np
import torch

from layers import moe_layers

TASKS = ("chart_qa", "document_ocr", "spatial_reasoning")


# --------------------------------------------------------------------------- statistics
def mutual_information_bits(joint):
    """joint: [T, E] counts -> I(task; expert) in bits."""
    p = joint / max(joint.sum(), 1e-12)
    pt, pe = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pt @ pe)[nz])).sum())


def routing_stats(path):
    """Returns dict with f [L,E], gbar [L,E], f_task {t: [L,E]}, mi [L] (bits), tasks, n_tokens."""
    with np.load(path) as z:
        counts, gate_sum, tasks = z["counts"].astype(np.float64), z["gate_sum"], z["tasks"]
    S, L, E = counts.shape
    tot = counts.sum(0)                                                  # [L, E]
    f = tot / np.maximum(tot.sum(-1, keepdims=True), 1)
    gbar = gate_sum.sum(0) / np.maximum(tot, 1)
    gbar[tot == 0] = 0.0
    names = sorted(set(tasks.tolist()))
    per_task = {t: counts[tasks == t].sum(0) for t in names}             # [L, E] each
    f_task = {t: c / np.maximum(c.sum(-1, keepdims=True), 1) for t, c in per_task.items()}
    mi = np.array([mutual_information_bits(np.stack([per_task[t][l] for t in names])) for l in range(L)])
    return {"f": f, "gbar": gbar, "f_task": f_task, "mi": mi, "tasks": names,
            "n_tokens": int(tot[0].sum()), "n_samples": S, "max_mi": float(np.log2(min(len(names), E)))}


def print_stats(st):
    L, E = st["f"].shape
    print(f"{st['n_samples']} samples, {st['n_tokens']} text tokens per layer, {L} layers x {E} experts")
    print(f"MI(task; expert) bits (max possible {st['max_mi']:.3f}): mean {st['mi'].mean():.4f}, "
          f"max {st['mi'].max():.4f} at layer {int(st['mi'].argmax())}")
    order = np.argsort(-st["mi"])
    print("top-8 layers by MI:", ", ".join(f"L{l}={st['mi'][l]:.3f}" for l in order[:8]))
    ent = -(st["f"] * np.log(np.clip(st["f"], 1e-12, 1))).sum(-1)
    print(f"expert-usage entropy (nats, max {np.log(E):.3f}): mean {ent.mean():.3f}, min {ent.min():.3f}")
    print("dominant expert per task (share of tokens), layers with the highest MI:")
    for l in order[:4]:
        row = "  ".join(f"{t[:5]}->e{int(st['f_task'][t][l].argmax())}({st['f_task'][t][l].max():.2f})"
                        for t in st["tasks"])
        print(f"  L{l:<3} {row}")


# --------------------------------------------------------------------------- fixed-mode configurations
def _set_fixed(layer, g):
    layer.mode, layer.fixed_g = "fixed", torch.as_tensor(np.asarray(g, dtype=np.float32))


def set_routed(model):
    for l in moe_layers(model):
        l.mode, l.fixed_g = "routed", None


def apply_full_merge(model, st):
    for i, l in enumerate(moe_layers(model)):
        _set_fixed(l, st["f"][i] * st["gbar"][i])


def apply_selective(model, st, k):
    """Keep the k layers with the highest MI routed; merge the rest. k=0 -> full merge, k=L -> original."""
    keep = set(np.argsort(-st["mi"])[:k].tolist())
    for i, l in enumerate(moe_layers(model)):
        if i in keep:
            l.mode, l.fixed_g = "routed", None
        else:
            _set_fixed(l, st["f"][i] * st["gbar"][i])
    return sorted(keep)


def profile_gates(st, task):
    out = []
    for i in range(st["f"].shape[0]):
        e = int(st["f_task"][task][i].argmax())
        g = np.zeros(st["f"].shape[1])
        g[e] = st["gbar"][i, e]
        out.append(g)
    return np.stack(out)


def apply_profile(model, st, task):
    for i, l in enumerate(moe_layers(model)):
        _set_fixed(l, profile_gates(st, task)[i])


# --------------------------------------------------------------------------- export
@torch.no_grad()
def merge_into_base(model):
    """Fold every fixed-mode (or dense, E=1) MoE layer into its base MLP weights and remove the MoE
    wrapper, so the model is a standard Qwen2-VL. The base must be UNQUANTIZED."""
    from layers import get_lm_layers, MoELayer
    for block in get_lm_layers(model):
        m = block.mlp
        if not isinstance(m, MoELayer):
            continue
        if m.route_tokens == "text":
            raise ValueError("route_tokens='text' gives image tokens g=0; that cannot be folded into dense weights")
        if m.num_experts == 1:
            g = torch.ones(1)
        elif m.mode == "fixed":
            g = m.fixed_g.float()
        else:
            raise ValueError("a layer is still in routed mode; only full merge / profile / dense can be exported")
        for name, delta in m.merged_delta_weights(g).items():
            w = getattr(m.base_mlp, name).weight
            if not w.is_floating_point() or w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError("export needs an unquantized base model")
            w.copy_((w.float() + delta.to(w.device).float()).to(w.dtype))
        block.mlp = m.base_mlp


def export(cfg, adapters, out_dir, configure_fn):
    """Unquantized fp16 base on CPU -> surgery -> load adapters -> configure gates -> fold -> save."""
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from model import apply_moe_surgery, load_adapters, resolve_token_ids
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.data.min_pixels,
                                              max_pixels=cfg.data.max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, torch_dtype=torch.float16)
    apply_moe_surgery(model, cfg, resolve_token_ids(processor)["image_pad"])
    load_adapters(model, adapters)
    configure_fn(model)
    merge_into_base(model)
    model._forward_pre_hooks.clear()                       # drop the token-type hook
    model.save_pretrained(out_dir, safe_serialization=True)
    processor.save_pretrained(out_dir)
    print(f"exported standard Qwen2-VL checkpoint -> {out_dir}")


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/moe.yaml", help="the config the adapters were trained with")
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--routing", default=None, help="routing_dev.npz from evaluate.py --record_routing")
    ap.add_argument("--mode", required=True, choices=["stats", "full_merge", "selective", "profile", "dense"])
    ap.add_argument("--k", type=int, nargs="+", default=[0, 4, 8, 14, 28])
    ap.add_argument("--task", default=None, help="profile export: which task's profile")
    ap.add_argument("--eval", action="store_true", help="evaluate on --split with the 4-bit model")
    ap.add_argument("--split", default="data/v3/dev.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="first N samples per task")
    ap.add_argument("--export", default=None, help="output folder for a standard checkpoint")
    ap.add_argument("--out", default=None, help="results json (default: next to the adapters)")
    args = ap.parse_args()

    from config import load
    cfg = load(args.config)
    st = None
    if args.mode != "dense":
        if not args.routing:
            ap.error("--routing is required for this mode")
        st = routing_stats(args.routing)
        print_stats(st)
    if args.mode == "stats":
        return

    results = {}
    if args.eval:
        from evaluate import evaluate, load_for_eval, load_samples
        model, processor, token_ids = load_for_eval(cfg, cfg.data.max_pixels, adapters=args.adapters)
        samples = load_samples(args.split, args.limit)
        root = os.path.dirname(os.path.abspath(args.split))
        run = lambda ss: evaluate(model, processor, ss, root, token_ids, progress=False)
        if args.mode == "full_merge":
            apply_full_merge(model, st)
            results["full_merge"] = run(samples)
        elif args.mode == "selective":
            for k in args.k:
                kept = apply_selective(model, st, k)
                results[f"k={k}"] = r = run(samples)
                print(f"selective k={k:<3} routed layers {kept}: macro {r['macro']:.4f}", flush=True)
            set_routed(model)
        elif args.mode == "profile":
            per_task = {}
            for t in st["tasks"]:
                apply_profile(model, st, t)
                per_task[t] = run([s for s in samples if s["task"] == t])["per_task"][t]
            results["profile"] = {"per_task": per_task,
                                  "macro": float(np.mean([v["score"] for v in per_task.values()]))}
        elif args.mode == "dense":
            results["dense"] = run(samples)
        for name, r in results.items():
            print(f"{name}: macro {r['macro']:.4f} "
                  + " ".join(f"{t}={v['score']:.3f}" for t, v in r["per_task"].items()))
        out = args.out or os.path.join(args.adapters, f"convert_{args.mode}.json")
        with open(out, "w") as f:
            json.dump({"mode": args.mode, "split": args.split, "results": results,
                       "mi_bits": st["mi"].tolist() if st else None}, f, indent=2)
        print(f"results -> {out}")
        del model
        torch.cuda.empty_cache()

    if args.export:
        if args.mode == "selective":
            ap.error("selective merge keeps routed layers and cannot be exported as a dense checkpoint")
        if args.mode == "profile" and args.task is None:
            ap.error("--task is required to export a profile")
        cfg_fn = {"full_merge": lambda m: apply_full_merge(m, st),
                  "profile": lambda m: apply_profile(m, st, args.task),
                  "dense": lambda m: None}[args.mode]
        export(cfg, args.adapters, args.export, cfg_fn)


if __name__ == "__main__":
    main()
