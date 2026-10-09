"""
MoPEQ-style sensitivity-guided mixed-precision quantisation of the frozen Qwen2-VL base.

MoPEQ (Chitty-Venkata et al., ICCVW 2025) gives each expert of a MoE VLM its own bit-width from a
sensitivity score. Our experts are LoRA adapters (2.3% of the parameters), so the same idea is applied
to the units that hold the memory: every vision block, the vision merger, and the attention and MLP
of every language layer. Embeddings and lm_head stay fp16, as with bitsandbytes NF4.

Pipeline (all with HQQ, so 2/3/4/8-bit share one quantiser; uniform plans are the baselines):
    1. sensitivity  s[u][b] = expected damage of quantising unit u to b bits
         kl      : KL(fp16 || model with only u quantised to b bits) on answer tokens of calibration
                   samples (direct measurement)
         hessian : HAWQ/MoPEQ-style  Tr(H_u)/n_u * ||W_u - Q_b(W_u)||^2, Hutchinson trace of the
                   answer-token CE loss
    2. allocate     multiple-choice knapsack: minimise sum_u s[u][b_u] subject to the memory budget
                   (exact DP on a 64-KiB grid)
    3. apply        the plan before evaluation / training:  evaluate.py --mopeq_plan plan.json

    python mopeq.py sensitivity --metric kl --calib 32 --out runs/mopeq/sens_kl.json
    python mopeq.py allocate --sens runs/mopeq/sens_kl.json --avg_bits 3 --out runs/mopeq/plan_kl_3.json
    python mopeq.py uniform --bits 4 --out runs/mopeq/plan_u4.json
"""
import argparse
import contextlib
import json
import math
import os
import random
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

BITS = (2, 3, 4, 8)
GROUP_SIZE = 64
META_BYTES_PER_GROUP = 4                     # HQQ: fp16 scale + fp16 zero per group


# --------------------------------------------------------------------------- units
def _vision(model):
    m = model.model if hasattr(model.model, "visual") else model
    return m.visual


def find_units(model):
    """OrderedDict unit name -> module. Units partition all quantisable base Linear layers."""
    from layers import MoELayer, get_lm_layers
    units = OrderedDict()
    vis = _vision(model)
    for i, blk in enumerate(vis.blocks):
        units[f"vis.{i}"] = blk
    if getattr(vis, "merger", None) is not None:
        units["vis.merger"] = vis.merger
    for i, layer in enumerate(get_lm_layers(model)):
        units[f"lm.{i}.attn"] = layer.self_attn
        mlp = layer.mlp
        units[f"lm.{i}.mlp"] = mlp.base_mlp if isinstance(mlp, MoELayer) else mlp
    return units


def unit_linears(module):
    """[(parent, attr, nn.Linear)] inside a unit (LoRA / router modules excluded)."""
    out = []
    for name, parent in module.named_modules():
        if "lora" in name or "router" in name:
            continue
        for attr, child in parent.named_children():
            if type(child) is nn.Linear:
                out.append((parent, attr, child))
    return out


def unit_params(module):
    return sum(l.weight.numel() for _, _, l in unit_linears(module))


def unit_bytes(n, bits, group_size=GROUP_SIZE):
    if bits >= 16:
        return 2 * n
    return n * bits / 8 + math.ceil(n / group_size) * META_BYTES_PER_GROUP


# --------------------------------------------------------------------------- quantisation
def _hqq(lin, bits, group_size, device=None, keep_orig=True):
    """keep_orig=False drops the fp16 weight (real memory saving); True keeps it for restore()."""
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    return HQQLinear(lin, BaseQuantizeConfig(nbits=bits, group_size=group_size),
                     compute_dtype=lin.weight.dtype, device=str(device or lin.weight.device),
                     del_orig=not keep_orig)


def quantize_unit(module, bits, group_size=GROUP_SIZE, device=None, keep_orig=True):
    """Replace every Linear of the unit by an HQQ layer (on `device`, default: where the Linear is).
    Returns a restore() callable (only meaningful with keep_orig=True)."""
    if bits >= 16:
        if device is not None:
            for parent, attr, lin in unit_linears(module):
                setattr(parent, attr, lin.to(device))
        return lambda: None
    swapped = []
    for parent, attr, lin in unit_linears(module):
        q = _hqq(lin, bits, group_size, device, keep_orig)
        q.requires_grad_(False)
        setattr(parent, attr, q)
        swapped.append((parent, attr, lin if keep_orig else None))

    def restore():
        for parent, attr, lin in swapped:
            setattr(parent, attr, lin)
    return restore


@contextlib.contextmanager
def quantized(module, bits, group_size=GROUP_SIZE):
    restore = quantize_unit(module, bits, group_size)
    try:
        yield
    finally:
        restore()


def apply_plan(model, plan, group_size=None, device=None):
    """plan: {"bits": {unit: bits}, "group_size": g}. Quantises in place (irreversible), dropping the
    fp16 weights. With `device`, a model loaded on CPU is quantised unit by unit straight onto the
    device and the remaining (unquantised) modules are moved after it, so the fp16 model never has
    to fit in GPU memory (4 GB GTX 1650)."""
    g = group_size or plan.get("group_size", GROUP_SIZE)
    units = find_units(model)
    missing = set(units) - set(plan["bits"])
    if missing:
        raise ValueError(f"plan has no bits for units {sorted(missing)[:5]}")
    for name, mod in units.items():
        quantize_unit(mod, int(plan["bits"][name]), g, device=device, keep_orig=False)
    if device is not None:
        move_unquantized(model, device)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return model


def move_unquantized(model, device):
    """Move every parameter/buffer that is not inside an HQQ layer (those are already placed)."""
    from hqq.core.quantize import HQQLinear
    moved = {}                                   # id -> moved Parameter: keeps tied weights tied

    def walk(m):
        if isinstance(m, HQQLinear):
            return
        for k, p in m._parameters.items():
            if p is not None and p.device != torch.device(device):
                if id(p) not in moved:
                    moved[id(p)] = nn.Parameter(p.data.to(device), requires_grad=p.requires_grad)
                m._parameters[k] = moved[id(p)]
        for k, b in m._buffers.items():
            if b is not None:
                m._buffers[k] = b.to(device)
        for c in m.children():
            walk(c)
    walk(model)


def plan_bytes(model_or_sizes, bits, group_size=GROUP_SIZE):
    sizes = model_or_sizes if isinstance(model_or_sizes, dict) else \
        {u: unit_params(m) for u, m in find_units(model_or_sizes).items()}
    return sum(unit_bytes(sizes[u], int(bits[u]), group_size) for u in sizes)


def hqq_error(lin, bits, group_size=GROUP_SIZE):
    """||W - Q_b(W)||^2 of one Linear (dequantised HQQ weight)."""
    if bits >= 16:
        return 0.0
    q = _hqq(lin, bits, group_size)
    err = float((q.dequantize().float() - lin.weight.float()).pow(2).sum())
    del q
    return err


# --------------------------------------------------------------------------- sensitivity
def _answer_positions(batch, token_ids):
    from kd import answer_shift_mask
    return answer_shift_mask(batch["input_ids"][0].cpu(), token_ids)


def _forward_logits(model, batch, mask):
    out = model(**batch, use_cache=False).logits[0, :-1]
    return out[mask.to(out.device)].float()


@torch.no_grad()
def sensitivity_kl(model, units, calib, token_ids, bits=BITS, group_size=GROUP_SIZE, log=print):
    """s[u][b] = mean KL(p_fp || p_q) over answer tokens, only unit u quantised to b bits."""
    model.eval()
    masks = [_answer_positions(b, token_ids) for b in calib]
    ref = [F.log_softmax(_forward_logits(model, b, m), -1).cpu() for b, m in zip(calib, masks)]
    sens = {}
    for k, (name, mod) in enumerate(units.items()):
        sens[name] = {}
        for nb in bits:
            if nb >= 16:
                sens[name][nb] = 0.0
                continue
            with quantized(mod, nb, group_size):
                kl, n = 0.0, 0
                for b, m, r in zip(calib, masks, ref):
                    lq = F.log_softmax(_forward_logits(model, b, m), -1).cpu()
                    kl += float((r.exp() * (r - lq)).sum(-1).clamp_min(0).sum())   # fp rounding can dip < 0
                    n += r.shape[0]
            sens[name][nb] = kl / max(n, 1)
        log(f"[{k + 1}/{len(units)}] {name}: " + " ".join(f"{b}b={v:.2e}" for b, v in sens[name].items()))
    return sens


def sensitivity_hessian(model, units, calib, token_ids, bits=BITS, group_size=GROUP_SIZE, n_iter=4,
                        units_per_pass=8, log=print):
    """HAWQ / MoPEQ style: s[u][b] = Tr(H_u)/n_u * ||W_u - Q_b(W_u)||^2, Hutchinson estimate of the
    trace of the Hessian of the answer-token CE loss w.r.t. the unit's Linear weights."""
    from utils import get_answer_labels
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    names = list(units)
    trace = {}
    for s in range(0, len(names), units_per_pass):
        group = names[s:s + units_per_pass]
        params = {u: [l.weight for _, _, l in unit_linears(units[u])] for u in group}
        flat = [p for u in group for p in params[u]]
        for p in flat:
            p.requires_grad_(True)
        acc, probes = {u: 0.0 for u in group}, 0
        for b in calib:
            labels = get_answer_labels(b["input_ids"], token_ids["im_start"], token_ids["assistant"],
                                       token_ids["im_end"])[:, 1:].to(b["input_ids"].device)
            for _ in range(n_iter):
                # fused SDPA kernels have no double backward: use the math kernel for the HVP
                with sdpa_kernel(SDPBackend.MATH):
                    logits = model(**b, use_cache=False).logits[:, :-1].float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1),
                                           ignore_index=-100)
                    g = torch.autograd.grad(loss, flat, create_graph=True)
                    v = [torch.randint_like(p, 2) * 2 - 1 for p in flat]
                    hv = torch.autograd.grad(g, flat, grad_outputs=v)
                vhv = [float((h.float() * w.float()).sum()) for h, w in zip(hv, v)]
                if all(math.isfinite(x) for x in vhv):            # fp16 HVPs can overflow: drop the probe
                    i = 0
                    for u in group:
                        for _ in params[u]:
                            acc[u] += vhv[i]
                            i += 1
                    probes += 1
                del g, hv, v, logits, loss
        for p in flat:
            p.requires_grad_(False)
        if probes == 0:
            raise FloatingPointError(f"all Hutchinson probes overflowed for {group}; use fp32 or smaller inputs")
        for u in group:
            trace[u] = acc[u] / probes
        log(f"hessian traces {group[0]}..{group[-1]}: " + " ".join(f"{u}={trace[u]:.3e}" for u in group))
    sens = {}
    for u in names:
        n = unit_params(units[u])
        lins = [l for _, _, l in unit_linears(units[u])]
        sens[u] = {nb: (max(trace[u], 0.0) / n) * sum(hqq_error(l, nb, group_size) for l in lins)
                   for nb in bits}
    return sens, trace


# --------------------------------------------------------------------------- allocation
def allocate(sens, sizes, budget_bytes, group_size=GROUP_SIZE, grid=2 ** 16):
    """Exact multiple-choice knapsack: min sum s[u][b] s.t. sum bytes <= budget (costs rounded UP to
    `grid` bytes, so the plan never exceeds the budget). Returns {unit: bits}."""
    units = list(sizes)
    opts = {u: sorted((int(b), float(s)) for b, s in sens[u].items()) for u in units}
    cost = {u: {b: math.ceil(unit_bytes(sizes[u], b, group_size) / grid) for b, _ in opts[u]} for u in units}
    B = int(budget_bytes // grid)
    if sum(min(cost[u].values()) for u in units) > B:
        raise ValueError("budget below the all-lowest-bits plan")
    INF = float("inf")
    best = [0.0] + [INF] * B                 # best[c] = min loss using exactly <= c grid cells
    choice = []
    for u in units:
        new, pick = [INF] * (B + 1), [None] * (B + 1)
        for c in range(B + 1):
            if best[c] == INF:
                continue
            for b, s in opts[u]:
                c2 = c + cost[u][b]
                if c2 <= B and best[c] + s < new[c2]:
                    new[c2], pick[c2] = best[c] + s, (b, c)
        best = new
        choice.append(pick)
    c = min(range(B + 1), key=lambda i: best[i])
    plan = {}
    for u, pick in zip(reversed(units), reversed(choice)):
        b, c = pick[c]
        plan[u] = b
    return {u: plan[u] for u in units}


def budget_for_avg_bits(sizes, avg_bits, group_size=GROUP_SIZE):
    return sum(sizes[u] * avg_bits / 8 + math.ceil(sizes[u] / group_size) * META_BYTES_PER_GROUP for u in sizes)


def summarize_plan(bits, sizes, group_size=GROUP_SIZE):
    n = sum(sizes.values())
    mib = plan_bytes(sizes, bits, group_size) / 2 ** 20
    avg = sum(sizes[u] * bits[u] for u in sizes) / n
    hist = {b: sum(1 for u in bits if bits[u] == b) for b in sorted(set(bits.values()))}
    by_part = {}
    for part in ("vis", "lm"):
        us = [u for u in sizes if u.startswith(part)]
        by_part[part] = sum(sizes[u] * bits[u] for u in us) / max(sum(sizes[u] for u in us), 1)
    return {"quantised_params": n, "mib": mib, "avg_bits": avg, "units_per_bits": hist,
            "avg_bits_vision": by_part["vis"], "avg_bits_language": by_part["lm"]}


# --------------------------------------------------------------------------- CLI
def calib_batches(cfg, processor, n, seed=0, device="cuda"):
    """n training samples, balanced over tasks, as batch-1 processor outputs (prompt + answer)."""
    from data import VQADataset, collate_fn
    ds = VQADataset(os.path.join(cfg.data.dir, "train.jsonl"), processor)
    by_task = {}
    for i, d in enumerate(ds.data):
        by_task.setdefault(d["task"], []).append(i)
    rng = random.Random(seed)
    idx = []
    for t in sorted(by_task):
        idx += rng.sample(by_task[t], min(len(by_task[t]), math.ceil(n / len(by_task))))
    out = []
    for i in idx[:n]:
        b = collate_fn([ds[i]], processor)
        out.append({k: v.to(device) for k, v in b.items() if torch.is_tensor(v)})
    return out


def load_fp16(cfg, device_index=0):
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from model import resolve_token_ids
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.data.min_pixels,
                                              max_pixels=cfg.data.max_pixels)
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, torch_dtype=torch.float16,
                                                            device_map={"": device_index}).eval()
    return model, processor, resolve_token_ids(processor)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sensitivity")
    s.add_argument("--config", default="configs/base.yaml")
    s.add_argument("--metric", choices=["kl", "hessian"], default="kl")
    s.add_argument("--calib", type=int, default=32)
    s.add_argument("--bits", type=int, nargs="+", default=list(BITS))
    s.add_argument("--group_size", type=int, default=GROUP_SIZE)
    s.add_argument("--n_iter", type=int, default=4, help="Hutchinson probes per sample (hessian)")
    s.add_argument("--units", default=None, help="regex: only these units")
    s.add_argument("--shard", type=int, default=0, help="this process handles units i %% num_shards == shard")
    s.add_argument("--num_shards", type=int, default=1)
    s.add_argument("--max_pixels", type=int, default=None, help="calibration resolution (default: config)")
    s.add_argument("--out", required=True)
    a = sub.add_parser("allocate")
    a.add_argument("--sens", nargs="+", required=True, help="one or more sensitivity json (merged)")
    a.add_argument("--avg_bits", type=float, required=True, help="memory budget = uniform plan at this bit-width")
    a.add_argument("--min_bits", type=int, default=2)
    a.add_argument("--out", required=True)
    u = sub.add_parser("uniform")
    u.add_argument("--sizes", nargs="+", default=None,
                   help="sensitivity json(s) (for the unit list, shards merged); default: build from model")
    u.add_argument("--config", default="configs/base.yaml")
    u.add_argument("--bits", type=int, required=True)
    u.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.cmd == "sensitivity":
        import re
        from config import load
        cfg = load(args.config)
        if args.max_pixels:
            cfg.data.max_pixels = args.max_pixels
        model, processor, token_ids = load_fp16(cfg)
        units = find_units(model)
        if args.units:
            units = OrderedDict((k, v) for k, v in units.items() if re.search(args.units, k))
        units = OrderedDict((k, v) for i, (k, v) in enumerate(units.items()) if i % args.num_shards == args.shard)
        calib = calib_batches(cfg, processor, args.calib)
        sizes = {k: unit_params(v) for k, v in units.items()}
        print(f"{len(units)} units, {sum(sizes.values()) / 1e6:.0f}M quantisable params, {len(calib)} calib samples")
        extra = {}
        if args.metric == "kl":
            sens = sensitivity_kl(model, units, calib, token_ids, args.bits, args.group_size)
        else:
            sens, extra["trace"] = sensitivity_hessian(model, units, calib, token_ids, args.bits,
                                                       args.group_size, args.n_iter)
        json.dump({"metric": args.metric, "group_size": args.group_size, "sizes": sizes, "sens": sens,
                   "calib": args.calib, **extra}, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")

    elif args.cmd == "allocate":
        sizes, sens, g, metric = {}, {}, None, set()
        for f in args.sens:
            d = json.load(open(f))
            sizes.update(d["sizes"])
            sens.update({k: {int(b): s for b, s in v.items() if int(b) >= args.min_bits} for k, v in d["sens"].items()})
            g = d["group_size"]
            metric.add(d["metric"])
        budget = budget_for_avg_bits(sizes, args.avg_bits, g)
        bits = allocate(sens, sizes, budget, g)
        summ = summarize_plan(bits, sizes, g)
        json.dump({"bits": bits, "group_size": g, "metric": sorted(metric), "avg_bits_budget": args.avg_bits,
                   "budget_mib": budget / 2 ** 20, "summary": summ}, open(args.out, "w"), indent=1)
        print(json.dumps(summ, indent=1), f"\n-> {args.out}")

    else:
        if args.sizes:
            sizes = {}
            for f in args.sizes:
                d = json.load(open(f))
                sizes.update(d["sizes"])
                g = d["group_size"]
        else:
            from accelerate import init_empty_weights
            from transformers import AutoConfig, Qwen2VLForConditionalGeneration
            from config import load
            cfg = load(args.config)
            with init_empty_weights():
                m = Qwen2VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(cfg.model_id))
            sizes, g = {k: unit_params(v) for k, v in find_units(m).items()}, GROUP_SIZE
        bits = {k: args.bits for k in sizes}
        json.dump({"bits": bits, "group_size": g, "metric": ["uniform"], "summary": summarize_plan(bits, sizes, g)},
                  open(args.out, "w"), indent=1)
        print(f"uniform {args.bits}-bit plan, {plan_bytes(sizes, bits, g) / 2 ** 20:.0f} MiB -> {args.out}")


if __name__ == "__main__":
    main()
