"""
Edge benchmark: latency, decode speed and peak memory of the deployable variants.

    # GPU (T4 / GTX 1650): NF4 4-bit weights, fp16 compute (same as training/eval)
    python bench.py --variants base routed merged profile --adapters runs/moe_s0/best
    # GTX 16xx (no tensor cores; fp16 GEMM ~5x slower than fp32 there): 4-bit weights, fp32 compute
    python bench.py --compute_dtype fp32 --variants base routed merged --adapters runs/moe_s0/best
    # CPU only: bf16 weights
    python bench.py --device cpu --variants base merged --adapters runs/moe_s0/best --limit 5

Variants
    base           untouched Qwen2-VL-2B
    routed         base + MoE-LoRA adapters, router active (what training produces)
    merged         full merge (convert.py), folded into the base -> standard Qwen2-VL checkpoint
    profile        one folded checkpoint per task (benchmarked on that task's samples only)
    dense          a dense LoRA run (num_experts=1) folded into the base
Folded checkpoints are exported once (fp16, CPU) to <run>/export_<name>/ and reused.

Each variant runs in its own process, so memory numbers are not polluted by the previous one.
Per sample: time to first token (prefill, generate with max_new_tokens=1), end-to-end latency with
the evaluation protocol (greedy, stop at newline / <|im_end|>), decode tokens/s, and the score.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch

from evaluate import encode_prompt, load_samples, stop_token_ids
from layers import set_telemetry
from metrics import compute_metric

TASKS = ("chart_qa", "document_ocr", "spatial_reasoning")


# --------------------------------------------------------------------------- config / export
def run_config(args):
    """The config the adapters were trained with (from <adapters>/meta.json), else --config."""
    from config import load
    cfg = load(args.config)
    meta = os.path.join(args.adapters, "meta.json") if args.adapters else None
    if meta and os.path.exists(meta):
        saved = json.load(open(meta)).get("config", {})
        for sec in ("data", "moe"):
            for k, v in saved.get(sec, {}).items():
                setattr(getattr(cfg, sec), k, v)
    return cfg


def default_routing(adapters):
    return os.path.join(os.path.dirname(os.path.abspath(adapters)), "routing_dev.npz")


def ensure_export(args, cfg, name, task=None):
    from convert import apply_full_merge, apply_profile, export, routing_stats
    out = os.path.join(os.path.dirname(os.path.abspath(args.adapters)), f"export_{name}")
    if os.path.exists(os.path.join(out, "config.json")):
        return out
    if name == "dense":
        fn = lambda m: None
    else:
        st = routing_stats(args.routing or default_routing(args.adapters))
        fn = (lambda m: apply_full_merge(m, st)) if task is None else (lambda m: apply_profile(m, st, task))
    print(f"exporting {name} -> {out} (one-off, fp16 on CPU)", flush=True)
    export(cfg, args.adapters, out, fn)
    return out


# --------------------------------------------------------------------------- loading
def load_variant(args, cfg, variant):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    from model import apply_moe_surgery, load_adapters, resolve_token_ids
    name, _, task = variant.partition(":")
    if name in ("merged", "profile", "dense"):
        path = ensure_export(args, cfg, variant.replace(":", "_"), task or None)
    else:
        path = cfg.model_id
    processor = AutoProcessor.from_pretrained(cfg.model_id, min_pixels=cfg.data.min_pixels,
                                              max_pixels=cfg.data.max_pixels)
    if args.device == "cuda":
        cd = {"fp16": torch.float16, "fp32": torch.float32}[args.compute_dtype]
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=cd)
        kw = dict(quantization_config=bnb, torch_dtype=cd, device_map={"": 0})
    else:
        kw = dict(torch_dtype={"bf16": torch.bfloat16, "fp32": torch.float32}[args.cpu_dtype])
    model = Qwen2VLForConditionalGeneration.from_pretrained(path, **kw).eval()
    token_ids = resolve_token_ids(processor)
    if name == "routed":
        apply_moe_surgery(model, cfg, token_ids["image_pad"], token_ids["im_start"])
        load_adapters(model, args.adapters)
        model.eval()
        set_telemetry(model, False)             # no routing telemetry/GPU syncs while timing
    return model, processor, token_ids


# --------------------------------------------------------------------------- measurement
def mem_now(device):
    if device == "cuda":
        return torch.cuda.memory_allocated() / 2 ** 20
    import psutil
    return psutil.Process().memory_info().rss / 2 ** 20


def mem_peak(device):
    if device == "cuda":
        return torch.cuda.max_memory_allocated() / 2 ** 20
    import psutil
    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 2 ** 20           # Windows: true peak; else current RSS


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_one(args, variant):
    torch.manual_seed(0)
    cfg = run_config(args)
    t_load = time.perf_counter()
    model, processor, token_ids = load_variant(args, cfg, variant)
    load_s = time.perf_counter() - t_load
    dev = args.device
    loaded_mib = mem_now(dev)              # CPU: RSS, which undercounts memory-mapped weights
    weights_mib = sum(t.numel() * t.element_size()
                      for t in list(model.parameters()) + list(model.buffers())) / 2 ** 20

    samples = load_samples(args.split, args.limit)
    task = variant.partition(":")[2]
    if task:
        samples = [s for s in samples if s["task"] == task]
    root = os.path.dirname(os.path.abspath(args.split))
    eos = stop_token_ids(processor, token_ids)
    gen_kw = dict(do_sample=False, pad_token_id=token_ids["im_end"], use_cache=True)
    device = next(model.parameters()).device

    def enc(s):
        return {k: v.to(device) for k, v in encode_prompt(processor, s, root).items()}

    for s in samples[:args.warmup]:
        model.generate(**enc(s), max_new_tokens=4, eos_token_id=eos, **gen_kw)
    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()

    rows = []
    for s in samples:
        inputs = enc(s)
        sync(dev); t0 = time.perf_counter()
        model.generate(**inputs, max_new_tokens=1, **gen_kw)
        sync(dev); ttft = time.perf_counter() - t0
        sync(dev); t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, eos_token_id=eos, **gen_kw)
        sync(dev); e2e = time.perf_counter() - t0
        new = out[0, inputs["input_ids"].shape[1]:]
        pred = processor.tokenizer.decode(new, skip_special_tokens=True).split("\n")[0].strip()
        n_new = int(new.numel())
        rows.append({"uid": s["uid"], "task": s["task"], "ttft": ttft, "latency": e2e, "new_tokens": n_new,
                     "decode_tps": (n_new - 1) / (e2e - ttft) if n_new > 1 and e2e > ttft else None,
                     "prompt_tokens": int(inputs["input_ids"].shape[1]),
                     "score": compute_metric(s["task"], pred, s["answers"])})

    tps = [r["decode_tps"] for r in rows if r["decode_tps"] is not None]
    by_task = {t: float(np.mean([r["score"] for r in rows if r["task"] == t]))
               for t in TASKS if any(r["task"] == t for r in rows)}
    res = {"variant": variant, "device": dev,
           "device_name": torch.cuda.get_device_name(0) if dev == "cuda" else cpu_name(),
           "precision": f"nf4+{args.compute_dtype}" if dev == "cuda" else args.cpu_dtype,
           "n": len(rows), "load_s": load_s, "weights_mib": weights_mib, "loaded_mib": loaded_mib, "peak_mib": mem_peak(dev),
           "ttft_ms": 1000 * float(np.mean([r["ttft"] for r in rows])),
           "latency_ms": 1000 * float(np.mean([r["latency"] for r in rows])),
           "latency_p90_ms": 1000 * float(np.percentile([r["latency"] for r in rows], 90)),
           "decode_tps": float(np.mean(tps)) if tps else None,
           "score_by_task": by_task, "score_macro": float(np.mean(list(by_task.values()))),
           "rows": rows}
    return res


def cpu_name():
    try:
        import platform
        return platform.processor() or platform.machine()
    except Exception:
        return "cpu"


# --------------------------------------------------------------------------- CLI
def expand(variants):
    out = []
    for v in variants:
        out += [f"profile:{t}" for t in TASKS] if v == "profile" else [v]
    return out


def print_table(results):
    print(f"\n{'variant':<28}{'n':>4}{'weights':>9}{'peak':>8}{'TTFT':>8}{'latency':>9}{'p90':>8}"
          f"{'tok/s':>7}{'score':>7}")
    print(f"{'':<28}{'':>4}{'MiB':>9}{'MiB':>8}{'ms':>8}{'ms':>9}{'ms':>8}")
    for r in results:
        tps = f"{r['decode_tps']:.1f}" if r["decode_tps"] else "-"
        print(f"{r['variant']:<28}{r['n']:>4}{r['weights_mib']:>9.0f}{r['peak_mib']:>8.0f}{r['ttft_ms']:>8.0f}"
              f"{r['latency_ms']:>9.0f}{r['latency_p90_ms']:>8.0f}{tps:>7}{r['score_macro']:>7.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["base", "routed", "merged"],
                    help="base routed merged profile profile:<task> dense")
    ap.add_argument("--adapters", default=None, help="runs/<run>/best (needed for all but base)")
    ap.add_argument("--routing", default=None, help="default: <run>/routing_dev.npz")
    ap.add_argument("--config", default="configs/base.yaml", help="used only if <adapters>/meta.json is missing")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute_dtype", default="fp16", choices=["fp16", "fp32"],
                    help="GPU compute dtype. fp32 on GTX 16xx: their fp16 GEMM is ~5x slower than fp32")
    ap.add_argument("--cpu_dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--split", default="data/v3/dev.jsonl")
    ap.add_argument("--limit", type=int, default=10, help="first N samples per task")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: torch default)")
    ap.add_argument("--out_dir", default="runs/bench")
    ap.add_argument("--tag", default=None, help="name of the result file (default: device name)")
    ap.add_argument("--_child", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA not available in this Python (install a CUDA build of torch, or use --device cpu)")

    if args._child:                                       # one variant, write json, exit
        res = bench_one(args, args._child)
        with open(os.environ["BENCH_OUT"], "w") as f:
            json.dump(res, f, indent=2)
        return

    variants = expand(args.variants)
    if any(v != "base" for v in variants) and not args.adapters:
        ap.error("--adapters is required for every variant except base")
    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or (f"{torch.cuda.get_device_name(0)}_{args.compute_dtype}" if args.device == "cuda"
                       else f"cpu_{args.cpu_dtype}")
    tag = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag)
    results = []
    for v in variants:
        out = os.path.join(args.out_dir, f"{tag}__{v.replace(':', '_')}.json")
        cmd = [sys.executable, os.path.abspath(__file__), "--_child", v] + \
              [a for a in sys.argv[1:] if a != "--_child"]
        print(f"=== {v}", flush=True)
        subprocess.run(cmd, check=True, env={**os.environ, "BENCH_OUT": out})
        results.append(json.load(open(out)))
    print_table(results)
    with open(os.path.join(args.out_dir, f"{tag}.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "rows"} for r in results], f, indent=2)
    print(f"-> {os.path.join(args.out_dir, tag + '.json')}")


if __name__ == "__main__":
    main()
