"""
T11 — edge cost benchmark: time-to-first-token, decode speed and memory, one model variant per run.
No routing recording, no metric computation: only the cost of answering.

    # GPU (4-bit NF4, the deployment format)
    python scripts/bench_edge.py --base --label base
    python scripts/bench_edge.py --config configs/moe.yaml --adapters runs/moe_s0/best --label moe_routed
    python scripts/bench_edge.py --config configs/moe.yaml --adapters runs/moe_s0/best \
        --fixed full_merge --routing runs/moe_s0/routing_dev.npz --label moe_fixed_unfolded
    python scripts/bench_edge.py --checkpoint exports/moe_s0_merged --label moe_merged
    # CPU (unquantized; laptop-class proxy)
    python scripts/bench_edge.py --base --device cpu --threads 4 --limit 3 --label base_cpu
    # table of everything measured so far
    python scripts/bench_edge.py --table

Per sample (batch 1, greedy): TTFT = generate(max_new_tokens=1); then exactly N new tokens
(min_new_tokens = max_new_tokens = N) -> decode tok/s = (N - 1) / (t_N - TTFT).
Memory: weights = model.get_memory_footprint(); peak = torch.cuda.max_memory_allocated (GPU) or the
process's max RSS (CPU, Linux). Rows are appended to --out (default runs/bench/bench.jsonl).
"""
import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def bench(model, processor, samples, root, token_ids, encode_fn, new_tokens=16, warmup=2):
    """Returns {"ttft_mean", "ttft_p50", "decode_tps_mean", "total_mean", "prompt_tokens_mean", "n", ...}."""
    model.eval()
    device = next(model.parameters()).device
    kw = dict(do_sample=False, use_cache=True, pad_token_id=token_ids["im_end"])

    def run(inputs, n):
        _sync(device)
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=n, min_new_tokens=n, **kw)
        _sync(device)
        return time.perf_counter() - t0, out.shape[1] - inputs["input_ids"].shape[1]

    enc = [{k: v.to(device) for k, v in encode_fn(processor, s, root).items()} for s in samples]
    for inputs in enc[:warmup]:
        run(inputs, 1)
        run(inputs, new_tokens)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    ttft, total, tps, plen = [], [], [], []
    for inputs in enc:
        t1, _ = run(inputs, 1)
        tn, got = run(inputs, new_tokens)
        if got != new_tokens:
            raise RuntimeError(f"generated {got} tokens, expected {new_tokens}")
        ttft.append(t1)
        total.append(tn)
        tps.append((new_tokens - 1) / max(tn - t1, 1e-9))
        plen.append(int(inputs["input_ids"].shape[1]))
    res = {"n": len(enc), "new_tokens": new_tokens,
           "ttft_mean": float(np.mean(ttft)), "ttft_p50": float(np.median(ttft)),
           "total_mean": float(np.mean(total)), "decode_tps_mean": float(np.mean(tps)),
           "prompt_tokens_mean": float(np.mean(plen)),
           "weights_mib": model.get_memory_footprint() / 2 ** 20 if hasattr(model, "get_memory_footprint") else None}
    if device.type == "cuda":
        res["peak_mib"] = torch.cuda.max_memory_allocated(device) / 2 ** 20
    else:
        try:
            import resource
            res["peak_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024   # Linux: KiB
        except ImportError:                                                            # Windows
            import psutil
            res["peak_mib"] = psutil.Process().memory_info().peak_wset / 2 ** 20
    return res


# --------------------------------------------------------------------------- loading
def load_variant(args, cfg):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration
    from model import apply_moe_surgery, load_adapters, resolve_token_ids
    src = args.checkpoint or cfg.model_id
    max_pixels = args.max_pixels or cfg.data.max_pixels
    processor = AutoProcessor.from_pretrained(src, min_pixels=min(cfg.data.min_pixels, max_pixels),
                                              max_pixels=max_pixels)
    if args.device == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            src, quantization_config=bnb, torch_dtype=torch.float16, device_map={"": 0})
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(src, torch_dtype=getattr(torch, args.dtype))
    token_ids = resolve_token_ids(processor)
    if args.adapters:
        apply_moe_surgery(model, cfg, token_ids["image_pad"], token_ids["im_start"])
        load_adapters(model, args.adapters)
        if args.fixed:
            from convert import apply_full_merge, apply_profile, routing_stats
            st = routing_stats(args.routing)
            if args.fixed == "full_merge":
                apply_full_merge(model, st)
            else:
                apply_profile(model, st, args.task)
    return model.eval(), processor, token_ids


def print_table(path):
    rows = [json.loads(l) for l in open(path)]
    print(f"{'label':<26}{'dev':>5}{'TTFT s':>9}{'tok/s':>8}{'total s':>9}{'weights MiB':>13}{'peak MiB':>10}")
    for r in rows:
        print(f"{r['label']:<26}{r['device']:>5}{r['ttft_mean']:>9.3f}{r['decode_tps_mean']:>8.1f}"
              f"{r['total_mean']:>9.3f}{(r['weights_mib'] or 0):>13.0f}{r['peak_mib']:>10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml", help="the config the adapters were trained with")
    ap.add_argument("--base", action="store_true", help="untouched base model")
    ap.add_argument("--adapters", default=None)
    ap.add_argument("--checkpoint", default=None, help="exported standard checkpoint (convert.py --export)")
    ap.add_argument("--fixed", default=None, choices=["full_merge", "profile"],
                    help="adapters in fixed-gate mode (unfolded), needs --routing")
    ap.add_argument("--routing", default=None)
    ap.add_argument("--task", default="document_ocr", help="--fixed profile: whose profile")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--dtype", default="float32", help="CPU only: float32 or bfloat16")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (torch.set_num_threads)")
    ap.add_argument("--split", default="data/v3/dev.jsonl")
    ap.add_argument("--limit", type=int, default=10, help="samples PER TASK")
    ap.add_argument("--new_tokens", type=int, default=16)
    ap.add_argument("--max_pixels", type=int, default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default="runs/bench/bench.jsonl")
    ap.add_argument("--table", action="store_true", help="print the table in --out and exit")
    args = ap.parse_args()

    if args.table:
        return print_table(args.out)
    if sum(map(bool, (args.base, args.adapters, args.checkpoint))) != 1:
        ap.error("give exactly one of --base, --adapters, --checkpoint")
    if args.fixed and not (args.adapters and args.routing):
        ap.error("--fixed needs --adapters and --routing")
    if args.threads:
        torch.set_num_threads(args.threads)

    from config import load
    from evaluate import encode_prompt, load_samples
    cfg = load(args.config)
    model, processor, token_ids = load_variant(args, cfg)
    samples = load_samples(args.split, args.limit)
    label = args.label or (args.checkpoint or args.adapters or "base")
    print(f"{label}: {len(samples)} samples on {args.device}, {args.new_tokens} new tokens each")
    res = bench(model, processor, samples, os.path.dirname(os.path.abspath(args.split)), token_ids,
                encode_prompt, args.new_tokens)
    res.update({"label": label, "device": args.device,
                "hw": torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor() or platform.machine(),
                "threads": torch.get_num_threads(), "dtype": "nf4" if args.device == "cuda" else args.dtype,
                "config": args.config, "adapters": args.adapters, "checkpoint": args.checkpoint,
                "fixed": args.fixed, "split": args.split})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(res) + "\n")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=1))
    print_table(args.out)


if __name__ == "__main__":
    main()
