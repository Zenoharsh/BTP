"""CPU test for scripts/bench_edge.py on a tiny random Qwen2-VL.   python test_bench.py"""
import importlib.util
import os
import types

import torch

from config import load
from layers import moe_layers
from model import apply_moe_surgery
from scripts.bench_edge import bench

_spec = importlib.util.spec_from_file_location(
    "crm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_real_model.py"))
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)


def run(config, fixed=False):
    cfg = load(config)
    cfg.moe.rank, cfg.moe.alpha = 4, 8
    model, proc, tid, batches = crm.load_tiny(types.SimpleNamespace(), cfg)
    if config != "configs/base.yaml":
        apply_moe_surgery(model, cfg, tid["image_pad"], tid["im_start"])
        if fixed:
            for l in moe_layers(model):
                l.mode, l.fixed_g = "fixed", torch.full((l.num_experts,), 1.0)
    prompt = batches[0][2]
    enc = lambda p, s, r: {k: v.clone() for k, v in prompt.items()}
    return bench(model, proc, [{}] * 3, ".", tid, enc, new_tokens=5, warmup=1)


def test_bench_all_variants():
    for config, fixed in [("configs/base.yaml", False), ("configs/moe.yaml", False),
                          ("configs/moe_seq.yaml", False), ("configs/moe.yaml", True),
                          ("configs/dense16.yaml", False)]:
        r = run(config, fixed)
        assert r["n"] == 3 and r["new_tokens"] == 5 and r["prompt_tokens_mean"] == 16
        assert 0 < r["ttft_mean"] <= r["total_mean"] and r["decode_tps_mean"] > 0
        assert r["peak_mib"] > 0 and r["weights_mib"] > 0
        print(config, fixed, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})


if __name__ == "__main__":
    test_bench_all_variants()
    print("[PASS] test_bench_all_variants\nAll 1 bench tests passed.")
