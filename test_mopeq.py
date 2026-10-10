"""CPU tests for mopeq.py (MoPEQ-style mixed precision) on a tiny random Qwen2-VL.   python test_mopeq.py"""
import copy
import importlib.util
import itertools
import json
import os
import tempfile
import types

import torch
import torch.nn as nn

from config import load
from layers import moe_layers
from model import apply_moe_surgery
from mopeq import (allocate, apply_plan, budget_for_avg_bits, find_units, plan_bytes, quantize_unit,
                   sensitivity_hessian, sensitivity_kl, summarize_plan, unit_bytes, unit_linears, unit_params)

_spec = importlib.util.spec_from_file_location(
    "crm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_real_model.py"))
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)

G = 16                                                    # tiny dims are multiples of 16


def tiny():
    cfg = load("configs/moe_seq.yaml")
    cfg.moe.rank, cfg.moe.alpha = 4, 8
    model, _, tid, batches = crm.load_tiny(types.SimpleNamespace(), cfg)
    return cfg, model, tid, batches[0][1]


def logits(model, b):
    with torch.no_grad():
        return model(**b, use_cache=False).logits


def test_units_partition_all_base_linears():
    _, model, _, _ = tiny()
    units = find_units(model)
    in_units = {id(l) for m in units.values() for _, _, l in unit_linears(m)}
    all_lin = {id(m) for n, m in model.named_modules() if type(m) is nn.Linear and not n.endswith("lm_head")}
    assert in_units == all_lin, (len(in_units), len(all_lin))
    assert any(u.startswith("vis.") for u in units) and "lm.1.mlp" in units and "lm.0.attn" in units


def test_quantize_restore_and_bits_order():
    _, model, _, b = tiny()
    ref = logits(model, b)
    unit = find_units(model)["lm.0.mlp"]
    errs = {}
    for nb in (2, 4, 8):
        restore = quantize_unit(unit, nb, G)
        errs[nb] = float((logits(model, b) - ref).abs().max())
        restore()
        assert torch.equal(logits(model, b), ref)                # exact restore
    assert errs[2] > errs[4] > errs[8] > 0, errs


def test_sensitivity_kl_and_hessian():
    _, model, tid, b = tiny()
    units = find_units(model)
    s = sensitivity_kl(model, units, [b], tid, (2, 4, 8), G, log=lambda *_: None)
    assert set(s) == set(units)
    for u in units:
        assert s[u][2] >= s[u][8] >= 0, (u, s[u])
    with torch.no_grad():                         # give the tiny head a sharper output distribution
        model.lm_head.weight.mul_(20)
    sh, tr = sensitivity_hessian(model, units, [b], tid, (2, 4, 8), G, n_iter=2, units_per_pass=3,
                                 log=lambda *_: None)
    assert set(tr) == set(units) and all(torch.isfinite(torch.tensor(v)) for v in tr.values())
    for u in units:
        assert sh[u][2] >= sh[u][4] >= sh[u][8] >= 0, (u, sh[u])
    assert not any(p.requires_grad for p in model.parameters())


def test_hessian_fd_matches_exact():
    """Finite-difference HVPs (the T4-sized default) agree with exact double-backprop HVPs."""
    _, model, tid, b = tiny()
    with torch.no_grad():
        model.lm_head.weight.mul_(20)
    units = find_units(model)
    kw = dict(bits=(2, 4), group_size=G, n_iter=3, units_per_pass=4, seed=1, log=lambda *_: None)
    _, t_exact = sensitivity_hessian(model, units, [b], tid, hvp="exact", **kw)
    _, t_fd = sensitivity_hessian(model, units, [b], tid, hvp="fd", **kw)
    scale = max(abs(v) for v in t_exact.values())
    for u in units:
        assert abs(t_fd[u] - t_exact[u]) <= 0.05 * abs(t_exact[u]) + 1e-3 * scale, (u, t_fd[u], t_exact[u])


def test_allocate_is_optimal_and_within_budget():
    sizes = {"a": 4096, "b": 8192, "c": 2048}
    sens = {"a": {2: 9.0, 4: 1.0, 8: 0.1}, "b": {2: 0.5, 4: 0.2, 8: 0.0}, "c": {2: 3.0, 4: 0.5, 8: 0.0}}
    budget = budget_for_avg_bits(sizes, 4, G)
    plan = allocate(sens, sizes, budget, G, grid=64)
    assert plan_bytes(sizes, plan, G) <= budget
    best = min((dict(zip(sizes, bs)) for bs in itertools.product((2, 4, 8), repeat=3)
                if plan_bytes(sizes, dict(zip(sizes, bs)), G) <= budget - 64 * 3),   # grid rounding slack
               key=lambda p: sum(sens[u][p[u]] for u in sizes))
    assert sum(sens[u][plan[u]] for u in sizes) <= sum(sens[u][best[u]] for u in sizes) + 1e-9
    assert plan["a"] >= 4 and plan["b"] == 2                     # spend bits where they matter
    assert abs(summarize_plan({u: 4 for u in sizes}, sizes, G)["avg_bits"] - 4) < 1e-9


def test_apply_plan_then_moe_training_step():
    cfg, model, tid, b = tiny()
    units = find_units(model)
    sizes = {u: unit_params(m) for u, m in units.items()}
    plan = {"bits": {u: (8 if u.startswith("vis") else 4) for u in units}, "group_size": G}
    ref = logits(model, b)
    apply_plan(model, plan)
    assert not any(type(m) is nn.Linear for n, m in model.named_modules() if not n.endswith("lm_head"))
    q = logits(model, b)
    assert (q - ref).abs().max() > 0 and torch.isfinite(q).all()
    # the MoE-LoRA surgery works on an HQQ base and gradients reach LoRA + router
    apply_moe_surgery(model, cfg, tid["image_pad"], tid["im_start"])
    with torch.no_grad():
        for l in moe_layers(model):
            for m in (l.lora_gate, l.lora_up, l.lora_down):
                m.B.normal_(0, 0.1)
    model.train()
    out = model(**b, use_cache=False).logits
    out.float().pow(2).mean().backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert trainable and all(p.grad is not None for p in trainable)
    assert sum(sizes.values()) > 0 and plan_bytes(sizes, plan["bits"], G) < 2 * sum(sizes.values())


def test_apply_plan_from_json_with_device():
    """The evaluate.py --mopeq_plan path: plan json -> apply_plan(..., device=...) on a CPU-loaded model."""
    cfg, model, tid, b = tiny()
    units = find_units(model)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "plan.json")
        json.dump({"bits": {u: 4 for u in units}, "group_size": G}, open(p, "w"))
        apply_plan(model, json.load(open(p)), device="cpu")
    assert torch.isfinite(logits(model, b)).all()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print(f"All {len(tests)} mopeq tests passed.")
