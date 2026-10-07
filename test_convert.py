"""CPU tests for convert.py on a tiny random Qwen2-VL.   python test_convert.py"""
import copy
import importlib.util
import os
import tempfile
import types

import numpy as np
import torch

from config import load
from convert import (apply_full_merge, apply_profile, apply_selective, merge_into_base,
                     mutual_information_bits, profile_gates, routing_stats, set_routed)
from evaluate import evaluate
from layers import MoELayer, get_lm_layers, moe_layers
from model import apply_moe_surgery

_spec = importlib.util.spec_from_file_location(
    "crm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_real_model.py"))
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)

TASKS = ["chart_qa", "document_ocr", "spatial_reasoning"]


def tiny(config="configs/moe.yaml", route_tokens="all", trained=True):
    cfg = load(config)
    cfg.moe.rank, cfg.moe.alpha, cfg.moe.route_tokens = 4, 8, route_tokens
    model, processor, tid, batches = crm.load_tiny(types.SimpleNamespace(), cfg)
    apply_moe_surgery(model, cfg, tid["image_pad"])
    if trained:
        torch.manual_seed(1)
        with torch.no_grad():
            for l in moe_layers(model):
                for m in (l.lora_gate, l.lora_up, l.lora_down):
                    m.B.normal_(0, 0.5)
                if l.router is not None:
                    l.router.gate.weight.normal_(0, 1.0)
    model.eval()
    return cfg, model, processor, tid, batches[0][1]


def synthetic_npz(path, L=2, E=4, aligned=True, S=30):
    rng = np.random.default_rng(0)
    tasks = np.array([TASKS[i % 3] for i in range(S)])
    counts = np.zeros((S, L, E), np.int64)
    for s in range(S):
        for l in range(L):
            if aligned:
                counts[s, l, TASKS.index(tasks[s])] = 10        # task i always uses expert i
            else:
                counts[s, l] = 5                                 # every task uses every expert equally
    gate_sum = counts * rng.uniform(0.8, 1.6, size=(1, L, E))
    np.savez(path, counts=counts, gate_sum=gate_sum, tasks=tasks, uids=np.arange(S).astype(str))


def logits(model, batch):
    with torch.no_grad():
        return model(**batch, use_cache=False).logits


def test_mutual_information():
    assert abs(mutual_information_bits(np.eye(3) * 7) - np.log2(3)) < 1e-9
    assert abs(mutual_information_bits(np.ones((3, 4)))) < 1e-12
    with tempfile.TemporaryDirectory() as d:
        synthetic_npz(os.path.join(d, "a.npz"), aligned=True)
        synthetic_npz(os.path.join(d, "u.npz"), aligned=False)
        a, u = routing_stats(os.path.join(d, "a.npz")), routing_stats(os.path.join(d, "u.npz"))
    assert np.allclose(a["mi"], np.log2(3)) and np.allclose(u["mi"], 0)
    assert np.allclose(a["f"].sum(-1), 1) and a["f"].shape == (2, 4)
    assert all(a["f_task"][t][0].argmax() == TASKS.index(t) for t in TASKS)
    assert np.allclose(a["gbar"][:, 3], 0)                              # expert 3 never used


def test_profile_gates_onehot_times_gbar():
    with tempfile.TemporaryDirectory() as d:
        synthetic_npz(os.path.join(d, "a.npz"))
        st = routing_stats(os.path.join(d, "a.npz"))
    g = profile_gates(st, "document_ocr")
    assert ((g > 0).sum(-1) == 1).all() and np.allclose(g[:, 1], st["gbar"][:, 1])


def test_full_merge_export_equals_fixed_mode():
    cfg, model, _, tid, batch = tiny()
    with tempfile.TemporaryDirectory() as d:
        synthetic_npz(os.path.join(d, "u.npz"), aligned=False)
        st = routing_stats(os.path.join(d, "u.npz"))
    apply_full_merge(model, st)
    ref = logits(model, batch)
    merged = copy.deepcopy(model)
    merge_into_base(merged)
    assert not any(isinstance(b.mlp, MoELayer) for b in get_lm_layers(merged))
    assert torch.allclose(logits(merged, batch), ref, atol=1e-4)
    set_routed(model)
    assert not torch.allclose(logits(model, batch), ref, atol=1e-4)    # routed != merged here


def test_profile_export_equals_fixed_mode():
    cfg, model, _, tid, batch = tiny()
    with tempfile.TemporaryDirectory() as d:
        synthetic_npz(os.path.join(d, "a.npz"))
        st = routing_stats(os.path.join(d, "a.npz"))
    apply_profile(model, st, "spatial_reasoning")
    ref = logits(model, batch)
    merged = copy.deepcopy(model)
    merge_into_base(merged)
    assert torch.allclose(logits(merged, batch), ref, atol=1e-4)


def test_selective_endpoints():
    cfg, model, _, tid, batch = tiny()
    with tempfile.TemporaryDirectory() as d:
        synthetic_npz(os.path.join(d, "u.npz"), aligned=False)
        st = routing_stats(os.path.join(d, "u.npz"))
    routed = logits(model, batch)
    apply_full_merge(model, st)
    full = logits(model, batch)
    L = len(moe_layers(model))
    assert apply_selective(model, st, L) == list(range(L))
    assert torch.allclose(logits(model, batch), routed)
    assert apply_selective(model, st, 0) == []
    assert torch.allclose(logits(model, batch), full)
    try:
        apply_selective(model, st, 1)
        merge_into_base(copy.deepcopy(model))
        raise AssertionError("routed layer exported")
    except ValueError:
        pass


def test_text_routing_cannot_be_exported_dense_can():
    cfg, model, _, tid, batch = tiny(route_tokens="text")
    with tempfile.TemporaryDirectory() as d:
        synthetic_npz(os.path.join(d, "u.npz"), aligned=False)
        st = routing_stats(os.path.join(d, "u.npz"))
    apply_full_merge(model, st)
    try:
        merge_into_base(model)
        raise AssertionError("text-routing export allowed")
    except ValueError:
        pass
    cfg, model, _, tid, batch = tiny("configs/dense16.yaml")
    ref = logits(model, batch)
    merge_into_base(model)
    assert torch.allclose(logits(model, batch), ref, atol=1e-4)


def test_stats_from_real_evaluate_output():
    cfg, model, proc, tid, batch = tiny()
    prompt = {k: v[:, :16] if k in ("input_ids", "attention_mask", "mm_token_type_ids") else v
              for k, v in batch.items()}
    samples = [{"uid": f"u{i}", "task": TASKS[i % 3], "question": "q", "answers": ["1"], "image": "x"}
               for i in range(6)]
    with tempfile.TemporaryDirectory() as d:
        rp = os.path.join(d, "r.npz")
        evaluate(model, proc, samples, d, tid, routing_path=rp, max_new_tokens=2,
                 encode_fn=lambda p, s, r: {k: v.clone() for k, v in prompt.items()}, warmup=0, progress=False)
        st = routing_stats(rp)
    assert st["f"].shape == (2, 4) and st["n_samples"] == 6 and st["n_tokens"] == 6 * 12
    assert np.all(st["mi"] >= -1e-12)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print(f"All {len(tests)} convert tests passed.")
