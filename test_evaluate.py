"""CPU tests for evaluate.py on a tiny random Qwen2-VL (no download).   python test_evaluate.py"""
import copy
import importlib.util
import json
import os
import tempfile
import types

import numpy as np
import torch

from config import load
from evaluate import evaluate, load_samples
from layers import moe_layers
from model import apply_moe_surgery, load_adapters, save_adapters

_spec = importlib.util.spec_from_file_location(
    "crm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_real_model.py"))
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)

TASKS = ["document_ocr", "chart_qa", "spatial_reasoning"]
SAMPLES = [{"uid": f"u{i}", "task": TASKS[i % 3], "question": "q", "answers": ["1"], "image": "x.jpg"}
           for i in range(6)]


def tiny(num_experts=4, route_tokens="all"):
    cfg = load("configs/moe.yaml")
    cfg.moe.rank, cfg.moe.alpha = 4, 8
    cfg.moe.num_experts, cfg.moe.route_tokens = num_experts, route_tokens
    model, processor, token_ids, batches = crm.load_tiny(types.SimpleNamespace(), cfg)
    prompt = batches[0][2]
    # vary the prompt per sample so outputs differ between samples
    def enc(proc, s, root):
        b = {k: v.clone() for k, v in prompt.items()}
        b["input_ids"][0, 1] = 5 + int(s["uid"][1:])
        return b
    return cfg, model, processor, token_ids, enc


def run(model, processor, token_ids, enc, **kw):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "p.jsonl")
        res = evaluate(model, processor, SAMPLES, d, token_ids, out_path=out, max_new_tokens=6,
                       encode_fn=enc, warmup=1, progress=False, **kw)
        rows = [json.loads(l) for l in open(out)]
        return res, rows


def test_identity_base_vs_surgery():
    cfg, model, proc, tid, enc = tiny()
    base_rows = run(model, proc, tid, enc)[1]
    apply_moe_surgery(model, cfg, tid["image_pad"])
    moe_rows = run(model, proc, tid, enc)[1]
    assert [r["raw"] for r in base_rows] == [r["raw"] for r in moe_rows]


def test_rows_and_summary():
    cfg, model, proc, tid, enc = tiny()
    apply_moe_surgery(model, cfg, tid["image_pad"])
    res, rows = run(model, proc, tid, enc)
    assert len(rows) == 6 and all(r["img_tokens"] == 4 for r in rows)
    assert {"uid", "task", "pred", "answers", "score", "em", "latency", "img_tokens"} <= set(rows[0])
    assert set(res["per_task"]) == set(TASKS) and all(v["n"] == 2 for v in res["per_task"].values())
    assert 0.0 <= res["macro"] <= 1.0


def test_routing_recording_text_tokens_only():
    for route in ("all", "text"):
        cfg, model, proc, tid, enc = tiny(route_tokens=route)
        apply_moe_surgery(model, cfg, tid["image_pad"])
        with tempfile.TemporaryDirectory() as d:
            rp = os.path.join(d, "r.npz")
            evaluate(model, proc, SAMPLES, d, tid, routing_path=rp, max_new_tokens=4,
                     encode_fn=enc, warmup=0, progress=False)
            with np.load(rp) as npz:                          # close the file (Windows locks it)
                z = {k: npz[k] for k in npz.files}
            L = len(moe_layers(model))
            assert z["counts"].shape == (6, L, 4) and z["gate_sum"].shape == (6, L, 4)
            n_text = 16 - 4                                   # tiny prompt: 16 tokens, 4 image tokens
            assert (z["counts"].sum(-1) == n_text).all(), z["counts"].sum(-1)
            assert list(z["tasks"]) == [s["task"] for s in SAMPLES]
            gbar = z["gate_sum"].sum(0) / np.maximum(z["counts"].sum(0), 1)
            assert np.all(gbar[z["counts"].sum(0) > 0] > 0)
        # recorder hooks must be removed afterwards
        assert all(len(l.router._forward_hooks) == 0 for l in moe_layers(model))


def test_adapters_roundtrip_changes_and_restores_preds():
    cfg, model, proc, tid, enc = tiny()
    fresh = copy.deepcopy(model)
    apply_moe_surgery(model, cfg, tid["image_pad"])
    base_raw = [r["raw"] for r in run(model, proc, tid, enc)[1]]
    torch.manual_seed(1)
    with torch.no_grad():
        for l in moe_layers(model):
            for m in (l.lora_gate, l.lora_up, l.lora_down):
                m.B.normal_(0, 2.0)
            l.router.gate.weight.normal_(0, 1.0)
    trained_raw = [r["raw"] for r in run(model, proc, tid, enc)[1]]
    assert trained_raw != base_raw, "trained adapters should change predictions"
    with tempfile.TemporaryDirectory() as d:
        save_adapters(model, d)
        apply_moe_surgery(fresh, cfg, tid["image_pad"])
        load_adapters(fresh, d)
    assert [r["raw"] for r in run(fresh, proc, tid, enc)[1]] == trained_raw


def test_state_restored_and_limit_per_task():
    cfg, model, proc, tid, enc = tiny()
    apply_moe_surgery(model, cfg, tid["image_pad"])
    model.train()
    run(model, proc, tid, enc)
    assert model.training
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.jsonl")
        with open(p, "w") as f:
            for t in TASKS:                                   # grouped by task, like dev/test
                for i in range(5):
                    f.write(json.dumps({"uid": f"{t}{i}", "task": t}) + "\n")
        s = load_samples(p, limit_per_task=2)
        assert [x["task"] for x in s] == [t for t in TASKS for _ in range(2)]


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}")
    print(f"All {len(tests)} evaluate tests passed.")
