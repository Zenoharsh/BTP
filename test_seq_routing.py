"""CPU tests for sequence-level routing (moe.route_level = "sequence") on a tiny Qwen2-VL.
    python test_seq_routing.py"""
import copy
import importlib.util
import json
import os
import tempfile
import types

import numpy as np
import torch

from config import load
from convert import apply_profile, merge_into_base, routing_stats
from evaluate import evaluate
from layers import moe_layers, prompt_mask_from_ids, set_telemetry
from model import apply_moe_surgery
from train import micro_batch_loss, train

_spec = importlib.util.spec_from_file_location(
    "crm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_real_model.py"))
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)

TASKS = ["chart_qa", "document_ocr", "spatial_reasoning"]


def tiny(trained=False, **over):
    cfg = load("configs/moe_seq.yaml")
    cfg.moe.rank, cfg.moe.alpha = 4, 8
    for k, v in over.items():
        sec, key = k.split("__")
        setattr(getattr(cfg, sec), key, v)
    model, proc, tid, batches = crm.load_tiny(types.SimpleNamespace(), cfg)
    teacher = copy.deepcopy(model)
    apply_moe_surgery(model, cfg, tid["image_pad"], tid["im_start"])
    if trained:
        torch.manual_seed(1)
        with torch.no_grad():
            for l in moe_layers(model):
                for m in (l.lora_gate, l.lora_up, l.lora_down):
                    m.B.normal_(0, 0.5)
                l.router.gate.weight.normal_(0, 1.0)
    return cfg, model, teacher, proc, tid, batches[0][1], batches[0][2]


def fwd(model, b):
    with torch.no_grad():
        return model(**b, use_cache=False).logits


def test_prompt_mask():
    ids = torch.tensor([[1, 9, 2, 3, 9, 4, 5]])                  # im_start = 9 at 1 and 4
    assert prompt_mask_from_ids(ids, 9).tolist() == [True, True, True, True, False, False, False]
    assert prompt_mask_from_ids(torch.tensor([[5]]), 9) is None   # decode step


def test_identity_at_init():
    cfg, model, teacher, _, tid, full, prompt = tiny()
    model.eval()
    assert torch.allclose(fwd(model, full), fwd(teacher, full), atol=1e-6)


def test_one_decision_per_sequence_independent_of_answer():
    cfg, model, _, _, tid, full, prompt = tiny(trained=True)
    model.eval()
    fwd(model, full)
    g_full = [l._seq_g.clone() for l in moe_layers(model)]
    other = {k: v.clone() for k, v in full.items()}
    other["input_ids"][0, -4] = 77                                  # different answer token
    fwd(model, other)
    assert all(torch.equal(a, l._seq_g) for a, l in zip(g_full, moe_layers(model)))
    fwd(model, prompt)                                              # prompt only (inference prefill)
    assert all(torch.allclose(a, l._seq_g, atol=1e-6) for a, l in zip(g_full, moe_layers(model)))
    q = {k: v.clone() for k, v in full.items()}
    q["input_ids"][0, 1] = 60                                       # different QUESTION token
    fwd(model, q)
    assert any(not torch.allclose(a, l._seq_g) for a, l in zip(g_full, moe_layers(model)))
    assert all(((l._seq_g > 0).sum(-1) == 1).all() for l in moe_layers(model))


def test_generate_matches_teacher_forced():
    """Decode steps reuse the prefill decision, so greedy generation must equal the argmax of a
    full forward pass over prompt + generated tokens."""
    cfg, model, _, _, tid, full, prompt = tiny(trained=True)
    model.eval()
    with torch.no_grad():
        out = model.generate(**prompt, max_new_tokens=5, do_sample=False, use_cache=True,
                             pad_token_id=tid["im_end"], eos_token_id=None)
    P = prompt["input_ids"].shape[1]
    b = dict(prompt)
    b["input_ids"] = out
    b["attention_mask"] = torch.ones_like(out)
    b["mm_token_type_ids"] = (out == tid["image_pad"]).long()
    pred = fwd(model, b)[0, P - 1:-1].argmax(-1)
    assert torch.equal(pred, out[0, P:]), (pred.tolist(), out[0, P:].tolist())


def test_router_gets_task_gradient_and_aux():
    cfg, model, _, _, tid, full, _ = tiny(loss__kd=0.0)
    with torch.no_grad():
        for l in moe_layers(model):
            for m in (l.lora_gate, l.lora_up, l.lora_down):
                m.B.normal_(0, 0.1)
    model.train()
    loss, parts = micro_batch_loss(model, {**full, "uids": ["a"], "tasks": ["chart_qa"]}, tid, cfg)
    loss.backward()
    assert parts["aux"] > 0
    assert all(l.router.gate.weight.grad.abs().max() > 0 for l in moe_layers(model))
    assert all(abs(float(l.usage_ema.sum()) - 1) < 1e-5 for l in moe_layers(model))


def test_training_loop_and_recording():
    cfg, model, _, proc, tid, full, prompt = tiny(loss__kd=0.0, train__epochs=3, train__grad_accum=2,
                                                  train__lr=3e-3, train__warmup_ratio=0.0)
    samples = []
    for i in range(6):
        b = {k: v.clone() for k, v in full.items()}
        b["input_ids"][0, 1] = 20 + i % 3                           # "task" token in the prompt
        b["input_ids"][0, -4] = 30 + i % 3
        samples.append({**b, "uids": [f"s{i}"], "tasks": [TASKS[i % 3]]})
    with tempfile.TemporaryDirectory() as d:
        train(cfg, model, tid, samples, os.path.join(d, "run"), cfg_dict={}, log_every=1000)
        steps = [json.loads(l) for l in open(os.path.join(d, "run", "train_log.jsonl"))]
        assert all("routing" in s and np.isfinite(s["aux"]) for s in steps)
        model.eval()

        def enc(p, s, r):
            b = {k: v.clone() for k, v in prompt.items()}
            b["input_ids"][0, 1] = 20 + int(s["uid"][1:]) % 3
            return b
        ev = [{"uid": f"u{i}", "task": TASKS[i % 3], "question": "q", "answers": ["1"], "image": "x"} for i in range(6)]
        rp = os.path.join(d, "r.npz")
        evaluate(model, proc, ev, d, tid, routing_path=rp, max_new_tokens=2, encode_fn=enc, warmup=0,
                 progress=False)
        with np.load(rp) as z:
            counts = z["counts"]
        assert (counts.sum(-1) == 12).all()                           # 16 prompt tokens - 4 image tokens
        assert ((counts > 0).sum(-1) == 1).all()                      # whole sample on one expert
        st = routing_stats(rp)
        assert st["mi"].shape == (2,)


def test_profile_export_still_exact():
    cfg, model, _, _, tid, full, _ = tiny(trained=True)
    model.eval()
    with tempfile.TemporaryDirectory() as d:
        S = 9
        counts = np.zeros((S, 2, 4), np.int64)
        tasks = np.array([TASKS[i % 3] for i in range(S)])
        for s in range(S):
            counts[s, :, TASKS.index(tasks[s])] = 12
        np.savez(os.path.join(d, "a.npz"), counts=counts, gate_sum=counts * 1.3, tasks=tasks,
                 uids=np.arange(S).astype(str))
        st = routing_stats(os.path.join(d, "a.npz"))
    apply_profile(model, st, "document_ocr")
    ref = fwd(model, full)
    merge_into_base(model)
    assert torch.allclose(fwd(model, full), ref, atol=1e-4)


def test_telemetry_off_same_outputs_no_metrics():
    """Eval with telemetry off: identical logits/generation, no aux/metrics; training unaffected."""
    for config in ("configs/moe.yaml", "configs/moe_seq.yaml"):
        cfg, model, _, _, tid, full, prompt = tiny(trained=True)
        if config == "configs/moe.yaml":
            for l in moe_layers(model):
                l.route_level = "token"
        model.eval()
        ref = fwd(model, full)
        assert all(l.metrics for l in moe_layers(model))
        assert set_telemetry(model, False) is True
        assert torch.equal(fwd(model, full), ref)
        assert all(l.metrics == {} and l.aux_loss is None for l in moe_layers(model))
        model.train()
        model(**full, use_cache=False)
        assert all(l.metrics and l.aux_loss is not None for l in moe_layers(model))
        set_telemetry(model, True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print(f"All {len(tests)} sequence-routing tests passed.")
