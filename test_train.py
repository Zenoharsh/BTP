"""CPU tests for kd.py, scripts/cache_teacher.py and train.py on a tiny random Qwen2-VL.   python test_train.py"""
import copy
import importlib.util
import json
import math
import os
import tempfile
import types

import torch
import torch.nn.functional as F

from config import load
from kd import answer_shift_mask, sparse_kd
from layers import moe_layers
from model import apply_moe_surgery, load_adapters
from scripts.cache_teacher import cache_one, cache_path, check
from train import TeacherCache, micro_batch_loss, train

_spec = importlib.util.spec_from_file_location(
    "crm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "check_real_model.py"))
crm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crm)


def tiny(config="configs/moe.yaml", n_samples=6, **over):
    cfg = load(config)
    cfg.moe.rank, cfg.moe.alpha = 4, 8
    for k, v in over.items():
        sec, key = k.split("__")
        setattr(getattr(cfg, sec), key, v)
    model, processor, token_ids, batches = crm.load_tiny(types.SimpleNamespace(), cfg)
    full = batches[0][1]
    samples = []
    for i in range(n_samples):
        b = {k: v.clone() for k, v in full.items()}
        b["input_ids"][0, 1] = 20 + i                # different question token
        b["input_ids"][0, -4] = 30 + i               # different answer token
        samples.append({**b, "uids": [f"s{i}"], "tasks": ["chart_qa"]})
    return cfg, model, token_ids, samples


def build_cache(teacher, samples, token_ids, d, k=20):
    for s in samples:
        inputs = {kk: v for kk, v in s.items() if torch.is_tensor(v)}
        torch.save(cache_one(teacher, inputs, token_ids, 2.0, k), cache_path(d, s["uids"][0]))


# --------------------------------------------------------------------------- kd math
def test_sparse_kd_full_support_equals_kl():
    torch.manual_seed(0)
    s, t = torch.randn(5, 30), torch.randn(5, 30)
    T = 2.0
    tp = F.softmax(t / T, -1)
    top = tp.topk(30, -1)
    ref = F.kl_div(F.log_softmax(s / T, -1), tp, reduction="batchmean") * T * T
    assert torch.allclose(sparse_kd(s, top.indices, top.values, T), ref, atol=1e-5)
    assert sparse_kd(t, top.indices, top.values, T).abs() < 1e-5      # identical -> 0


def test_cache_entry_and_identity_kd():
    """Teacher = unmodified model; student = same model after surgery (identity at init).
    With k = full vocab, KD ~ 0. (With k < V the floor is -log(topk_mass) * T^2, see next test.)"""
    cfg, model, tid, samples = tiny()
    teacher = copy.deepcopy(model).eval()
    with tempfile.TemporaryDirectory() as d:
        build_cache(teacher, samples, tid, d, k=128)
        c = torch.load(cache_path(d, "s0"))
        L = samples[0]["input_ids"].shape[1]
        assert c["seq_len"] == L and c["mask"].numel() == L - 1
        assert c["probs"].shape == (int(c["mask"].sum()), 128) and c["indices"].dtype == torch.int32
        assert c["mask"].sum() == 3                                 # 2 answer tokens + <|im_end|>
        assert ((c["topk_mass"] > 0) & (c["topk_mass"] <= 1.0001)).all()
        apply_moe_surgery(model, cfg, tid["image_pad"])
        model.eval()
        cache = TeacherCache(d, [s["uids"][0] for s in samples])
        _, parts = micro_batch_loss(model, samples[0], tid, cfg, cache)
        assert parts["kd"] < 2e-3, parts                           # fp16 cache rounding only


def test_topk_floor_equals_minus_log_mass():
    """Documented property of sparse KD on a truncated support: for an identical student the loss is
    mean(-log topk_mass) * T^2, so topk_mass close to 1 is required (reported by the cache check)."""
    torch.manual_seed(0)
    T, logits = 2.0, torch.randn(4, 128)
    p = F.softmax(logits / T, -1).topk(20, -1)
    floor = (-p.values.sum(-1).log()).mean() * T * T
    assert torch.allclose(sparse_kd(logits, p.indices, p.values, T), floor, atol=1e-5)


def test_length_mismatch_is_caught():
    cfg, model, tid, samples = tiny()
    with tempfile.TemporaryDirectory() as d:
        build_cache(copy.deepcopy(model), samples, tid, d)
        apply_moe_surgery(model, cfg, tid["image_pad"])
        cache = TeacherCache(d, ["s0"])
        cache.entries["s0"]["seq_len"] += 1
        try:
            micro_batch_loss(model, samples[0], tid, cfg, cache)
            raise AssertionError("mismatch not detected")
        except RuntimeError as e:
            assert "length mismatch" in str(e)


def test_missing_cache_fails_early():
    with tempfile.TemporaryDirectory() as d:
        try:
            TeacherCache(d, ["nope"])
            raise AssertionError("missing cache not detected")
        except FileNotFoundError:
            pass


# --------------------------------------------------------------------------- training loop
def run_training(cfg, model, tid, samples, d, cache=None, dev_fn=None):
    return train(cfg, model, tid, samples, os.path.join(d, "run"), cache=cache, dev_fn=dev_fn,
                 cfg_dict={"test": True}, log_every=1000)


def test_overfit_moe_with_kd_and_telemetry():
    cfg, model, tid, samples = tiny(train__epochs=30, train__grad_accum=2, train__lr=3e-3,
                                    train__router_lr=1e-2, train__warmup_ratio=0.0, train__eval_every=10,
                                    loss__kd=0.1)
    with torch.no_grad():   # random tiny head has a tiny logit range: CE could never get near 0 otherwise
        model.lm_head.weight.mul_(30)
    teacher = copy.deepcopy(model).eval()
    apply_moe_surgery(model, cfg, tid["image_pad"])
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    with tempfile.TemporaryDirectory() as d:
        build_cache(teacher, samples, tid, d)
        cache = TeacherCache(d, [s["uids"][0] for s in samples])
        calls = []

        def dev_fn(m, epoch):
            calls.append((epoch, m.training))
            return {"macro": epoch / 100, "per_task": {"chart_qa": {"score": epoch / 100}}}

        summ = run_training(cfg, model, tid, samples, d, cache, dev_fn)
        logs = [json.loads(l) for l in open(os.path.join(d, "run", "train_log.jsonl"))]
        steps = [r for r in logs if r["type"] == "step"]
        assert len(steps) == 30 * math.ceil(6 / 2) == summ["steps"]
        first, last = steps[0], steps[-1]
        assert last["ce"] < 0.1 * first["ce"], (first["ce"], last["ce"])
        for key in ("lr", "ce", "kd", "aux", "sec_per_step", "peak_vram_mib", "routing"):
            assert key in last
        r = last["routing"]
        assert {"entropy_token_mean_mean", "entropy_token_mean_max", "overflow_rate_max", "load_cv_mean"} <= set(r)
        assert set(r["post_counts"]) == {"0", "1"}                   # tiny model has 2 layers
        assert first["lr"] == {"lora": 3e-3, "router": 1e-2} and last["lr"]["lora"] < 1e-5   # cosine decays to ~0
        assert [c[0] for c in calls] == [10, 20, 30] and all(c[1] for c in calls)  # called in train mode
        assert summ["best_epoch"] == 30
        assert os.path.exists(os.path.join(d, "run", "best", "adapters.pt"))
        # best adapters reload into a fresh surgery model and reproduce the trained logits
        fresh = copy.deepcopy(teacher)
        apply_moe_surgery(fresh, cfg, tid["image_pad"])
        load_adapters(fresh, os.path.join(d, "run", "best"))
        model.eval(); fresh.eval()
        inp = {k: v for k, v in samples[0].items() if torch.is_tensor(v)}
        with torch.no_grad():
            assert torch.allclose(model(**inp).logits, fresh(**inp).logits, atol=1e-5)


def test_grad_accum_window_and_short_last_window():
    cfg, model, tid, samples = tiny(n_samples=5, train__epochs=2, train__grad_accum=2, loss__kd=0.0)
    apply_moe_surgery(model, cfg, tid["image_pad"])
    with tempfile.TemporaryDirectory() as d:
        summ = run_training(cfg, model, tid, samples, d)
        assert summ["steps"] == 2 * 3                                # 5 micro-batches / 2 -> 3 steps per epoch


def test_dense_baseline_runs():
    cfg, model, tid, samples = tiny("configs/dense16.yaml", train__epochs=2, train__grad_accum=3, loss__kd=0.0)
    apply_moe_surgery(model, cfg, tid["image_pad"])
    assert all(l.router is None for l in moe_layers(model))
    with tempfile.TemporaryDirectory() as d:
        run_training(cfg, model, tid, samples, d)
        steps = [json.loads(l) for l in open(os.path.join(d, "run", "train_log.jsonl"))]
        assert all("routing" not in s and s["aux"] == 0.0 for s in steps)


def test_cache_integrity_check():
    cfg, model, tid, samples = tiny()
    with tempfile.TemporaryDirectory() as d:
        build_cache(model, samples, tid, d)
        split = os.path.join(d, "train.jsonl")
        with open(split, "w") as f:
            for s in samples:
                f.write(json.dumps({"uid": s["uids"][0]}) + "\n")
        assert check(d, [split])
        os.remove(cache_path(d, "s3"))
        assert not check(d, [split])


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"[PASS] {t.__name__}", flush=True)
    print(f"All {len(tests)} train tests passed.")
