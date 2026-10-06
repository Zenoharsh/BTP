"""CPU unit tests for the v3 MoE-LoRA layer. Run:  python test_v3.py   (or pytest test_v3.py)"""
import math
import torch
import torch.nn as nn
from layers import MoELayer, StackedLoRA

H, I = 32, 64
torch.manual_seed(0)


class DummyMLP(nn.Module):  # mirrors Qwen2MLP
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def make(E=4, **kw):
    base = DummyMLP()
    return base, MoELayer(base, H, I, num_experts=E, rank=4, alpha=8, **kw)


def randomize_B(m, std=0.1):
    for mod in (m.lora_gate, m.lora_up, m.lora_down):
        nn.init.normal_(mod.B, std=std)


def manual_reference(m, x, g):
    """Explicit per-expert loop implementation used to check the fused kernel."""
    b = m.base_mlp
    def d(mod, inp):
        return sum(g[:, e:e + 1] * (inp @ mod.expert_delta_weight(e).T) for e in range(m.num_experts))
    gate = b.gate_proj(x) + d(m.lora_gate, x)
    up = b.up_proj(x) + d(m.lora_up, x)
    h = m.act_fn(gate) * up
    return b.down_proj(h) + d(m.lora_down, h)


def test_identity_at_init():
    base, m = make()
    x = torch.randn(2, 10, H)
    assert torch.allclose(m(x), base(x), atol=1e-6)


def test_router_gets_task_gradient():
    """The bug in the old code: router grad ~1e-9. Must now be clearly non-zero."""
    _, m = make()
    randomize_B(m)
    x = torch.randn(1, 50, H)
    m(x).pow(2).mean().backward()          # task loss only, no aux loss
    assert m.router.gate.weight.grad.abs().max().item() > 1e-4


def test_fused_equals_loop():
    _, m = make()
    randomize_B(m)
    x = torch.randn(40, H)
    with torch.no_grad():
        y = m(x)
        g = m._gates(x)
        ref = manual_reference(m, x, g)
    assert torch.allclose(y, ref, atol=1e-5)


def test_gate_value_top1():
    _, m = make()
    x = torch.randn(30, H)
    g = m._gates(x)
    assert ((g > 0).sum(-1) == 1).all()                # exactly one expert per token
    # with near-uniform init, the selected gate is ~ E * 0.25 = 1
    assert torch.allclose(g.sum(-1), torch.ones(30), atol=1e-2)


def test_train_eval_identical():
    _, m = make()
    randomize_B(m)
    x = torch.randn(25, H)
    m.train(); a = m(x)
    m.eval(); b = m(x)
    assert torch.equal(a, b)


def test_text_only_routing():
    base, m = make(route_tokens="text")
    randomize_B(m)
    x = torch.randn(20, H)
    img = torch.zeros(20, dtype=torch.bool); img[:12] = True
    m.token_is_image = img
    y = m(x)
    assert torch.allclose(y[img], base(x[img]), atol=1e-6)            # image tokens: base only
    assert not torch.allclose(y[~img], base(x[~img]), atol=1e-4)      # text tokens: adapted
    assert m.metrics["routable_tokens"] == 8
    m.token_is_image = None                                           # decode step: all text
    assert m.metrics is not None and m(x[:1]).shape == (1, H)


def test_capacity_accounting():
    _, m = make(capacity_factor=1.25)
    with torch.no_grad():                    # force every token to expert 0
        m.router.gate.weight.zero_(); m.router.gate.weight[0, 0] = 50.0
    x = torch.randn(100, H); x[:, 0] = 1.0
    g = m._gates(x)
    cap = math.ceil(100 / 4 * 1.25)          # 32
    mt = m.metrics
    assert mt["capacity"] == cap
    assert mt["pre_counts"] == [100.0, 0.0, 0.0, 0.0]
    assert mt["post_counts"] == [float(cap), 0.0, 0.0, 0.0]
    assert mt["overflow"] == 100 - cap
    assert abs(mt["overflow_rate"] - (100 - cap) / 100) < 1e-6
    assert int((g.sum(-1) == 0).sum()) == 100 - cap      # overflow tokens -> base only
    assert sum(mt["pre_counts"]) == mt["routable_tokens"]


def test_layer_average_hides_overflow():
    """Reproduces the old log pattern: averaged counts <= capacity, yet overflow > 0."""
    counts = torch.tensor([[120., 40, 36, 30], [10, 90, 60, 66]])
    cap = math.ceil(226 * 1.25 / 4)
    overflow = (counts - cap).clamp_min(0).sum(1) / 226
    assert counts.mean(0).max() <= cap and overflow.mean() > 0


def test_fixed_mode_equals_merged_weights():
    base, m = make()
    randomize_B(m)
    w = torch.tensor([0.1, 0.4, 0.3, 0.2])
    m.mode, m.fixed_g = "fixed", w
    x = torch.randn(15, H)
    with torch.no_grad():
        y = m(x)
        dW = m.merged_delta_weights(w)
        merged = DummyMLP()
        merged.load_state_dict(base.state_dict())
        for k_, d in dW.items():
            getattr(merged, k_).weight += d
        assert torch.allclose(y, merged(x), atol=1e-5)


def test_dense_lora_baseline():
    base, m = make(E=1)
    randomize_B(m)
    x = torch.randn(9, H)
    with torch.no_grad():
        dW = m.merged_delta_weights([1.0])
        merged = DummyMLP(); merged.load_state_dict(base.state_dict())
        for k_, d in dW.items():
            getattr(merged, k_).weight += d
        assert torch.allclose(m(x), merged(x), atol=1e-5)
    assert m.router is None


def test_aux_loss_trains_router():
    _, m = make()
    x = torch.randn(64, H)
    m(x)
    m.aux_loss.backward()
    assert m.router.gate.weight.grad is not None


def test_only_lora_and_router_trainable():
    _, m = make()
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith(("lora_", "router.")) for n in names)
    assert not any(n.startswith("base_mlp") for n in names)



def test_bf16_activations():
    base, m = make()
    base.to(torch.bfloat16)                    # base in bf16 like the NF4 compute dtype
    randomize_B(m)
    x = torch.randn(1, 12, H, dtype=torch.bfloat16)
    y = m(x)
    assert y.dtype == torch.bfloat16 and torch.isfinite(y).all()
    y.float().pow(2).mean().backward()
    assert m.lora_gate.A.grad is not None and m.router.gate.weight.grad.abs().max() > 0


def test_token_type_hook():
    from layers import install_token_type_hook

    class Layer(nn.Module):
        def __init__(self, mlp): super().__init__(); self.mlp = mlp

    class Inner(nn.Module):
        def __init__(self, layers): super().__init__(); self.layers = nn.ModuleList(layers)

    class Model(nn.Module):
        def __init__(self, layer):
            super().__init__(); self.model = Inner([layer]); self.emb = nn.Embedding(100, H)
        def forward(self, input_ids=None):
            return self.model.layers[0].mlp(self.emb(input_ids))

    base, m = make(route_tokens="text")
    randomize_B(m)
    model = Model(Layer(m))
    install_token_type_hook(model, image_token_id=7)
    ids = torch.tensor([[7, 7, 7, 3, 4]])
    y = model(input_ids=ids)
    emb = model.emb(ids)
    assert torch.allclose(y[0, :3], base(emb[0, :3]), atol=1e-6)    # image tokens -> base
    y2 = model(input_ids=torch.tensor([[5]]))                         # decode step
    assert y2.shape == (1, 1, H) and m.metrics["routable_tokens"] == 1

if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} v3 tests passed.")
