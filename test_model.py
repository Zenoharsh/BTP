"""CPU tests for model.py surgery + adapter I/O using a tiny Qwen2-like stand-in. Run: python test_model.py"""
import tempfile
import types
import torch
import torch.nn as nn
from layers import MoELayer, moe_layers
from model import apply_moe_surgery, save_adapters, load_adapters, adapter_state_dict

H, I, V, IMG = 32, 64, 50, 7


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, I, bias=False); self.up_proj = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False); self.act_fn = nn.SiLU()
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.self_attn = nn.Linear(H, H); self.mlp = MLP()
    def forward(self, x):
        return x + self.mlp(x + self.self_attn(x))


class LM(nn.Module):
    def __init__(self, n=3):
        super().__init__(); self.layers = nn.ModuleList([Block() for _ in range(n)])


class Inner(nn.Module):
    def __init__(self):
        super().__init__(); self.language_model = LM(); self.embed = nn.Embedding(V, H)


class FakeQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner(); self.lm_head = nn.Linear(H, V)
        self.config = types.SimpleNamespace(text_config=types.SimpleNamespace(hidden_size=H, intermediate_size=I))
    def forward(self, input_ids=None):
        x = self.model.embed(input_ids)
        for b in self.model.language_model.layers:
            x = b(x)
        return self.lm_head(x)


def cfg(route="all", E=4, rank=4):
    moe = types.SimpleNamespace(num_experts=E, rank=rank, alpha=2 * rank, top_k=1, capacity_factor=None,
                                route_tokens=route, lb_coef=0.01, z_coef=1e-3)
    return types.SimpleNamespace(moe=moe)


def build(route="all", E=4):
    torch.manual_seed(0)
    m = FakeQwen()
    ref = FakeQwen(); ref.load_state_dict(m.state_dict())
    apply_moe_surgery(m, cfg(route, E), IMG)
    return m, ref


def perturb(m):
    for l in moe_layers(m):
        for mod in (l.lora_gate, l.lora_up, l.lora_down):
            nn.init.normal_(mod.B, std=0.1)


IDS = torch.tensor([[IMG, IMG, IMG, 3, 4, 5]])


def test_all_mlps_replaced_and_identity():
    m, ref = build()
    assert len(moe_layers(m)) == 3
    assert torch.allclose(m(input_ids=IDS), ref(input_ids=IDS), atol=1e-6)


def test_only_lora_router_trainable_base_untouched():
    m, _ = build()
    tr = [n for n, p in m.named_parameters() if p.requires_grad]
    assert tr and all((".lora_" in n) or (".router." in n) for n in tr)
    assert all(p.dtype == torch.float32 for n, p in m.named_parameters() if p.requires_grad)
    assert not any(p.requires_grad for n, p in m.named_parameters() if "base_mlp" in n or "self_attn" in n)


def test_hook_installed_text_routing():
    m, ref = build(route="text")
    perturb(m)
    m(input_ids=IDS)
    l0 = moe_layers(m)[0]
    assert l0.token_is_image is not None and l0.token_is_image.tolist() == [True, True, True, False, False, False]
    assert l0.metrics["routable_tokens"] == 3


def test_gradients_reach_lora_and_router():
    m, _ = build()
    perturb(m)
    m(input_ids=IDS).logsumexp(-1).mean().backward()
    for l in moe_layers(m):
        assert l.lora_gate.A.grad.abs().sum() > 0
        assert l.router.gate.weight.grad.abs().sum() > 0


def test_save_load_roundtrip():
    m, _ = build()
    perturb(m)
    out = m(input_ids=IDS)
    with tempfile.TemporaryDirectory() as d:
        save_adapters(m, d, {"variant": "test"})
        sd = torch.load(f"{d}/adapters.pt")
        assert all((".lora_" in k) or (".router." in k) for k in sd)
        m2, _ = build()
        assert not torch.allclose(m2(input_ids=IDS), out)
        load_adapters(m2, d)
        assert torch.allclose(m2(input_ids=IDS), out, atol=1e-6)


def test_dense_variant_has_no_router():
    m, _ = build(E=1)
    assert all(l.router is None for l in moe_layers(m))
    assert not any(".router." in k for k in adapter_state_dict(m))


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t(); print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} model tests passed.")