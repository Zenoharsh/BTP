"""
Shared-base Mixture-of-LoRA-Experts layer (v3).

Each Qwen2 MLP block  y = down( act(gate(x)) * up(x) )  becomes

    gate'(x) = gate(x) + sum_e g_e(x) * dGate_e(x)
    up'(x)   = up(x)   + sum_e g_e(x) * dUp_e(x)
    y        = down(h) + sum_e g_e(x) * dDown_e(h),   h = act(gate'(x)) * up'(x)

where dX_e = scaling * B_e A_e is expert e's LoRA delta and g(x) is the gate vector:
    routed mode : g_e = (E/k) * p_e(x) for the k selected experts, 0 otherwise      (A1)
    fixed  mode : g   = a constant vector (used for merging / profile adapters)      (A7)
    dense  mode : E == 1, g = 1 (plain LoRA baseline with identical code path)       (A6)

Properties (all tested in test_v3.py):
- Base MLP is computed ONCE for every token and is never scaled  -> exact identity when B = 0.
- No Python loop over experts: the E low-rank adapters of a projection are stacked into one
  [E*r, in] A matrix and one [out, E*r] B matrix; the gate is applied as a block mask on the
  rank dimension (two GEMMs per projection, same FLOPs as one rank-E*r LoRA).            (A2)
- Dropless by default; optional capacity is a mask on g (overflow -> base only). Training and
  eval behave identically.                                                              (A3)
- route_tokens="text": tokens flagged as image tokens get g = 0 (base only).            (A4)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from router import TopKRouter


class StackedLoRA(nn.Module):
    """E independent LoRA adapters for one linear projection, stored stacked."""

    def __init__(self, in_features, out_features, num_experts, rank, alpha):
        super().__init__()
        self.num_experts, self.rank = num_experts, rank
        self.scaling = alpha / rank
        self.A = nn.Parameter(torch.empty(num_experts * rank, in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_features, num_experts * rank, dtype=torch.float32))
        for e in range(num_experts):  # same init as nn.Linear / PEFT LoRA-A, per expert block
            nn.init.kaiming_uniform_(self.A.data[e * rank:(e + 1) * rank], a=math.sqrt(5))

    def forward(self, x, g):
        """x: [N, in]; g: [N, E] gate values. Returns delta [N, out] in x.dtype."""
        z = F.linear(x.to(self.A.dtype), self.A)                       # [N, E*r]
        z = z * g.to(z.dtype).repeat_interleave(self.rank, dim=-1)      # block mask / gate
        return (F.linear(z, self.B) * self.scaling).to(x.dtype)

    def expert_delta_weight(self, e):
        """Dense weight delta of expert e: scaling * B_e @ A_e  -> [out, in]."""
        sl = slice(e * self.rank, (e + 1) * self.rank)
        return self.scaling * (self.B[:, sl] @ self.A[sl])


class MoELayer(nn.Module):
    def __init__(self, base_mlp, hidden_size, intermediate_size, num_experts=4, rank=16, alpha=32,
                 top_k=1, capacity_factor=None, route_tokens="all",
                 lb_coef=0.01, z_coef=1e-3):
        super().__init__()
        assert route_tokens in ("all", "text")
        self.base_mlp = base_mlp
        for p in self.base_mlp.parameters():
            p.requires_grad = False
        self.act_fn = getattr(base_mlp, "act_fn", nn.SiLU())
        self.num_experts, self.top_k = num_experts, top_k
        self.capacity_factor = capacity_factor
        self.route_tokens = route_tokens
        self.lb_coef, self.z_coef = lb_coef, z_coef

        self.router = TopKRouter(hidden_size, num_experts, top_k) if num_experts > 1 else None
        self.lora_gate = StackedLoRA(hidden_size, intermediate_size, num_experts, rank, alpha)
        self.lora_up = StackedLoRA(hidden_size, intermediate_size, num_experts, rank, alpha)
        self.lora_down = StackedLoRA(intermediate_size, hidden_size, num_experts, rank, alpha)

        # runtime state
        self.mode = "routed"          # "routed" | "fixed"
        self.fixed_g = None           # [E] tensor used in fixed mode
        self.token_is_image = None    # [N] bool, set by the model-level pre-hook (A4)
        self.record = False           # if True, keep per-token expert ids for MI analysis
        self.aux_loss = None          # differentiable LB + z loss of the last forward
        self.metrics = {}             # detached telemetry of the last forward (A8)
        self.last_expert_ids = None

    # ------------------------------------------------------------------ gates
    def _image_mask(self, n, device):
        m = self.token_is_image
        if m is None or m.numel() != n:      # e.g. decode steps: only new text tokens
            return torch.zeros(n, dtype=torch.bool, device=device)
        return m.to(device)

    def _routed_gates(self, x):
        n, E, k = x.shape[0], self.num_experts, self.top_k
        logits, probs, top_idx = self.router(x)
        sel = torch.zeros_like(probs).scatter(-1, top_idx, 1.0)          # [N,E] one-hot (k-hot)
        routable = torch.ones(n, dtype=torch.bool, device=x.device)
        if self.route_tokens == "text":
            routable = ~self._image_mask(n, x.device)
        sel = sel * routable.unsqueeze(-1)
        pre_counts = sel.sum(0)

        if self.capacity_factor is not None:                              # optional, A3
            n_r = int(routable.sum().item())
            cap = int(math.ceil(n_r * k / E * self.capacity_factor))
            # keep the highest-probability assignments per expert
            score = torch.where(sel > 0, probs, torch.full_like(probs, -1.0))
            rank_in_expert = score.argsort(0, descending=True).argsort(0)
            sel = sel * (rank_in_expert < cap)
        else:
            cap = None

        g = sel * probs * (E / k)                                          # A1
        # ---- auxiliary losses on routable tokens only, per layer
        if routable.any():
            p_r, s_r = probs[routable], (sel[routable] > 0).float()
            f = s_r.mean(0) / k                     # fraction of assignments per expert
            P = p_r.mean(0)
            lb = E * torch.sum(f * P)
            z = torch.logsumexp(logits[routable], -1).pow(2).mean()
            self.aux_loss = self.lb_coef * lb + self.z_coef * z
        else:
            self.aux_loss = logits.sum() * 0.0

        with torch.no_grad():
            post_counts = (sel > 0).float().sum(0)
            n_r = int(routable.sum().item())
            pr = probs[routable] if n_r > 0 else probs
            tok_ent = -(pr * pr.clamp_min(1e-9).log()).sum(-1).mean()
            pm = pr.mean(0)
            self.metrics = {
                "tokens": n, "routable_tokens": n_r, "capacity": cap,
                "pre_counts": pre_counts.tolist(), "post_counts": post_counts.tolist(),
                "overflow": float(pre_counts.sum() - post_counts.sum()),
                "overflow_rate": float((pre_counts.sum() - post_counts.sum()) / max(n_r * k, 1)),
                "entropy_token_mean": float(tok_ent),
                "entropy_of_mean": float(-(pm * pm.clamp_min(1e-9).log()).sum()),
                "mean_prob": pm.tolist(),
                "top1_conf": float(pr.max(-1).values.mean()),
                "load_cv": float(post_counts.std() / post_counts.mean().clamp_min(1e-9)),
            }
            if self.record:
                ids = top_idx[:, 0].clone()
                ids[~routable] = -1
                self.last_expert_ids = ids.cpu()
        return g

    def _gates(self, x):
        n, E = x.shape[0], self.num_experts
        if E == 1:                                                        # dense LoRA baseline
            self.aux_loss, self.metrics = None, {}
            return torch.ones(n, 1, device=x.device)
        if self.mode == "fixed":                                          # merged / profile
            self.aux_loss, self.metrics = None, {}
            g = self.fixed_g.to(x.device, torch.float32).expand(n, E).clone()
            if self.route_tokens == "text":
                g[self._image_mask(n, x.device)] = 0.0
            return g
        return self._routed_gates(x)

    # ---------------------------------------------------------------- forward
    def forward(self, hidden_states):
        shape = hidden_states.shape
        x = hidden_states.reshape(-1, shape[-1])
        g = self._gates(x)
        b = self.base_mlp
        gate = b.gate_proj(x) + self.lora_gate(x, g)
        up = b.up_proj(x) + self.lora_up(x, g)
        h = self.act_fn(gate) * up
        y = b.down_proj(h) + self.lora_down(h, g)
        return y.reshape(shape)

    # ------------------------------------------------------- deployment (A7)
    def merged_delta_weights(self, g):
        """Dense weight deltas equal to fixed-mode behaviour with gate vector g (length E)."""
        out = {}
        for name, mod in (("gate_proj", self.lora_gate), ("up_proj", self.lora_up),
                          ("down_proj", self.lora_down)):
            out[name] = sum(float(g[e]) * mod.expert_delta_weight(e) for e in range(self.num_experts))
        return out


# ---------------------------------------------------------------- model helpers
def get_lm_layers(model):
    """Consistent Qwen2-VL language-layer traversal (handles both HF layouts)."""
    m = model.model
    return m.language_model.layers if hasattr(m, "language_model") else m.layers


def moe_layers(model):
    return [l.mlp for l in get_lm_layers(model) if isinstance(l.mlp, MoELayer)]


def install_token_type_hook(model, image_token_id):
    """Sets token_is_image on every MoE layer from input_ids before each forward.
    Works for training and for generate(): during decoding input_ids holds only the
    new (text) tokens, so the mask is all-False."""
    layers = moe_layers(model)

    def hook(module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        mask = None if ids is None else (ids == image_token_id).reshape(-1)
        for l in layers:
            l.token_is_image = mask
    return model.register_forward_pre_hook(hook, with_kwargs=True)


def total_aux_loss(model):
    """Mean (not sum) of per-layer router losses (A5)."""
    losses = [l.aux_loss for l in moe_layers(model) if l.aux_loss is not None]
    return torch.stack(losses).mean() if losses else None
