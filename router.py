"""
Top-k router for the shared-base Mixture-of-LoRA-Experts (v3).

Design notes (see BTP_Master_Plan_ICORT2027.md, change A1/A5):
- The router is always computed in float32 (T4 / GTX 1650 have no native bf16).
- It returns the FULL softmax probabilities. The MoE layer turns them into
  per-expert gate values g = (E / k) * p_e for the selected experts.
  We deliberately do NOT renormalise the top-k weights: for k=1 that makes
  the gate identically 1.0 and the router receives zero task gradient
  (measured 1.2e-9 on the old code).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=1, init_std=1e-4):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, dtype=torch.float32)
        nn.init.normal_(self.gate.weight, std=init_std)

    def forward(self, x):
        """x: [N, H] (any dtype). Returns logits [N,E] fp32, probs [N,E] fp32, top-k indices [N,k]."""
        logits = self.gate(x.to(self.gate.weight.dtype))
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_idx = torch.topk(probs, self.top_k, dim=-1).indices
        return logits, probs, top_idx
