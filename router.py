import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=1, load_balance_coef=1.0, z_loss_coef=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        
        # Initialize conservatively to avoid symmetry/collapse
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=1e-4)

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        if len(original_shape) > 2:
            hidden_states = hidden_states.view(-1, original_shape[-1])
            
        gate_input = hidden_states.to(dtype=self.gate.weight.dtype)
        logits = self.gate(gate_input)
        
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        metrics = {}
        
        if self.training:
            zeros = torch.zeros_like(routing_weights)
            token_to_expert_mask = zeros.scatter(-1, top_k_indices, 1.0)
            
            tokens_per_expert = token_to_expert_mask.mean(dim=0)
            router_prob_per_expert = routing_weights.mean(dim=0)
            
            balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
            z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean() if self.z_loss_coef > 0 else 0.0
            
            # Store detached metrics to avoid memory leaks
            metrics["aux_loss"] = (self.load_balance_coef * balancing_loss) + (self.z_loss_coef * z_loss)
            
            entropy = -torch.sum(routing_weights * torch.log(routing_weights + 1e-6), dim=-1).mean()
            metrics["routing_entropy"] = entropy.item()
            
            load_std = tokens_per_expert.std()
            load_mean = tokens_per_expert.mean()
            metrics["load_cv"] = (load_std / load_mean).item() if load_mean > 0 else 0.0
            
            metrics["expert_counts"] = token_to_expert_mask.sum(dim=0).detach().cpu().tolist()
            metrics["mean_routing_prob"] = router_prob_per_expert.detach().cpu().tolist()
            metrics["top1_confidence"] = routing_weights.max(dim=-1)[0].mean().item()
        else:
            metrics["aux_loss"] = 0.0
            metrics["routing_entropy"] = 0.0
            metrics["load_cv"] = 0.0
            metrics["expert_counts"] = []
            metrics["mean_routing_prob"] = []
            metrics["top1_confidence"] = 0.0
            
        if len(original_shape) > 2:
            top_k_weights = top_k_weights.view(*original_shape[:-1], self.top_k)
            top_k_indices = top_k_indices.view(*original_shape[:-1], self.top_k)
            
        return top_k_weights.to(hidden_states.dtype), top_k_indices, metrics
