import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    def __init__(self, hidden_size, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states):
        # Ensure input is 2D: [num_tokens, hidden_size]
        original_shape = hidden_states.shape
        if len(original_shape) > 2:
            hidden_states = hidden_states.view(-1, original_shape[-1])
            
        # Device Guard: Ensure router parameters match hidden_states device
        if self.gate.weight.device != hidden_states.device:
            self.gate = self.gate.to(hidden_states.device)

        # Dtype Guard: Cast hidden_states to match gate dtype (typically float32 for Z-loss stability)
        gate_input = hidden_states.to(dtype=self.gate.weight.dtype)

        logits = self.gate(gate_input)
        
        # 1. Router Z-Loss: Penalize large logits to prevent fp16 overflow
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        
        # 2. Routing probabilities
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
        
        # Top-K selection
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # Normalize top-K weights so they sum to 1 for each token
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 3. Load-Balancing Loss
        # We need the fraction of tokens routed to each expert
        zeros = torch.zeros_like(routing_weights)
        token_to_expert_mask = zeros.scatter(-1, top_k_indices, 1.0)
        
        # Fraction of tokens dispatched to each expert
        tokens_per_expert = token_to_expert_mask.mean(dim=0)
        
        # Average routing probability assigned to each expert across all tokens
        router_prob_per_expert = routing_weights.mean(dim=0)
        
        # Load balancing loss = num_experts * sum(fraction_of_tokens * avg_prob)
        balancing_loss = self.num_experts * torch.sum(tokens_per_expert * router_prob_per_expert)
        
        # Combine losses
        aux_loss = balancing_loss + 0.001 * z_loss
        
        # Reshape back to original dimensions if needed
        if len(original_shape) > 2:
            top_k_weights = top_k_weights.view(*original_shape[:-1], self.top_k)
            top_k_indices = top_k_indices.view(*original_shape[:-1], self.top_k)
            
        return top_k_weights.to(hidden_states.dtype), top_k_indices, aux_loss
