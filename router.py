import torch
import torch.nn as nn
import torch.nn.functional as F

class TopKRouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int = 4, top_k: int = 2, alpha: float = 0.01, z_loss_coeff: float = 0.001):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.alpha = alpha
        self.z_loss_coeff = z_loss_coeff
        
        # Gating network
        self.gating = nn.Linear(hidden_size, num_experts, bias=False)
        
    def forward(self, hidden_states: torch.Tensor):
        # Flatten batch and seq dimensions if needed
        original_shape = hidden_states.shape
        if len(original_shape) > 2:
            hidden_states = hidden_states.view(-1, original_shape[-1])
            
        logits = self.gating(hidden_states)
        
        # Get raw probabilities over all experts for auxiliary loss
        routing_probs = F.softmax(logits, dim=-1)
        
        # Top-K selection
        topk_logits, expert_indices = torch.topk(logits, self.top_k, dim=-1)
        
        # Normalize top-k probabilities (for multiplying with expert outputs)
        routing_weights = F.softmax(topk_logits, dim=-1)
        
        # Compute Load Balancing Auxiliary Loss
        # 1. Token fraction per expert (f_i)
        expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).sum(dim=1).float()
        token_fraction = expert_mask.mean(dim=0)
        
        # 2. Mean routing probability per expert (P_i)
        mean_routing_prob = routing_probs.mean(dim=0)
        
        # 3. Auxiliary Loss calculation
        aux_loss = self.alpha * self.num_experts * torch.sum(token_fraction * mean_routing_prob)
        
        # Compute Router Z-Loss
        z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1)))
        aux_loss = aux_loss + self.z_loss_coeff * z_loss
        
        # Reshape back to original shape if necessary
        if len(original_shape) > 2:
            routing_weights = routing_weights.view(*original_shape[:-1], self.top_k)
            expert_indices = expert_indices.view(*original_shape[:-1], self.top_k)
            
        return routing_weights, expert_indices, aux_loss
