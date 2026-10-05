import torch
import torch.nn as nn
import math
from router import TopKRouter

class ExpertLoRA(nn.Module):
    def __init__(self, hidden_size, intermediate_size, rank=16, alpha=32):
        super().__init__()
        self.scaling = alpha / rank
        self.gate_A = nn.Linear(hidden_size, rank, bias=False)
        self.gate_B = nn.Linear(rank, intermediate_size, bias=False)
        self.up_A = nn.Linear(hidden_size, rank, bias=False)
        self.up_B = nn.Linear(rank, intermediate_size, bias=False)
        self.down_A = nn.Linear(intermediate_size, rank, bias=False)
        self.down_B = nn.Linear(rank, hidden_size, bias=False)
        
        nn.init.zeros_(self.gate_B.weight)
        nn.init.zeros_(self.up_B.weight)
        nn.init.zeros_(self.down_B.weight)

class SharedBaseLoRAExpert(nn.Module):
    def __init__(self, original_mlp, hidden_size, intermediate_size, num_experts, rank=16, alpha=32):
        super().__init__()
        # Shared frozen base
        self.base_mlp = original_mlp
        for param in self.base_mlp.parameters():
            param.requires_grad = False
            
        # Independent LoRA adapters per expert
        self.experts = nn.ModuleList([
            ExpertLoRA(hidden_size, intermediate_size, rank, alpha) 
            for _ in range(num_experts)
        ])
        self.act_fn = nn.SiLU()

    def forward(self, x, expert_idx):
        expert = self.experts[expert_idx]
        
        # Base outputs
        base_gate = self.base_mlp.gate_proj(x)
        base_up = self.base_mlp.up_proj(x)
        
        # LoRA outputs
        lora_gate = expert.gate_B(expert.gate_A(x)) * expert.scaling
        lora_up = expert.up_B(expert.up_A(x)) * expert.scaling
        
        # Combined activations
        intermediate = self.act_fn(base_gate + lora_gate) * (base_up + lora_up)
        
        # Down projection
        base_down = self.base_mlp.down_proj(intermediate)
        lora_down = expert.down_B(expert.down_A(intermediate)) * expert.scaling
        
        return base_down + lora_down

class MoELayer(nn.Module):
    def __init__(self, original_mlp, hidden_size, intermediate_size, num_experts=4, top_k=1, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        self.router = TopKRouter(hidden_size, num_experts, top_k=top_k)
        
        self.shared_experts = SharedBaseLoRAExpert(
            original_mlp, hidden_size, intermediate_size, num_experts
        )
        
        self.metrics = {}

    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, original_shape[-1])
        num_tokens = hidden_states.shape[0]
        
        routing_weights, expert_indices, router_metrics = self.router(hidden_states)
        self.metrics = router_metrics
        
        final_output = torch.zeros_like(hidden_states)
        
        # P0-3: Strict capacity semantics
        expert_capacity = int(math.ceil((num_tokens / self.num_experts) * self.capacity_factor))
        dropped_tokens_total = 0
        processed_mask = torch.zeros(num_tokens, dtype=torch.bool, device=hidden_states.device)

        for i in range(self.num_experts):
            expert_mask = (expert_indices == i)
            token_mask = expert_mask.any(dim=-1)
            
            num_assigned = token_mask.sum().item()
            if num_assigned == 0:
                continue
                
            if self.training and num_assigned > expert_capacity:
                assigned_weights = routing_weights[expert_mask]
                token_idx = token_mask.nonzero(as_tuple=True)[0]
                _, sorted_idx = torch.topk(assigned_weights, expert_capacity)
                top_token_idx = token_idx[sorted_idx]
                
                new_token_mask = torch.zeros_like(token_mask)
                new_token_mask[top_token_idx] = True
                
                dropped = num_assigned - expert_capacity
                dropped_tokens_total += dropped
                
                token_mask = new_token_mask

            expert_tokens = hidden_states[token_mask]
            expert_output = self.shared_experts(expert_tokens, expert_idx=i)
            
            selected_expert_mask = expert_mask[token_mask]
            selected_weights = routing_weights[token_mask]
            weight_per_token = selected_weights[selected_expert_mask].unsqueeze(-1)
            
            final_output[token_mask] += expert_output * weight_per_token
            processed_mask[token_mask] = True

        # Fallback path for dropped tokens (Zero-LoRA base MLP path)
        unprocessed_mask = ~processed_mask
        if unprocessed_mask.any():
            unprocessed_tokens = hidden_states[unprocessed_mask]
            fallback_output = self.shared_experts.base_mlp(unprocessed_tokens)
            final_output[unprocessed_mask] += fallback_output

        self.metrics["drop_rate"] = dropped_tokens_total / num_tokens if num_tokens > 0 else 0.0
        
        final_output = final_output.view(*original_shape)
        return final_output
