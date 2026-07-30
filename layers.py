import torch
import torch.nn as nn
import copy
from router import TopKRouter

class MoELayer(nn.Module):
    def __init__(self, mlp_block: nn.Module, hidden_size: int, num_experts: int = 4, top_k: int = 2, capacity_factor: float = 1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        # Router
        self.router = TopKRouter(hidden_size=hidden_size, num_experts=num_experts, top_k=top_k)
        
        # Experts (independent copies of the original MLP block)
        self.experts = nn.ModuleList([copy.deepcopy(mlp_block) for _ in range(num_experts)])
        
    def forward(self, hidden_states: torch.Tensor):
        original_shape = hidden_states.shape
        # Flatten sequence for processing
        if len(original_shape) > 2:
            hidden_states = hidden_states.view(-1, original_shape[-1])
            
        # Get routing logic
        routing_weights, expert_indices, aux_loss = self.router(hidden_states)
        
        final_output = torch.zeros_like(hidden_states)
        
        total_tokens = hidden_states.shape[0]
        expert_capacity = int((total_tokens / self.num_experts) * self.capacity_factor)
        
        # Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert
            expert_mask = (expert_indices == i)
            
            # Boolean mask for tokens routed to this expert
            token_mask = expert_mask.any(dim=-1)
            
            if not token_mask.any():
                continue
                
            num_tokens = token_mask.sum().item()
            if num_tokens > expert_capacity:
                # We need to drop tokens exceeding the capacity.
                # Get the routing weights for the tokens assigned to this expert.
                expert_weights = torch.full((total_tokens,), float('-inf'), device=hidden_states.device)
                token_idx = token_mask.nonzero(as_tuple=True)[0]
                
                # Extract routing weights specifically assigned to this expert for each token
                mask_for_expert = expert_mask[token_idx]
                weights_for_expert = routing_weights[token_idx][mask_for_expert]
                
                expert_weights[token_idx] = weights_for_expert
                
                # Keep only the top 'expert_capacity' tokens
                _, top_token_idx = torch.topk(expert_weights, expert_capacity)
                
                token_mask = torch.zeros_like(token_mask, dtype=torch.bool)
                token_mask[top_token_idx] = True
                
            # Extract actual tokens
            expert_tokens = hidden_states[token_mask]
            
            # Pass through the expert
            expert_output = expert(expert_tokens)
            
            # Apply corresponding routing weights
            selected_weights = routing_weights[token_mask]
            selected_expert_mask = expert_mask[token_mask]
            weight_per_token = selected_weights[selected_expert_mask].unsqueeze(-1)
            
            expert_output = expert_output * weight_per_token
            
            # Accumulate to final output
            final_output[token_mask] += expert_output
            
        # Reshape back to original sequence shape
        if len(original_shape) > 2:
            final_output = final_output.view(*original_shape)
            
        self.latest_aux_loss = aux_loss
        return final_output
