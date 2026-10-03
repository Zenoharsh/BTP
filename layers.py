import torch
import torch.nn as nn
import copy
from router import TopKRouter

class MoELayer(nn.Module):
    def __init__(self, original_mlp, hidden_size, num_experts=4, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        
        self.router = TopKRouter(hidden_size, num_experts, top_k=top_k)
        
        # Clone original Qwen2MLP into 4 independent experts
        self.experts = nn.ModuleList([copy.deepcopy(original_mlp) for _ in range(num_experts)])
        
        # Native Hugging Face models don't expect a tuple (output, loss) from MLPs.
        # We store it here and extract it during the training loop.
        self.latest_aux_loss = 0.0
        
    def forward(self, hidden_states):
        original_shape = hidden_states.shape
        # Flatten to [num_tokens, hidden_size] for processing
        hidden_states = hidden_states.view(-1, original_shape[-1])
        num_tokens = hidden_states.shape[0]
        
        # 1. Routing
        routing_weights, expert_indices, aux_loss = self.router(hidden_states)
        self.latest_aux_loss = aux_loss
        
        # Because we flattened the input above, the router returns [num_tokens, top_k] 
        final_output = torch.zeros_like(hidden_states)
        
        # 2. Strict Capacity Constraints (Token Dropping)
        # Total tokens dispatched = num_tokens * top_k
        expert_capacity = int((num_tokens * self.top_k / self.num_experts) * self.capacity_factor)
        
        # 3. Process each expert
        for i, expert in enumerate(self.experts):
            # Find tokens assigned to this expert across any of their top_k choices
            expert_mask = (expert_indices == i) # [num_tokens, top_k] boolean mask
            token_mask = expert_mask.any(dim=-1) # [num_tokens] boolean mask
            
            if not token_mask.any():
                continue
                
            num_assigned_tokens = token_mask.sum().item()
            
            if num_assigned_tokens > expert_capacity:
                # Token Dropping: Too many tokens assigned. We must select the top `expert_capacity` tokens 
                # based on their routing weight to THIS specific expert.
                
                # Extract the routing weight specifically for this expert
                assigned_weights = routing_weights[expert_mask]
                
                # Get the indices of the assigned tokens in the [num_tokens] dimension
                token_idx = token_mask.nonzero(as_tuple=True)[0]
                
                # Sort the assigned tokens by their routing weight
                _, sorted_idx = torch.topk(assigned_weights, expert_capacity)
                top_token_idx = token_idx[sorted_idx]
                
                # Create a new mask that drops the lower-weighted tokens (they bypass this expert entirely)
                new_token_mask = torch.zeros_like(token_mask)
                new_token_mask[top_token_idx] = True
                token_mask = new_token_mask
                
            # Extract the tokens that survived the capacity check
            expert_tokens = hidden_states[token_mask]
            
            # Forward pass through the cloned Qwen2MLP expert
            expert_output = expert(expert_tokens)
            
            # Apply routing weights
            selected_expert_mask = expert_mask[token_mask]
            selected_weights = routing_weights[token_mask]
            weight_per_token = selected_weights[selected_expert_mask].unsqueeze(-1)
            
            # Weight the output and accumulate
            final_output[token_mask] += expert_output * weight_per_token
            
        # 4. Reconstruct original 3D shape
        final_output = final_output.view(*original_shape)
        
        return final_output
