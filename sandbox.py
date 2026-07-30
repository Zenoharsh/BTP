import torch
import torch.nn as nn
from router import TopKRouter
from layers import MoELayer

def main():
    print("Initializing Sandbox environment...")
    
    # Dummy MLP block for testing
    class DummyMLP(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.fc1 = nn.Linear(hidden_size, hidden_size * 2)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
            
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))
            
    hidden_size = 128
    num_experts = 4
    top_k = 2
    
    mlp = DummyMLP(hidden_size)
    moe_layer = MoELayer(mlp, hidden_size, num_experts, top_k)
    
    # Create dummy input [batch_size, seq_len, hidden_size]
    batch_size = 2
    seq_len = 10
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    output = moe_layer(x)
    aux_loss = moe_layer.latest_aux_loss
    
    print(f"Output shape: {output.shape}")
    print(f"Auxiliary Loss: {aux_loss.item():.4f}")
    
    # Verify shape matches
    assert output.shape == x.shape, "Output shape mismatch!"
    print("Sandbox test passed successfully!")

if __name__ == "__main__":
    main()
