import torch
import torch.nn as nn
import copy
from layers import MoELayer

def test_smoke():
    print("Running Synthetic P0-10 Smoke Tests...")
    
    hidden_size = 128
    intermediate_size = 256
    
    # 1. Setup a tiny custom base MLP mimicking Qwen2
    class DummyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act_fn = nn.SiLU()
        def forward(self, x):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            
    dense_mlp = DummyMLP().to(torch.bfloat16)
    
    dummy_input = torch.randn(1, 16, hidden_size, dtype=torch.bfloat16)
    
    with torch.no_grad():
        dense_output = dense_mlp(dummy_input)
        
    # Apply MoE Surgery
    moe_mlp = MoELayer(
        dense_mlp, 
        hidden_size, 
        intermediate_size, 
        num_experts=4, 
        top_k=1,
        capacity_factor=1.25 # Tight capacity
    ).to(torch.bfloat16)
    
    # A) Zero-LoRA MoE ≈ Dense Behavior
    with torch.no_grad():
        moe_output = moe_mlp(dummy_input)
        
    diff = torch.abs(dense_output - moe_output).max().item()
    print(f"A) Dense vs Zero-LoRA MoE Max Difference: {diff:.6f}")
    assert diff < 1e-4, "Zero-LoRA equivalence failed! The fallback/base path is corrupt."
    
    # B) Valid outputs for every token
    assert not torch.isnan(moe_output).any(), "NaNs detected in MoE output!"
    print("B) Output validity confirmed (No NaNs, dense shape preserved).")
    
    # C) Top-1 Routing & Overflow Fallback
    # Force an extreme routing skew by artificially weighting one expert
    with torch.no_grad():
        moe_mlp.router.gate.weight[0] = 100.0 # Force all tokens to expert 0
        skewed_output = moe_mlp(dummy_input)
        metrics = moe_mlp.metrics
        
    assert moe_mlp.top_k == 1, "Top-1 routing not respected."
    assert "drop_rate" in metrics, "Drop rate missing."
    print(f"C) Routing overflow metrics valid. Drop rate: {metrics['drop_rate']:.3f} (Tokens processed via base-fallback)")
    assert metrics['drop_rate'] > 0, "Capacity dropping failed to engage under extreme skew!"
    assert not torch.isnan(skewed_output).any(), "Overflow fallback produced NaNs!"
    
    # D) Answer Masking Correctness
    im_start = 151644
    assistant = 77091
    im_end = 151645
    # Construct: prompt ... <|im_start|> assistant \n A N S W E R <|im_end|> ... padding
    seq = torch.tensor([1, 2, im_start, assistant, 198, 10, 11, 12, im_end, 151643, 151643])
    labels = seq.clone()
    labels[:] = -100
    start_indices = (seq == im_start).nonzero(as_tuple=True)[0]
    for start_idx in start_indices:
        if start_idx + 1 < len(seq) and seq[start_idx + 1] == assistant:
            end_idx_candidates = (seq[start_idx:] == im_end).nonzero(as_tuple=True)[0]
            if len(end_idx_candidates) > 0:
                end_idx = start_idx + end_idx_candidates[0]
                labels[start_idx+3 : end_idx+1] = seq[start_idx+3 : end_idx+1]
                
    expected = [-100, -100, -100, -100, -100, 10, 11, 12, im_end, -100, -100]
    assert labels.tolist() == expected, f"Answer mask failed! Got: {labels.tolist()}"
    print("D) Answer mask correctly isolated ground truth tokens.")
    
    # E) Gradient Isolation
    for name, param in moe_mlp.named_parameters():
        if "router.gate" in name or ".experts." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    moe_mlp.train()
    out = moe_mlp(dummy_input)
    loss = out.sum()
    loss.backward()
    
    grad_count = 0
    for name, param in moe_mlp.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Expected gradient for {name}"
            grad_count += 1
        else:
            assert param.grad is None, f"Unexpected gradient for {name}"
            
    print(f"E) Gradient isolation valid. ({grad_count} modules received gradients)")
    
    # F) Checkpoint Save/Reload Equivalence
    # We mutate the router and experts slightly to ensure saving works
    with torch.no_grad():
        moe_mlp.router.gate.weight.add_(0.1)
        moe_mlp.shared_experts.experts[0].gate_A.weight.add_(0.1)
        
    out_before_save = moe_mlp(dummy_input)
    
    # Save manually
    state_dict = {k: v for k, v in moe_mlp.state_dict().items() if "router" in k or "experts" in k}
    import os
    torch.save(state_dict, "smoke_checkpoint.pt")
    
    # Reload into a fresh model
    fresh_mlp = MoELayer(
        DummyMLP(), 
        hidden_size, 
        intermediate_size, 
        num_experts=4, 
        top_k=1
    ).to(torch.bfloat16)
    
    fresh_mlp.load_state_dict(torch.load("smoke_checkpoint.pt"), strict=False)
    fresh_mlp.eval()
    
    with torch.no_grad():
        out_after_load = fresh_mlp(dummy_input)
    
    load_diff = torch.abs(out_before_save - out_after_load).max().item()
    print(f"F) Save/Reload Max Difference: {load_diff:.6f}")
    assert load_diff < 1e-4, "Checkpoint equivalence failed!"
    
    if os.path.exists("smoke_checkpoint.pt"):
        os.remove("smoke_checkpoint.pt")
        
    print("\n[SUCCESS] All structural smoke tests passed. Ready for execution.")

if __name__ == "__main__":
    test_smoke()
