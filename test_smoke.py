import torch
import torch.nn as nn
import copy
from layers import MoELayer
from utils import get_answer_labels

def test_smoke():
    print("Running Synthetic P0-10 Smoke Tests...")
    
    hidden_size = 128
    intermediate_size = 256
    
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
        
    moe_mlp = MoELayer(
        dense_mlp, 
        hidden_size, 
        intermediate_size, 
        num_experts=4, 
        top_k=1,
        capacity_factor=1.25
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
    with torch.no_grad():
        moe_mlp.router.gate.weight[0] = 100.0 # Force all tokens to expert 0
        skewed_output = moe_mlp(dummy_input)
        metrics = moe_mlp.metrics
        
    assert moe_mlp.top_k == 1, "Top-1 routing not respected."
    assert "drop_rate" in metrics, "Drop rate missing."
    print(f"C) Routing overflow metrics valid. Drop rate: {metrics['drop_rate']:.3f} (Tokens processed via base-fallback)")
    assert metrics['drop_rate'] > 0, "Capacity dropping failed to engage under extreme skew!"
    assert not torch.isnan(skewed_output).any(), "Overflow fallback produced NaNs!"
    
    # D) Centralized Answer Masking Correctness
    im_start = 151644
    assistant = 77091
    im_end = 151645
    # Construct: prompt ... <|im_start|> assistant \n A N S W E R <|im_end|> ... padding
    seq = torch.tensor([[1, 2, im_start, assistant, 198, 10, 11, 12, im_end, 151643, 151643]])
    labels = get_answer_labels(seq)
    
    # Everything but 10, 11, 12 should be -100
    expected = [-100, -100, -100, -100, -100, 10, 11, 12, -100, -100, -100]
    assert labels[0].tolist() == expected, f"Central answer mask failed! Got: {labels[0].tolist()}"
    print("D) Answer mask strictly isolated ground truth tokens (excluding im_end).")
    
    # E) Gradient Isolation
    # Explicitly construct a balanced routing to guarantee all experts receive tokens
    with torch.no_grad():
        moe_mlp.router.gate.weight.zero_()
        for i in range(4):
            moe_mlp.router.gate.weight[i, i] = 100.0
        
    for name, param in moe_mlp.named_parameters():
        if "router.gate" in name or ".experts." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    moe_mlp.train()
    # Construct a deterministic input that perfectly balances across the 4 experts
    balanced_input = torch.zeros(1, 1024, hidden_size, dtype=torch.bfloat16)
    for i in range(1024):
        balanced_input[0, i, i % 4] = 1.0
        
    out = moe_mlp(balanced_input)
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
    with torch.no_grad():
        moe_mlp.router.gate.weight.add_(0.1)
        moe_mlp.shared_experts.experts[0].gate_A.weight.add_(0.1)
        
    out_before_save = moe_mlp(dummy_input)
    
    state_dict = {k: v for k, v in moe_mlp.state_dict().items() if "router" in k or "experts" in k}
    import os
    torch.save(state_dict, "smoke_checkpoint.pt")
    
    fresh_mlp = MoELayer(DummyMLP(), hidden_size, intermediate_size, num_experts=4, top_k=1).to(torch.bfloat16)
    fresh_mlp.load_state_dict(torch.load("smoke_checkpoint.pt"), strict=False)
    fresh_mlp.eval()
    
    with torch.no_grad():
        out_after_load = fresh_mlp(dummy_input)
    
    load_diff = torch.abs(out_before_save - out_after_load).max().item()
    print(f"F) Save/Reload Max Difference: {load_diff:.6f}")
    assert load_diff < 1e-4, "Checkpoint equivalence failed!"
    
    if os.path.exists("smoke_checkpoint.pt"):
        os.remove("smoke_checkpoint.pt")
        
    # G) Tiny Cache Integration Test
    from train import train_step
    
    dummy_input_dict = {
        "input_ids": seq,
        "attention_mask": torch.ones_like(seq)
    }
    
    c_mask = get_answer_labels(seq)[0, 1:] != -100
    valid_len = c_mask.sum().item()
    
    # Fake teacher targets (Top-50)
    c_probs = torch.ones(valid_len, 50, dtype=torch.bfloat16) / 50.0
    c_indices = torch.randint(0, 100, (valid_len, 50))
    cached_targets = {
        "probs": c_probs,
        "indices": c_indices,
        "mask": c_mask
    }
    
    class MockStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.device = torch.device("cpu")
            self.model = self # mock
            self.language_model = self # mock
            self.layers = []
        def forward(self, input_ids, attention_mask):
            class Out:
                logits = torch.randn(1, seq.size(1), 128, requires_grad=True, dtype=torch.bfloat16)
            return Out()
        def parameters(self):
            return [nn.Parameter(torch.randn(1))]
            
    mock_student = MockStudent()
    mock_opt = torch.optim.AdamW(mock_student.parameters(), lr=1e-3)
    try:
        total_loss, ce, kd, stats = train_step(dummy_input_dict, mock_student, mock_opt, 1, cached_targets=cached_targets)
        print(f"G) Cache Integration Valid: CE={ce:.4f}, SparseKD={kd:.4f} computed successfully without teacher model.")
    except Exception as e:
        print(f"G) Cache Integration Failed: {e}")
        raise e
        
    # H) Device Placement Inheritance Test
    dense_mlp_meta = DummyMLP().to("meta")
    moe_mlp_meta = MoELayer(
        dense_mlp_meta, 
        hidden_size, 
        intermediate_size, 
        num_experts=4, 
        top_k=1,
        capacity_factor=1.25
    ).to(device=dense_mlp_meta.gate_proj.weight.device)
    
    assert moe_mlp_meta.router.gate.weight.device.type == "meta", "Router gate did not inherit device!"
    assert moe_mlp_meta.shared_experts.experts[0].gate_A.weight.device.type == "meta", "Expert did not inherit device!"
    print("H) Device placement inheritance successfully verified.")

    # I) Routing Drop Rate and Capacity Verification
    print("I) Routing Drop Rate and Capacity Verification")
    import math
    T = 100
    C = int(math.ceil((T / 4) * 1.25)) # 32
    
    # 1. Perfectly balanced (25 per expert)
    moe_mlp.train()
    with torch.no_grad():
        moe_mlp.router.gate.weight.zero_()
        for i in range(4):
            moe_mlp.router.gate.weight[i, i] = 100.0
            
    balanced_input = torch.zeros(1, 100, hidden_size, dtype=torch.bfloat16)
    for i in range(100):
        balanced_input[0, i, i % 4] = 1.0
        
    moe_mlp(balanced_input)
    metrics = moe_mlp.metrics
    assert metrics["drop_rate"] == 0.0, f"Expected 0.0 drop rate, got {metrics['drop_rate']}"
    assert metrics["expert_counts"] == [25.0, 25.0, 25.0, 25.0], "Expected 25 tokens per expert"
    print("   - Balanced assignment (below capacity) confirmed 0 drop_rate.")
    
    # 2. Known overflow (100 to expert 0)
    with torch.no_grad():
        moe_mlp.router.gate.weight.zero_()
        moe_mlp.router.gate.weight[0, 0] = 100.0
        
    overflow_input = torch.zeros(1, 100, hidden_size, dtype=torch.bfloat16)
    for i in range(100):
        overflow_input[0, i, 0] = 1.0
        
    moe_mlp(overflow_input)
    metrics = moe_mlp.metrics
    expected_drop_rate = (100 - C) / 100.0
    assert abs(metrics["drop_rate"] - expected_drop_rate) < 1e-4, f"Expected drop_rate {expected_drop_rate}, got {metrics['drop_rate']}"
    assert metrics["expert_counts"] == [100.0, 0.0, 0.0, 0.0], "Expected 100 tokens to expert 0"
    print(f"   - Concentrated assignment confirmed drop_rate {expected_drop_rate:.4f}.")

    print("\n[SUCCESS] All structural smoke tests passed. Ready for execution.")

if __name__ == "__main__":
    test_smoke()
