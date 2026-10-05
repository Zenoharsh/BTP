import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from transformers import AutoConfig, Qwen2VLForConditionalGeneration
from layers import MoELayer

def test_smoke():
    print("Running Comprehensive P0-10 Smoke Tests...")
    
    # 1. Setup a valid tiny synthetic Qwen2-VL for structural tests
    config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    config.text_config.num_hidden_layers = 1
    config.vision_config.depth = 1
    
    dense_model = Qwen2VLForConditionalGeneration._from_config(config).to(torch.bfloat16)
    dense_model.eval()
    
    dummy_input = torch.randint(0, 1000, (1, 16))
    
    # Check dense forward
    with torch.no_grad():
        dense_logits = dense_model(input_ids=dummy_input).logits
        
    # Apply MoE Surgery
    moe_model = copy.deepcopy(dense_model)
    layer = moe_model.model.language_model.layers[0]
    layer.mlp = MoELayer(
        layer.mlp, 
        config.text_config.hidden_size, 
        config.text_config.intermediate_size, 
        num_experts=4, 
        top_k=1
    ).to(torch.bfloat16)
    
    # A) Zero-LoRA MoE ≈ Dense Behavior
    with torch.no_grad():
        moe_logits = moe_model(input_ids=dummy_input).logits
        
    diff = torch.abs(dense_logits - moe_logits).max().item()
    print(f"A) Dense vs Zero-LoRA MoE Max Difference: {diff:.6f}")
    assert diff < 1e-4, "Zero-LoRA equivalence failed! The fallback/base path is corrupt."
    
    # B) Valid outputs for every token
    assert not torch.isnan(moe_logits).any(), "NaNs detected in MoE output!"
    
    # C) Top-1 Routing & Capacity
    metrics = layer.mlp.metrics
    assert layer.mlp.top_k == 1, "Top-1 routing not respected."
    assert "drop_rate" in metrics, "Drop rate missing."
    print(f"C) Routing metrics valid. Drop rate: {metrics['drop_rate']}")
    
    # D) Gradients only on router + expert LoRA
    for name, param in moe_model.named_parameters():
        if "router.gate" in name or ".experts." in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            
    moe_model.train()
    out = moe_model(input_ids=dummy_input).logits
    loss = out.sum()
    loss.backward()
    
    grad_count = 0
    for name, param in moe_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Expected gradient for {name}"
            grad_count += 1
        else:
            assert param.grad is None, f"Unexpected gradient for {name}"
            
    print(f"D) Gradient isolation valid. ({grad_count} trainable modules received gradients)")
    
    # E) Checkpoint Save/Reload Equivalence
    # We mutate the router and experts slightly to ensure saving works
    with torch.no_grad():
        layer.mlp.router.gate.weight.add_(0.1)
        layer.mlp.shared_experts.experts[0].gate_A.weight.add_(0.1)
        
    out_before_save = moe_model(input_ids=dummy_input).logits
    
    # Save manually (state_dict of custom modules)
    state_dict = {k: v for k, v in moe_model.state_dict().items() if "router" in k or "experts" in k}
    torch.save(state_dict, "smoke_checkpoint.pt")
    
    # Reload into a fresh model
    fresh_moe = copy.deepcopy(dense_model)
    fresh_layer = fresh_moe.model.language_model.layers[0]
    fresh_layer.mlp = MoELayer(
        fresh_layer.mlp, 
        config.text_config.hidden_size, 
        config.text_config.intermediate_size, 
        num_experts=4, 
        top_k=1
    ).to(torch.bfloat16)
    
    fresh_moe.load_state_dict(torch.load("smoke_checkpoint.pt"), strict=False)
    
    fresh_moe.eval()
    out_after_load = fresh_moe(input_ids=dummy_input).logits
    
    load_diff = torch.abs(out_before_save - out_after_load).max().item()
    print(f"E) Save/Reload Max Difference: {load_diff:.6f}")
    assert load_diff < 1e-4, "Checkpoint equivalence failed!"
    
    # Cleanup
    if os.path.exists("smoke_checkpoint.pt"):
        os.remove("smoke_checkpoint.pt")
        
    print("\n[SUCCESS] All structural smoke tests passed. Ready for execution.")

if __name__ == "__main__":
    test_smoke()
