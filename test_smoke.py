import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration
from layers import MoELayer
import copy

def test_smoke():
    print("Running P0-10 Smoke Tests...")
    
    # Tiny dummy model for testing
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    config.text_config.num_hidden_layers = 1
    config.text_config.hidden_size = 128
    config.text_config.intermediate_size = 256
    config.vision_config = None # Disable vision for raw text structural test
    
    # 1. Dense checkpoint integrity
    dense_model = Qwen2VLForConditionalGeneration._from_config(config)
    dense_model.eval()
    
    dummy_input = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        dense_logits = dense_model(input_ids=dummy_input).logits
        
    # 2. Dense-to-zero-LoRA MoE equivalence
    moe_model = copy.deepcopy(dense_model)
    hidden_size = config.text_config.hidden_size
    intermediate_size = config.text_config.intermediate_size
    
    layer = moe_model.model.language_model.layers[0]
    layer.mlp = MoELayer(layer.mlp, hidden_size, intermediate_size, num_experts=4, top_k=1)
    
    with torch.no_grad():
        moe_logits = moe_model(input_ids=dummy_input).logits
        
    diff = torch.abs(dense_logits - moe_logits).max().item()
    print(f"Dense vs Zero-LoRA MoE Max Difference: {diff}")
    assert diff < 1e-4, "Zero-LoRA equivalence failed! Base path is corrupt."
    
    # 3. Routing correctness
    metrics = layer.mlp.metrics
    assert "top1_confidence" in metrics
    print(f"Routing metrics populated successfully: drop_rate={metrics['drop_rate']}")
    
    print("[SUCCESS] Smoke tests passed. Ready for training.")

if __name__ == "__main__":
    test_smoke()
