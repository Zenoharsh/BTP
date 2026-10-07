import torch
from transformers import Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from layers import MoELayer

def get_mlp_layers(model):
    """P0-6: Consistent layer traversal helper"""
    return model.model.language_model.layers

def build_model(config, processor):
    print("Loading Model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        config.model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    
    if config.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("Performing Architecture Surgery (Sparse Upcycling)...")
    hidden_size = model.config.text_config.hidden_size
    intermediate_size = model.config.text_config.intermediate_size
    
    layers = get_mlp_layers(model)
    for i, layer in enumerate(layers):
        original_mlp = layer.mlp
        target_device = original_mlp.down_proj.weight.device
        
        moe_layer = MoELayer(
            base_mlp=original_mlp, 
            hidden_size=hidden_size, 
            intermediate_size=intermediate_size, 
            num_experts=config.moe.num_experts, 
            rank=config.moe.rank, 
            alpha=config.moe.alpha,
            top_k=config.moe.top_k, 
            capacity_factor=config.moe.capacity_factor, 
            route_tokens=config.moe.route_tokens,
            lb_coef=config.moe.lb_coef, 
            z_coef=config.moe.z_coef
        )
        layer.mlp = moe_layer.to(dtype=torch.bfloat16, device=target_device)
        
    print("Unfreezing Routers and Custom Expert LoRA...")
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if "router.gate" in name or ".experts." in name or "lora_" in name:
            param.requires_grad = True
            if "router.gate" in name:
                param.data = param.data.to(torch.float32)
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name} ({param.numel()})")
    print(f"Total trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}")
    
    # Resolve token IDs
    unk_id = processor.tokenizer.unk_token_id
    
    image_pad_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    assert image_pad_id is not None and image_pad_id != unk_id, "Failed to resolve <|image_pad|> ID"
    
    im_start_id = processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
    assistant_id = processor.tokenizer.convert_tokens_to_ids('assistant')
    im_end_id = processor.tokenizer.convert_tokens_to_ids('<|im_end|>')
    
    # Qwen uses the string 'assistant' without special tokens for the role marker itself
    # but 'assistant' might map to UNK if it's not a single token. Let's fallback to encode if needed.
    if assistant_id is None or assistant_id == unk_id:
        assistant_id = processor.tokenizer.encode('assistant', add_special_tokens=False)[0]
        
    token_ids = {
        'image_pad': image_pad_id,
        'im_start': im_start_id,
        'assistant': assistant_id,
        'im_end': im_end_id
    }
    
    return model, token_ids
