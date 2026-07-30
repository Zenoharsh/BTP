import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from layers import MoELayer

def setup_models(model_id="Qwen/Qwen1.5-0.5B"):
    print("Loading Teacher Model...")
    # 1. Load Teacher (Frozen)
    teacher = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    print("Loading Student Model in 4-bit...")
    # 2. Load Student (with 4-bit QLoRA config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )
    student = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )

    print("Performing Architecture Surgery...")
    # 3. Architecture Surgery: Replace Dense MLP with MoELayer
    hidden_size = student.config.hidden_size
    for i, layer in enumerate(student.model.layers):
        original_mlp = layer.mlp
        moe_layer = MoELayer(original_mlp, hidden_size=hidden_size, num_experts=4)
        layer.mlp = moe_layer

    print("Applying QLoRA to Experts...")
    # 4. Apply LoRA specifically to experts
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Targeting the linear layers inside our experts. 
        # Using regex to match all nested experts within the replaced mlp block.
        target_modules=[
            ".*experts.*gate_proj.*", 
            ".*experts.*up_proj.*", 
            ".*experts.*down_proj.*"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )
    student = get_peft_model(student, lora_config)

    print("Unfreezing Routers...")
    # 5. Unfreeze Routers and set to fp32
    for name, param in student.named_parameters():
        if "router" in name:
            param.requires_grad = True
            param.data = param.data.to(torch.float32)

    return teacher, student

def train_step(batch, teacher_model, student_model, optimizer, temperature=2.0):
    """
    Executes a single training step using Knowledge Distillation and Auxiliary Loss.
    """
    input_ids = batch['input_ids'].to(student_model.device)
    attention_mask = batch['attention_mask'].to(student_model.device)
    
    optimizer.zero_grad()
    
    # 1. Teacher Forward (Strictly no gradients to save memory and compute)
    with torch.no_grad():
        teacher_outputs = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
        teacher_logits = teacher_outputs.logits
        
    # 2. Student Forward
    student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
    student_logits = student_outputs.logits
    
    # Catch and aggregate the Aux Losses from the custom MoE Layers
    student_aux_loss = 0.0
    for layer in student_model.model.layers:
        if hasattr(layer.mlp, 'latest_aux_loss'):
            student_aux_loss += layer.mlp.latest_aux_loss
            
    # 3. Knowledge Distillation (KD) Loss using KL Divergence
    scaled_student_logits = student_logits / temperature
    scaled_teacher_logits = teacher_logits / temperature
    
    student_log_probs = F.log_softmax(scaled_student_logits, dim=-1)
    teacher_probs = F.softmax(scaled_teacher_logits, dim=-1)
    
    kd_loss = F.kl_div(
        student_log_probs, 
        teacher_probs, 
        reduction="batchmean"
    ) * (temperature ** 2) 
    
    # 4. Total Loss Calculation
    total_loss = kd_loss + student_aux_loss
    
    # Backpropagation
    total_loss.backward()
    optimizer.step()
    
    aux_val = student_aux_loss.item() if isinstance(student_aux_loss, torch.Tensor) else student_aux_loss
    return total_loss.item(), kd_loss.item(), aux_val
