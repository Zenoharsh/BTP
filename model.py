"""
Model construction for BTP v3 (task T4).

build_model(cfg, processor) -> (model, token_ids)
    Loads Qwen2-VL-2B in NF4, replaces every language MLP with MoELayer, freezes everything
    except LoRA/router params, installs the image-token hook (needed for route_tokens="text"),
    and enables non-reentrant gradient checkpointing.

Rules (see ANTIGRAVITY_INSTRUCTIONS_v3.md):
- NEVER call .to(dtype) on a module that contains bitsandbytes 4-bit layers. Only the new
  LoRA/router modules are moved (device only; they stay float32 by design).
- Compute dtype is float16: T4 and GTX 1650 (Turing) have no native bf16.
- Parameter percentages use the UNQUANTIZED parameter count (numel of packed 4-bit tensors undercounts).
"""
import os
import json
import subprocess
import torch
from layers import MoELayer, get_lm_layers, install_token_type_hook

def count_unquantized_params(model_id):
    """Exact parameter count of the unquantized model, built on the meta device (no download of weights)."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, Qwen2VLForConditionalGeneration
    with init_empty_weights():
        m = Qwen2VLForConditionalGeneration._from_config(AutoConfig.from_pretrained(model_id))
    return sum(p.numel() for p in m.parameters())


def resolve_token_ids(processor):
    tok = processor.tokenizer
    unk = tok.unk_token_id
    ids = {
        "image_pad": tok.convert_tokens_to_ids("<|image_pad|>"),
        "im_start": tok.convert_tokens_to_ids("<|im_start|>"),
        "im_end": tok.convert_tokens_to_ids("<|im_end|>"),
    }
    a = tok.encode("assistant", add_special_tokens=False)
    assert len(a) == 1, f"'assistant' is not a single token: {a}"
    ids["assistant"] = a[0]
    for k, v in ids.items():
        assert v is not None and v != unk, f"Could not resolve token id for {k}"
    return ids


def apply_moe_surgery(model, cfg, image_token_id, im_start_id=None):
    """Replace every language-model MLP with MoELayer. Works on any model exposing
    model.model.(language_model.)layers[i].mlp with gate/up/down_proj."""
    tc = model.config.text_config if hasattr(model.config, "text_config") else model.config
    m = cfg.moe
    for layer in get_lm_layers(model):
        base = layer.mlp
        dev = base.down_proj.weight.device
        new = MoELayer(base, tc.hidden_size, tc.intermediate_size,
                       num_experts=m.num_experts, rank=m.rank, alpha=m.alpha, top_k=m.top_k,
                       capacity_factor=m.capacity_factor, route_tokens=m.route_tokens,
                       lb_coef=m.lb_coef, z_coef=m.z_coef,
                       route_level=getattr(m, "route_level", "token"))
        # move ONLY the new float32 modules; the (possibly 4-bit) base is untouched
        for sub in (new.lora_gate, new.lora_up, new.lora_down, new.router):
            if sub is not None:
                sub.to(dev)
        new.usage_ema = new.usage_ema.to(dev)
        layer.mlp = new

    for n, p in model.named_parameters():
        p.requires_grad = is_trainable_name(n)

    install_token_type_hook(model, image_token_id, im_start_id)
    return model


def is_trainable_name(name):
    return ".lora_" in name or ".router." in name


def trainable_report(model, total_params):
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": n, "total_unquantized": total_params,
            "trainable_pct": 100.0 * n / total_params}


# Module-name patterns for parts of the base that can be kept in fp16 (cfg.skip_quant). Both the
# transformers 5 names (prefix match) and the older 4.x names (substring match) are listed.
SKIP_QUANT_PATTERNS = {
    "vision": ["model.visual", "visual"],
    "language": ["model.language_model", "model.layers"],
}


def bnb_config(skip_quant=()):
    """4-bit NF4 config; parts named in skip_quant ("vision", "language") stay fp16."""
    from transformers import BitsAndBytesConfig
    skip = None
    if skip_quant:
        unknown = set(skip_quant) - set(SKIP_QUANT_PATTERNS)
        if unknown:
            raise ValueError(f"skip_quant: unknown {sorted(unknown)}, choose from {sorted(SKIP_QUANT_PATTERNS)}")
        skip = ["lm_head"] + [p for s in skip_quant for p in SKIP_QUANT_PATTERNS[s]]
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True,
                              bnb_4bit_compute_dtype=torch.float16, llm_int8_skip_modules=skip)


def quant_report(model):
    """Count 4-bit vs plain linear layers in the vision tower and the language model."""
    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or ".lora_" in name or ".router" in name:
            continue
        part = "vision" if "visual" in name else "lm_head" if "lm_head" in name else "language"
        k = "4bit" if type(mod).__name__ == "Linear4bit" else "fp16"
        out.setdefault(part, {"4bit": 0, "fp16": 0})[k] += 1
    return out


def load_hqq_base(cfg, plan_path, device_index=0):
    """fp16 base loaded on CPU, then quantised unit by unit onto the GPU with a mopeq.py plan
    (HQQ mixed precision), so the fp16 model never has to fit in GPU memory."""
    from transformers import Qwen2VLForConditionalGeneration
    from mopeq import apply_plan
    if cfg.skip_quant:
        raise ValueError("mopeq_plan sets every unit's precision; do not combine it with skip_quant")
    model = Qwen2VLForConditionalGeneration.from_pretrained(cfg.model_id, torch_dtype=torch.float16)
    dev = f"cuda:{device_index}" if torch.cuda.is_available() else "cpu"
    with open(plan_path) as f:
        apply_plan(model, json.load(f), device=dev)
    print(f"base quantised with HQQ plan {plan_path}")
    return model


def build_model(cfg, processor, device_index=0):
    from transformers import Qwen2VLForConditionalGeneration
    if cfg.mopeq_plan:
        model = load_hqq_base(cfg, cfg.mopeq_plan, device_index)
    else:
        bnb = bnb_config(cfg.skip_quant)
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            cfg.model_id, quantization_config=bnb, torch_dtype=torch.float16,
            device_map={"": device_index})
    token_ids = resolve_token_ids(processor)
    apply_moe_surgery(model, cfg, token_ids["image_pad"], token_ids["im_start"])

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False          # required with checkpointing during training

    if cfg.skip_quant:
        print(f"skip_quant={cfg.skip_quant} -> linear layers {quant_report(model)}")
    r = trainable_report(model, count_unquantized_params(cfg.model_id))
    print(f"Trainable params: {r['trainable']:,} / {r['total_unquantized']:,} "
          f"(unquantized) = {r['trainable_pct']:.3f}%")
    return model, token_ids


# ------------------------------------------------------------------ adapters I/O
def adapter_state_dict(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if is_trainable_name(n)}


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def save_adapters(model, path, cfg_dict=None, extra=None):
    os.makedirs(path, exist_ok=True)
    torch.save(adapter_state_dict(model), os.path.join(path, "adapters.pt"))
    meta = {"commit": _git_commit(), "config": cfg_dict or {}, **(extra or {})}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_adapters(model, path, strict=True):
    sd = torch.load(os.path.join(path, "adapters.pt"), map_location="cpu")
    params = dict(model.named_parameters())
    missing = [k for k in params if is_trainable_name(k) and k not in sd]
    unexpected = [k for k in sd if k not in params]
    if strict and (missing or unexpected):
        raise RuntimeError(f"Adapter mismatch. missing={missing[:5]} unexpected={unexpected[:5]}")
    with torch.no_grad():
        for k, v in sd.items():
            if k in params:
                params[k].copy_(v.to(params[k].device, params[k].dtype))
    return model