import yaml
import sys
import os
from dataclasses import dataclass, field
from typing import Optional, Any

@dataclass
class DataConfig:
    dir: str = "data/v3"
    min_pixels: int = 50176
    max_pixels: int = 401408
    limit: Optional[int] = None
    teacher_cache: str = "teacher_cache/v3"
    dev_limit: Optional[int] = None     # first N dev samples PER TASK for per-epoch eval

@dataclass
class MoEConfig:
    num_experts: int = 4
    rank: int = 16
    alpha: int = 32
    top_k: int = 1
    capacity_factor: Optional[float] = None
    route_tokens: str = "all"
    lb_coef: float = 0.01
    z_coef: float = 0.001

@dataclass
class LossConfig:
    ce: float = 1.0
    kd: float = 1.0
    kd_temperature: float = 2.0

@dataclass
class TrainConfig:
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 8
    lr: float = 2.0e-4
    router_lr: float = 1.0e-3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.05
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    log_every: int = 1
    eval_every_epoch: bool = True
    eval_every: int = 1                 # evaluate on dev every N epochs (and always after the last)
    num_workers: int = 2
    gradient_checkpointing: bool = True

@dataclass
class Config:
    model_id: str = "Qwen/Qwen2-VL-2B-Instruct"
    data: DataConfig = field(default_factory=DataConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

def _merge_dicts(base, update):
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            _merge_dicts(base[k], v)
        else:
            base[k] = v

def load(yaml_path: str = None) -> Config:
    base_path = os.path.join(os.path.dirname(__file__), "configs", "base.yaml")
    
    with open(base_path, 'r') as f:
        base_dict = yaml.safe_load(f) or {}
        
    if yaml_path:
        with open(yaml_path, 'r') as f:
            update_dict = yaml.safe_load(f) or {}
        _merge_dicts(base_dict, update_dict)
        
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            if args[i] == "--set" and i + 1 < len(args):
                key_val = args[i+1]
                if "=" in key_val:
                    key_path, val_str = key_val.split("=", 1)
                    
                    try:
                        val = yaml.safe_load(val_str)
                    except:
                        val = val_str
                        
                    parts = key_path.split(".")
                    d = base_dict
                    for p in parts[:-1]:
                        if p not in d:
                            d[p] = {}
                        d = d[p]
                    d[parts[-1]] = val
                i += 2
            else:
                i += 1
                
    cfg = Config()
    cfg.model_id = base_dict.get("model_id", cfg.model_id)
    if "data" in base_dict:
        for k, v in base_dict["data"].items():
            setattr(cfg.data, k, v)
    if "moe" in base_dict:
        for k, v in base_dict["moe"].items():
            setattr(cfg.moe, k, v)
    if "loss" in base_dict:
        for k, v in base_dict["loss"].items():
            setattr(cfg.loss, k, v)
    if "train" in base_dict:
        for k, v in base_dict["train"].items():
            setattr(cfg.train, k, v)
            
    return cfg
