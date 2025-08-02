from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str = "auto"
    
    # PEFT configuration
    use_peft: bool = True
    peft_type: str = "lora"  # "lora" or "adalora"
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass  
class DataConfig:
    """Data configuration"""
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    eval_split: str = "test"
    max_length: int = 512
    max_prompt_length: int = 256
    
    # Data processing
    num_proc: int = 4
    remove_unused_columns: bool = False
    
    # GRPO-specific data settings
    num_completions_per_prompt: int = 4  # Group size for GRPO
    temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.9


@dataclass
class Config:
    """Main configuration class combining all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    seed: int = 42
    local_rank: int = -1

def get_config_for_debugging() -> Config:
    """Get configuration optimized for debugging"""
    config = Config()
    
    # Reduce sizes for faster debugging
    config.data.max_length = 256
    config.data.max_prompt_length = 128
    config.data.num_completions_per_prompt = 2
    
    return config