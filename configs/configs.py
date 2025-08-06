from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch

@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_revision: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: torch.dtype = torch.float16
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
class GRPOConfig:
    """GRPO algorithm configuration"""
    beta: float = 0.1  # KL penalty coefficient
    group_size: int = 4  # Number of completions per prompt
    
    use_baseline: bool = True
    baseline_decay: float = 0.95
    advantage_normalization: str = "group"  # "group", "batch", or "none"
    
    # Trust region
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    
    # Memory optimization
    chunk_size: int = 64 
    gradient_checkpointing: bool = True
    use_mixed_precision: bool = True

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Basic training parameters
    num_epochs: int = 3
    per_device_train_batch_size: int = 1  # Small due to memory constraints
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-6
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    
    # Scheduler
    lr_scheduler_type: str = "cosine"
    
    # Evaluation and logging
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 10
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Output directories
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    # Resuming
    resume_from_checkpoint: Optional[str] = None
    
    # Optimization flags
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 0  # 0 for debugging, increase for production
    remove_unused_columns: bool = False
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True  # Better for modern GPUs
    
    # Memory optimization
    max_memory_per_gpu: Optional[str] = None  # e.g., "10GB"

@dataclass
class Config:
    """Main configuration class combining all configs"""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
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