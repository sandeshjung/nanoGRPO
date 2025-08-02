import torch
import torch.nn as nn 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from configs.configs import ModelConfig


class GRPOModelWrapper:

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.ref_model = None
        self.generation_config = None

    def load_model_and_tokenizer(self):
        print(f"Loading Model: {self.config.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            revision = self.config.model_revision,
            trust_remote_code = self.config.trust_remote_code,
            padding_side = "left" 
        )

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision = self.config.model_revision,
            trust_remote_code = self.config.trust_remote_code,
            torch_dtype = self.config.torch_dtype,
            device_map = self.config.device_map,
            attn_implementation = "flash_attention_2" if torch.cuda.is_available() else None,
        )

        # Resize token embeddings if needed
        if len(self.tokenizer) != self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

        print(f"Model Loaded: {self.model.config.model_type}")
        print(f"Model Parameters: {self.model.num_parameters():,}")

        return self.model, self.tokenizer


def test_model_loading():

    from configs.configs import get_config_for_debugging

    config = get_config_for_debugging()

    model_wrapper = GRPOModelWrapper(config.model)

    model, tokenizer = model_wrapper.load_model_and_tokenizer()

    print(f"Model type: {model}")


if __name__ == "__main__":
    test_model_loading()