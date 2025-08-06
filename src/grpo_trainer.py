import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import get_scheduler
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import math
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from configs.configs import Config, GRPOConfig
from src.model import GRPOModelWrapper

@dataclass
class GRPOBatch:

    prompts: List[str]
    completions: List[List[str]]
    prompt_input_ids: torch.Tensor
    completion_input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_lengths: List[int]
    rewards: Optional[torch.Tensor] = None

class GRPOTrainer:

    def __init__(self, config, model_wrapper):
        self.config = config
        self.grpo_config = config.grpo
        self.model_wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.ref_model = model_wrapper.ref_model

        self.global_step = 0 
        self.epoch = 0
        self.best_eval_reward = -float('inf')

        self.baseline = 0.0

        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()

    def _setup_optimizer(self):

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        estimated_steps = 1000

        self.scheduler = get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.warmup_steps,
            num_training_steps=estimated_steps
        )

        print(f"Optimizer setup: AdamW with {len(trainable_params)} trainable parameters")

    def generate_completions_for_batch(self, prompts):

        generation_result = self.model_wrapper.generate_completions(
            prompts=prompts,
            num_completions=self.grpo_config.group_size,
            max_new_tokens=self.config.data.max_length - self.config.data.max_prompt_length,
            temperature=self.config.data.temperature,
            top_p=self.config.data.top_p
        )

        completions = generation_result["completions"]
        prompt_lengths = generation_result["prompt_lengths"]

        all_completion_texts = []
        all_input_ids = []
        all_attention_masks = []

        for prompt, prompt_completions in zip(prompts. completions):
            for completion in prompt_completions:
                full_text = prompt + completion
                all_completion_texts.append(full_text)

                encoded = self.tokenizer(
                    full_text,
                    max_length = self.config.data.max_length,
                    padding = False,
                    truncation = True,
                    return_tensors = "pt"
                )

                all_input_ids.append(encoded.input_ids.squeeze(0))
                all_attention_masks.append(encoded.attention_mask.squeeze(0))

        max_len = max(len(ids) for ids in all_input_ids)
        padded_input_ids = []
        padded_attention_masks = []

        for ids, mask in zip(all_input_ids, all_attention_masks):
            pad_len = max_len - len(ids)
            padded_input_ids.append(
                F.pad(ids, (0, pad_len), value=self.tokenizer.pad_token_id)
            )
            padded_attention_masks.append(
                F.pad(mask, (0, pad_len), value=0)
            )

        completion_input_ids = torch.stack(padded_input_ids).to(self.model.device)
        attention_mask = torch.stack(padded_attention_masks).to(self.model.device)

        prompt_encoded = self.tokenizer(
            prompts, 
            padding=True,
            truncation=True,
            max_length=self.config.data.max_prompt_length,
            return_tensors="pt"
        )

        return GRPOBatch(
            prompts=prompts,
            completions=completions,
            prompt_input_ids=prompt_encoded.input_ids.to(self.model.device),
            completion_input_ids=completion_input_ids,
            attention_mask=attention_mask,
            prompt_lengths=prompt_lengths,
        )

def test_grpo_trainer():
    """Test function for GRPO trainer."""
    from configs.configs import get_config_for_debugging
    from src.model import GRPOModelWrapper

    config = get_config_for_debugging()

    model_wrapper = GRPOModelWrapper(config.model)
    model, tokenizer = model_wrapper.load_model_and_tokenizer()
    model = model_wrapper.setup_peft()
    ref_model = model_wrapper.setup_reference_model()

    trainer = GRPOTrainer(config, model_wrapper)

    test_prompts = [
        "Question: What is 2 + 2?\n\nLet me solve this step by step.\n\n",
        "Question: What is 5 * 3?\n\nLet me solve this step by step.\n\n"
    ]

    batch = trainer.generate_completions_for_batch(test_prompts)

    print("GRPO trainer test results:")
    print(f"Generated batch size: {len(batch.prompts)}")
    print(f"Completions shape: {len(batch.completions)} x {len(batch.completions[0])}")

if __name__ == "__main__":
    test_grpo_trainer()
        