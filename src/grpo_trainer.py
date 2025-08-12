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

        for prompt, prompt_completions in zip(prompts, completions):
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
    
    def compute_rewards(self, batch):
        batch_size = len(batch.prompts)
        group_size = self.grpo_config.group_size
        
        # In practice, this would be a proper reward model or evaluation metric
        rewards = []
        
        for i in range(batch_size):
            group_rewards = []
            for j in range(group_size):
                completion = batch.completions[i][j]
                
                # Simple reward: length-based 
                reward = len(completion.split()) / 50.0  # Normalize by average length
                
                # Add some randomness for demonstration
                reward += np.random.normal(0, 0.1)
                group_rewards.append(reward)
            
            rewards.append(group_rewards)
        
        return torch.tensor(rewards, dtype=torch.float32, device=self.model.device)
    
    def compute_advantages(self, rewards):
        batch_size, group_size = rewards.shape
        
        if self.grpo_config.advantage_normalization == "group":
            # Group-wise advantage: reward - mean(group_rewards)
            group_means = rewards.mean(dim=1, keepdim=True)  # [batch_size, 1]
            advantages = rewards - group_means
            
        elif self.grpo_config.advantage_normalization == "batch":
            # Batch-wise advantage: reward - mean(all_rewards)
            batch_mean = rewards.mean()
            advantages = rewards - batch_mean
            
        elif self.grpo_config.advantage_normalization == "none":
            # Use baseline
            if self.grpo_config.use_baseline:
                advantages = rewards - self.baseline
            else:
                advantages = rewards
        else:
            raise ValueError(f"Unknown advantage normalization: {self.grpo_config.advantage_normalization}")
        
        # Update baseline with exponential moving average
        if self.grpo_config.use_baseline:
            current_mean_reward = rewards.mean().item()
            self.baseline = (
                self.grpo_config.baseline_decay * self.baseline + 
                (1 - self.grpo_config.baseline_decay) * current_mean_reward
            )
        
        return advantages
    
    def compute_grpo_loss(self, batch):
        self.model.train()
        
        batch_size = len(batch.prompts)
        group_size = self.grpo_config.group_size
        
        # Compute rewards and advantages
        rewards = self.compute_rewards(batch)
        advantages = self.compute_advantages(rewards)
        
        # Flatten [batch_size * group_size, seq_len]
        flat_advantages = advantages.flatten()  # [batch_size * group_size]
        
        # Compute log probabilities in chunks to save memory
        chunk_size = self.grpo_config.chunk_size
        total_sequences = batch.completion_input_ids.shape[0]
        
        all_log_probs = []
        all_kl_divs = []
        
        for start_idx in range(0, total_sequences, chunk_size):
            end_idx = min(start_idx + chunk_size, total_sequences)
            
            # Get chunk data
            chunk_input_ids = batch.completion_input_ids[start_idx:end_idx]
            chunk_attention_mask = batch.attention_mask[start_idx:end_idx]
            
            # Compute prompt lengths for this chunk
            chunk_prompt_lengths = []
            for i in range(start_idx, end_idx):
                batch_idx = i // group_size
                chunk_prompt_lengths.append(batch.prompt_lengths[batch_idx])
            
            # Forward pass through model
            with torch.amp.autocast(enabled=self.grpo_config.use_mixed_precision, device_type='cpu'):
                outputs = self.model(
                    input_ids=chunk_input_ids,
                    attention_mask=chunk_attention_mask,
                    return_dict=True,
                )
                
                # Compute log probabilities for completion tokens only
                logits = outputs.logits[:, :-1, :]  # [chunk_size, seq_len-1, vocab_size]
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Get target tokens
                target_ids = chunk_input_ids[:, 1:]  # [chunk_size, seq_len-1]
                
                # Gather log probs for actual tokens
                token_log_probs = torch.gather(
                    log_probs, 
                    dim=-1, 
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)  # [chunk_size, seq_len-1]
                
                # Mask to completion tokens only
                completion_mask = torch.zeros_like(token_log_probs)
                for i, prompt_len in enumerate(chunk_prompt_lengths):
                    if prompt_len - 1 < completion_mask.shape[1]:
                        completion_mask[i, prompt_len-1:] = chunk_attention_mask[i, 1:][prompt_len-1:]
                
                # Sum log probs over completion tokens
                sequence_log_probs = (token_log_probs * completion_mask).sum(dim=-1)
                all_log_probs.append(sequence_log_probs)
                
                # Compute KL divergence if reference model exists
                if self.ref_model is not None:
                    kl_div = self.model_wrapper.compute_kl_divergence(
                        chunk_input_ids, 
                        chunk_attention_mask, 
                        chunk_prompt_lengths
                    )
                    all_kl_divs.append(kl_div)
        
        # Concatenate results from all chunks
        log_probs = torch.cat(all_log_probs, dim=0)  # [batch_size * group_size]
        
        if all_kl_divs:
            kl_divs = torch.cat(all_kl_divs, dim=0)  # [batch_size * group_size]
        else:
            kl_divs = torch.zeros_like(log_probs)
        
        # Compute GRPO loss
        # Policy gradient loss: -log_prob * advantage
        pg_loss = -(log_probs * flat_advantages).mean()
        
        # KL penalty
        kl_loss = self.grpo_config.beta * kl_divs.mean()
        
        # Total loss
        total_loss = pg_loss + kl_loss
        
        return {
            "loss": total_loss,
            "pg_loss": pg_loss,
            "kl_loss": kl_loss,
            "mean_reward": rewards.mean(),
            "mean_advantage": advantages.mean(),
            "kl_div": kl_divs.mean(),
        }
    
    def training_step(self, batch):
        # Compute loss
        loss_dict = self.compute_grpo_loss(batch)
        loss = loss_dict["loss"]
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if self.config.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.training.max_grad_norm
            )
        
        # Optimizer step
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        # Update global step
        self.global_step += 1
        
        # Convert tensors to float for logging
        metrics = {k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()}
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        metrics["baseline"] = self.baseline
        
        return metrics
    
    def evaluate(self, eval_dataloader):
        self.model.eval()
        
        total_reward = 0.0
        total_samples = 0
        all_rewards = []
        
        with torch.no_grad():
            for batch_data in eval_dataloader:
                # Convert dataloader batch to GRPOBatch
                prompts = batch_data["prompts"]
                
                # Generate completions for evaluation
                grpo_batch = self.generate_completions_for_batch(prompts)
                
                # Compute rewards
                rewards = self.compute_rewards(grpo_batch)
                
                # Accumulate metrics
                batch_reward = rewards.mean().item()
                total_reward += batch_reward * len(prompts)
                total_samples += len(prompts)
                all_rewards.extend(rewards.flatten().cpu().tolist())
        
        # Compute final metrics
        avg_reward = total_reward / total_samples if total_samples > 0 else 0.0
        
        eval_metrics = {
            "eval_reward": avg_reward,
            "eval_reward_std": np.std(all_rewards) if all_rewards else 0.0,
            "eval_samples": total_samples,
        }
        
        # Update best reward
        if avg_reward > self.best_eval_reward:
            self.best_eval_reward = avg_reward
            eval_metrics["new_best"] = True
        else:
            eval_metrics["new_best"] = False
        
        return eval_metrics
    
    def save_checkpoint(self, save_path: str, metrics: Optional[Dict] = None):

        checkpoint = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_reward": self.best_eval_reward,
            "baseline": self.baseline,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
        }
        
        if metrics:
            checkpoint["metrics"] = metrics
        
        torch.save(checkpoint, f"{save_path}/training_state.pt")
        self.model_wrapper.save_model(save_path)
        
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint(self, load_path: str):
        checkpoint_path = f"{load_path}/training_state.pt"
        
        if not torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
        else:
            checkpoint = torch.load(checkpoint_path)
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_reward = checkpoint["best_eval_reward"]
        self.baseline = checkpoint["baseline"]
        
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.model_wrapper.load_model(load_path)
        
        print(f"Checkpoint loaded from {load_path}")
        print(f"Resumed at step {self.global_step}, epoch {self.epoch}")

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

    loss_dict = trainer.compute_grpo_loss(batch)

    print("GRPO trainer test results:")
    print(f"Generated batch size: {len(batch.prompts)}")
    print(f"Completions shape: {len(batch.completions)} x {len(batch.completions[0])}")
    print(f"Loss: {loss_dict['loss'].item():.4f}")
    print(f"PG Loss: {loss_dict['pg_loss'].item():.4f}")
    print(f"KL Loss: {loss_dict['kl_loss'].item():.4f}")
    print(f"Mean reward: {loss_dict['mean_reward'].item():.4f}")
    print(f"Sample completion: {batch.completions[0][0][:100]}...")

if __name__ == "__main__":
    test_grpo_trainer()
        