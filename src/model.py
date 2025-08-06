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

    def setup_peft(self):
        if not self.config.use_peft:
            return self.model

        print("Setting up PEFT (LoRA) ...")

        if hasattr(self.model, 'is_loaded_in_8bit') or hasattr(self.model, 'is_loaded_in_4bit'):
            self.model = prepare_model_for_kbit_training(self.model)

        # create lora config
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r = self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none"
        )

        # Apply peft
        self.model = get_peft_model(self.model, peft_config) # type: ignore

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())

        print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

        return self.model

    def setup_reference_model(self):
        print("Setting up reference model ...")

        # load reference model (same as base but frozen)
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            revision = self.config.model_revision,
            trust_remote_code = self.config.trust_remote_code,
            torch_dtype = self.config.torch_dtype,
            device_map = self.config.device_map,
        )

        # freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False

        self.ref_model.eval()

        print("Reference model setup complete")
        
        return self.ref_model

    def generate_completions(self, prompts, num_completions = 4, max_new_tokens = 256, temperature = 0.8, top_p = 0.9):

        self.model.eval() # type: ignore

        all_completions = []
        all_prompt_lengths = []

        with torch.no_grad():
            for prompt in prompts:
                prompt_inputs = self.tokenizer(
                    prompt,
                    return_tensors = "pt",
                    padding = False,
                    truncation = False
                ).to(self.model.device) # type: ignore

                prompt_length = prompt_inputs.input_ids.shape[-1]
                all_prompt_lengths.append(prompt_length)

                # generate multiple
                completions_for_prompt = []

                for _ in range(num_completions):
                    with torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
                        outputs = self.model.generate(
                            **prompt_inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            repetition_penalty=1.1
                        )

                    generate_ids = outputs[0][prompt_length:]
                    completion = self.tokenizer.decode(generate_ids, skip_special_tokens=True)
                    completions_for_prompt.append(completion)

                all_completions.append(completions_for_prompt)

        return {
            "completions": all_completions,
            "prompt_lengths": all_prompt_lengths,
        }

    def compute_log_probabilities(self, input_ids, attention_mask, prompt_lengths = None):

        self.model.eval() # type: ignore
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ) # type: ignore
            
            # Get log probabilities
            logits = outputs.logits  # [batch_size, seq_len, vocab_size]
            log_probs = torch.log_softmax(logits, dim=-1)  # [batch_size, seq_len, vocab_size]
            
            # Get log probs for actual tokens (shift by 1)
            target_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
            log_probs = log_probs[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            
            # Gather log probs for target tokens
            token_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch_size, seq_len-1]
            
            # Apply attention mask (shifted)
            attention_mask_shifted = attention_mask[:, 1:]  # [batch_size, seq_len-1]
            token_log_probs = token_log_probs * attention_mask_shifted
            
            # If prompt lengths provided, only compute log probs for completions
            if prompt_lengths is not None:
                masked_log_probs = torch.zeros_like(token_log_probs)
                for i, prompt_len in enumerate(prompt_lengths):
                    # Only keep log probs after prompt (prompt_len - 1 because of shift)
                    if prompt_len - 1 < token_log_probs.shape[1]:
                        masked_log_probs[i, prompt_len-1:] = token_log_probs[i, prompt_len-1:]
                token_log_probs = masked_log_probs
            
            return token_log_probs

    def compute_kl_divergence(self, input_ids, attention_mask, prompt_lengths = None):
        
        if self.ref_model is None:
            return torch.zeros(input_ids.shape[0], device=input_ids.device)
        
        # Get log probs from current model
        current_log_probs = self.compute_log_probabilities(
            input_ids, attention_mask, prompt_lengths
        )
        
        # Get log probs from reference model
        with torch.no_grad():
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            
            ref_logits = ref_outputs.logits[:, :-1, :]  # [batch_size, seq_len-1, vocab_size]
            ref_log_probs = torch.log_softmax(ref_logits, dim=-1)
            
            target_ids = input_ids[:, 1:]  # [batch_size, seq_len-1]
            ref_token_log_probs = torch.gather(
                ref_log_probs,
                dim=-1,
                index=target_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            # Apply attention mask
            attention_mask_shifted = attention_mask[:, 1:]
            ref_token_log_probs = ref_token_log_probs * attention_mask_shifted
            
            # Mask to completion only if prompt lengths provided
            if prompt_lengths is not None:
                masked_ref_log_probs = torch.zeros_like(ref_token_log_probs)
                for i, prompt_len in enumerate(prompt_lengths):
                    if prompt_len - 1 < ref_token_log_probs.shape[1]:
                        masked_ref_log_probs[i, prompt_len-1:] = ref_token_log_probs[i, prompt_len-1:]
                ref_token_log_probs = masked_ref_log_probs
        
        # Compute KL divergence: KL(current || ref) = current_log_prob - ref_log_prob
        kl_div = current_log_probs - ref_token_log_probs
        
        # Sum over sequence length (only where attention mask is 1)
        attention_mask_shifted = attention_mask[:, 1:]
        if prompt_lengths is not None:
            # Only sum over completion tokens
            completion_mask = torch.zeros_like(attention_mask_shifted)
            for i, prompt_len in enumerate(prompt_lengths):
                if prompt_len - 1 < completion_mask.shape[1]:
                    completion_mask[i, prompt_len-1:] = attention_mask_shifted[i, prompt_len-1:]
            kl_div = (kl_div * completion_mask).sum(dim=-1)
        else:
            kl_div = (kl_div * attention_mask_shifted).sum(dim=-1)
        
        return kl_div

    def save_model(self, save_path):
        """Save the model"""
        if self.config.use_peft:
            self.model.save_pretrained(save_path)
        else:
            self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"✅ Model saved to {save_path}")
    
    def load_model(self, load_path):
        """Load the model"""
        if self.config.use_peft:
            self.model = PeftModel.from_pretrained(self.model, load_path)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(load_path)
        print(f"✅ Model loaded from {load_path}")

def test_model_loading():

    from configs.configs import get_config_for_debugging

    config = get_config_for_debugging()

    model_wrapper = GRPOModelWrapper(config.model)

    model, tokenizer = model_wrapper.load_model_and_tokenizer()

    model = model_wrapper.setup_peft()

    ref_model = model_wrapper.setup_reference_model()

    test_prompts = ["Question: What is 2 + 2?\n\nLet me solve this step by step.\n\n"]
    
    completions = model_wrapper.generate_completions(
        test_prompts, 
        num_completions=2,
        max_new_tokens=50
    )

    print("Model loading results:")
    print(f"Model type: {model.config.model_type}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Generated completions: {len(completions['completions'][0])}")
    print(f"Sample completion: {completions['completions'][0][0][:100]}...")

if __name__ == "__main__":
    test_model_loading()