import re
import json
from typing import Dict, List, Optional, Tuple, Any
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer
import torch
from torch.utils.data import DataLoader
import numpy as np 
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from configs.configs import DataConfig

def extract_answer(text):
    """
    Extract the numerical answer from GSM8K solution string
    
    Args:
        text: The text string containing the answer
        
    Returns:
        The extracted numerical answer as string, or None if not found
    """
    patterns =  [
        r"####\s*([0-9,]+(?:\.[0-9]+)?)",  # #### format
        r"The answer is\s*([0-9,]+(?:\.[0-9]+)?)",  # explicit format
        r"\$([0-9,]+(?:\.[0-9]+)?)",  # dollar format
    ]

    for pattern in patterns: 
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")
    return None

def format_gsm8k_prompt(question):
    """
    Format GSM8K question as a prompt
    
    Args:
        question: The raw question from GSM8K
        
    Returns:
        Formatted prompt string
    """
    return f"Question: {question}\n\nLet me solve this step by step.\n\n"

def format_gsm8k_completion(question, answer):
    """
    Format GSM8K question and answer as a complete text
    
    Args:
        question: The question
        answer: The answer
        
    Returns:
        Complete formatted text
    """
    prompt = format_gsm8k_prompt(question)
    return f"{prompt}{answer}"


class GSM8KDataset:
    """GSM8K dataset handler for GRPO training"""
    
    def __init__(self, config: DataConfig, tokenizer: PreTrainedTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.dataset = None
        self.train_dataset = None
        self.eval_dataset = None

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    def load_dataset(self):

        print("🔄 Loading GSM8K dataset...")
        
        dataset = load_dataset(
            self.config.dataset_name,
            self.config.dataset_config,
            trust_remote_code = True
        )

        train_data = dataset[self.config.train_split] # type: ignore
        self.train_dataset = self._preprocess_dataset(train_data, is_training = True) # type: ignore

        eval_data = dataset[self.config.eval_split] # type: ignore
        self.eval_dataset = self._preprocess_dataset(eval_data, is_training=False) # type: ignore
        
        print(f"✅ Loaded {len(self.train_dataset)} training samples")
        print(f"✅ Loaded {len(self.eval_dataset)} evaluation samples")
        
        return self.train_dataset, self.eval_dataset

    def _preprocess_dataset(self, dataset:Dataset, is_training:bool = True):

        def preprocess_function(examples):
            questions = examples["question"]
            answers = examples["answer"]

            prompts = [format_gsm8k_prompt(q) for q in questions]
            completions = [format_gsm8k_completion(q, a) for q, a in zip(questions, answers)]

            numerical_answers = [extract_answer(a) for a in answers]

            prompt_encodings = self.tokenizer(
                prompts,
                truncation=True, 
                max_length=self.config.max_prompt_length, 
                padding=False,
                return_tensors=None,
            )

            completion_encodings = self.tokenizer(
                completions,
                truncation=True,
                max_length=self.config.max_length,
                padding=False,
                return_tensors=None
            )

            return {
                "question": questions,
                "answer": answers,
                "numerical_answer": numerical_answers,
                "prompt": prompts,
                "completion": completions,
                "prompt_input_ids": prompt_encodings["input_ids"],
                "prompt_attention_mask": prompt_encodings["attention_mask"],
                "input_ids": completion_encodings["input_ids"],
                "attention_mask": completion_encodings["attention_mask"],
            }

        processed_dataset = dataset.map(
            preprocess_function,
            batched=True, 
            num_proc=self.config.num_proc,
            remove_columns=dataset.column_names if self.config.remove_unused_columns else [],
            desc="Preprocessing dataset"
        )

        return processed_dataset

    def create_grpo_dataloader(self, dataset: Dataset, batch_size, shuffle=True):

        def collate_fn(batch):

            questions, prompts, completions, numerical_answers = zip(
                *[(item["question"], item["prompt"], item["completion"], item["numerical_answer"]) for item in batch]
            )

            prompt_input_ids, prompt_attention_mask = zip(
                *[(item["prompt_input_ids"], item["prompt_attention_mask"]) for item in batch]
            )

            input_ids, attention_mask = zip(
                *[(item["input_ids"], item["attention_mask"]) for item in batch]
            )

            # Pad sequences
            max_prompt_len = max(len(ids) for ids in prompt_input_ids)
            max_completion_len = max(len(ids) for ids in input_ids)

            # Pad prompts
            padded_prompt_ids = []
            padded_prompt_masks = []
            for ids, mask in zip(prompt_input_ids, prompt_attention_mask):
                pad_len = max_prompt_len - len(ids)
                padded_prompt_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_prompt_masks.append(mask + [0] * pad_len)

            # Pad completions  
            padded_input_ids = []
            padded_attention_masks = []
            for ids, mask in zip(input_ids, attention_mask):
                pad_len = max_completion_len - len(ids)
                padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * pad_len)
                padded_attention_masks.append(mask + [0] * pad_len)

            return {
                "questions": questions,
                "prompts": prompts,
                "completions": completions,
                "numerical_answers": numerical_answers,
                "prompt_input_ids": torch.tensor(padded_prompt_ids, dtype=torch.long),
                "prompt_attention_mask": torch.tensor(padded_prompt_masks, dtype=torch.long),
                "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(padded_attention_masks, dtype=torch.long),
            }
        
        return DataLoader(
            dataset, # type: ignore
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=0,  # Set to 0 for debugging, increase for production
            pin_memory=True,
        )

def test_data_loading():
    """Test function for data loading"""
    from transformers import AutoTokenizer
    from configs.configs import get_config_for_debugging
    
    config = get_config_for_debugging()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-1.5B-Instruct",
        trust_remote_code=True
    )
    
    # Create dataset handler
    dataset_handler = GSM8KDataset(config.data, tokenizer)
    
    # Load datasets
    train_dataset, eval_dataset = dataset_handler.load_dataset()
    
    # Create dataloader
    train_dataloader = dataset_handler.create_grpo_dataloader(
        train_dataset, 
        batch_size=2, 
        shuffle=True
    )
    
    # Test one batch
    batch = next(iter(train_dataloader))
    
    print("Data loading test results:")
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Batch size: {len(batch['questions'])}")
    print(f"Prompt shape: {batch['prompt_input_ids'].shape}")
    print(f"Completion shape: {batch['input_ids'].shape}")
    print(f"Sample question: {batch['questions'][0][:100]}...")
    print(f"Sample prompt: {batch['prompts'][0][:100]}...")
    print(f"Sample numerical answer: {batch['numerical_answers'][0]}")
    
    return True


if __name__ == "__main__":
    test_data_loading()