import os
import sys
import argparse
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch
import numpy as np
import wandb
from tqdm.auto import tqdm
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from configs.configs import Config, get_default_config, get_config_for_debugging
from src.data_loader import GSM8KDataset
from src.model import GRPOModelWrapper
from src.grpo_trainer import GRPOTrainer

console = Console()

def setup_logging(log_level = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )
    
    return logging.getLogger("tiny-grpo")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_wandb(config, run_name):
    if not config.logging.use_wandb:
        return None
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"grpo-{timestamp}"
    
    # Initialize wandb
    wandb.init(
        project=config.logging.wandb_project,
        entity=config.logging.wandb_entity,
        name=run_name,
        config=config.to_dict(),
        tags=["grpo", "qwen", "gsm8k"],
    )
    
    return wandb

def create_run_dir(base_dir= "./runs"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(base_dir, f"grpo-{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def log_metrics(logger, step, metrics, prefix = ""):
    if prefix:
        logger.info(f"[{prefix}] Step {step}")
    else:
        logger.info(f"Step {step}")
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Wandb logging
    if wandb.run is not None:
        wandb_metrics = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}
        wandb_metrics["step"] = step
        wandb.log(wandb_metrics)


def save_training_info(run_dir, config, args):
    import json
    
    # Save config
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2, default=str)
    
    # Save args
    args_path = os.path.join(run_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    
    console.print(f"Training info saved to {run_dir}")


def display_training_summary(config, model_wrapper):
    table = Table(title="Training Summary")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="magenta")
    
    # Model info
    table.add_row("Model", config.model.model_name)
    table.add_row("PEFT", "LoRA" if config.model.use_peft else "Full Fine-tuning")
    if config.model.use_peft:
        table.add_row("LoRA Rank", str(config.model.lora_r))
        table.add_row("LoRA Alpha", str(config.model.lora_alpha))
    
    # Training info
    table.add_row("Dataset", config.data.dataset_name)
    table.add_row("Group Size", str(config.grpo.group_size))
    table.add_row("Batch Size", str(config.training.per_device_train_batch_size))
    table.add_row("Learning Rate", str(config.training.learning_rate))
    table.add_row("Epochs", str(config.training.num_epochs))
    
    # Memory optimization
    table.add_row("Mixed Precision", str(config.grpo.use_mixed_precision))
    table.add_row("Gradient Checkpointing", str(config.grpo.gradient_checkpointing))
    table.add_row("Chunk Size", str(config.grpo.chunk_size))
    
    console.print(table)

def main():
    parser = argparse.ArgumentParser(description="Train Tiny-GRPO model")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Use debug configuration")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--output-dir", type=str, help="Output directory override")
    parser.add_argument("--run-name", type=str, help="Custom run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("Starting Tiny-GRPO training")

    if args.debug:
        config = get_config_for_debugging()
        logger.info("Using debug configuration")
    else:
        config = get_default_config()
        logger.info("Using default configuration")

    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.no_wandb:
        config.logging.use_wandb = False
    if args.resume:
        config.training.resume_from_checkpoint = args.resume

    set_seed(config.seed)
    logger.info(f"Random seed set to {config.seed}")

    run_dir = create_run_dir(config.training.output_dir)
    config.training.output_dir = run_dir
    config.training.logging_dir = os.path.join(run_dir, "logs")
    
    save_training_info(run_dir, config, args)
    
    wandb_run = setup_wandb(config, args.run_name)
    if wandb_run:
        logger.info(f"Wandb initialized.")

    try:
        logger.info("Loading model and tokenizer")
        model_wrapper = GRPOModelWrapper(config.model)
        model, tokenizer = model_wrapper.load_model_and_tokenizer()

        # Setup peft
        if config.model.use_peft:
            model = model_wrapper.setup_peft()

        # Setup reference model
        ref_model = model_wrapper.setup_reference_model()

        display_training_summary(config, model_wrapper)
        
        # Loading dataset
        logger.info("Loading dataset")
        dataset_handler = GSM8KDataset(config.data, tokenizer)
        train_dataset, eval_dataset = dataset_handler.load_dataset()

        train_dataloader = dataset_handler.create_grpo_dataloader(
            train_dataset, 
            batch_size=config.training.per_device_train_batch_size,
            shuffle=True
        )

        eval_dataloader = dataset_handler.create_grpo_dataloader(
            eval_dataset,
            batch_size=config.training.per_device_eval_batch_size,
            shuffle=False
        )

        logger.info(f"Train batches: {len(train_dataloader)}")
        logger.info(f"Evaluation batches: {len(eval_dataloader)}")

        # Create trainer
        logger.info("Initializing GRPO trainer")
        trainer = GRPOTrainer(config, model_wrapper)

        # Update scheduler
        total_steps = len(train_dataloader) * config.training.num_epochs
        trainer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=total_steps,
            eta_min=config.training.learning_rate * 0.1
        )

        # Resume from checkpoint if specified
        if config.training.resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {config.training.resume_from_checkpoint}")
            trainer.load_checkpoint(config.training.resume_from_checkpoint)
        
        logger.info("Starting training loop.")

        total_steps = 0
        best_eval_reward = -float('inf')

        for epoch in range(trainer.epoch, config.training.num_epochs):
            trainer.epoch = epoch
            logger.info(f"Epoch {epoch + 1}/{config.training.num_epochs}")

            model.train()
            epoch_metrics = {"loss": 0.0, "pg_loss": 0.0, "kl_loss": 0.0, "reward": 0.0}
            epoch_steps = 0

            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            )

            with progress:
                task = progress.add_task(f"Training Epoch {epoch + 1}", total=len(train_dataloader))

                for step, batch_data in enumerate(train_dataloader):
                    # Convert dataloader batch to grpo batch
                    prompts = batch_data["prompts"]

                    progress.console.log(f"Generating completions for {len(prompts)} prompts...")
                    grpo_batch = trainer.generate_completions_for_batch(prompts)

                    # Training step
                    step_metrics = trainer.training_step(grpo_batch)

                    # Accumulate metrics
                    for key, value in step_metrics.items():
                        if key in epoch_metrics:
                            epoch_metrics[key] += value
                    epoch_steps += 1
                    total_steps += 1

                    if total_steps % config.training.logging_steps == 0:
                        log_metrics(logger, total_steps, step_metrics, "train")

                    # Evaluation
                    if total_steps % config.training.eval_steps == 0:
                        logger.info("Running evaluation.")
                        eval_metrics = trainer.evaluate(eval_dataloader)
                        log_metrics(logger, total_steps, eval_metrics, "eval")
                        
                        # Check if best model
                        if eval_metrics["new_best"]:
                            best_eval_reward = eval_metrics["eval_reward"]
                            best_model_path = os.path.join(run_dir, "best_model")
                            trainer.save_checkpoint(best_model_path, eval_metrics)
                            logger.info(f"New best model saved! Reward: {best_eval_reward:.4f}")
                    
                    # Save checkpoint
                    if total_steps % config.training.save_steps == 0:
                        checkpoint_path = os.path.join(run_dir, f"checkpoint-{total_steps}")
                        trainer.save_checkpoint(checkpoint_path, step_metrics)
                    
                    progress.update(task, advance=1)

            # End of epoch logging
            avg_epoch_metrics = {k: v / epoch_steps for k, v in epoch_metrics.items()}
            log_metrics(logger, total_steps, avg_epoch_metrics, f"epoch_{epoch}")
            
            # Save epoch checkpoint
            epoch_checkpoint_path = os.path.join(run_dir, f"epoch-{epoch}")
            trainer.save_checkpoint(epoch_checkpoint_path, avg_epoch_metrics)

        # Final evaluation
        logger.info("🏁 Running final evaluation...")
        final_eval_metrics = trainer.evaluate(eval_dataloader)
        log_metrics(logger, total_steps, final_eval_metrics, "final_eval")
        
        # Save final model
        final_model_path = os.path.join(run_dir, "final_model")
        trainer.save_checkpoint(final_model_path, final_eval_metrics)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Models saved to: {run_dir}")
        logger.info(f"Best eval reward: {best_eval_reward:.4f}")
            

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        console.print_exception()
        return 1

    finally:
        if wandb.run is not None:
            wandb.finish()
        
        if torch.mps.is_available():
            torch.mps.empty_cache()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)