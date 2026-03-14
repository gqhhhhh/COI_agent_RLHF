"""
Module 3b: Reward Model (RM) Trainer.

Trains a reward model on preference pairs (chosen/rejected) using
HuggingFace's trl RewardTrainer with LoRA on Qwen2.5-7B.
The model outputs a scalar score for each input sequence.
"""

import logging
import os
from dataclasses import dataclass, field

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from trl import RewardConfig, RewardTrainer

from src.utils.data_utils import load_jsonl

logger = logging.getLogger(__name__)


@dataclass
class RMTrainerConfig:
    """Configuration for Reward Model training."""
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/reward_model"
    learning_rate: float = 1e-5
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_length: int = 2048
    logging_steps: int = 10
    save_steps: int = 100
    warmup_ratio: float = 0.1
    fp16: bool = True
    # LoRA parameters
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )


def prepare_rm_dataset(
    preference_pairs: list[dict], tokenizer
) -> Dataset:
    """
    Convert preference pairs to the format expected by RewardTrainer.

    RewardTrainer expects columns: 'chosen' and 'rejected' as tokenized text.

    Args:
        preference_pairs: List of dicts with 'chosen' and 'rejected' message lists.
        tokenizer: Tokenizer for applying chat template.

    Returns:
        HuggingFace Dataset with 'chosen' and 'rejected' text columns.
    """
    chosen_texts = []
    rejected_texts = []

    for pair in preference_pairs:
        chosen_msgs = pair.get("chosen", [])
        rejected_msgs = pair.get("rejected", [])

        if not chosen_msgs or not rejected_msgs:
            continue

        chosen_text = tokenizer.apply_chat_template(
            chosen_msgs, tokenize=False, add_generation_prompt=False
        )
        rejected_text = tokenizer.apply_chat_template(
            rejected_msgs, tokenize=False, add_generation_prompt=False
        )

        chosen_texts.append(chosen_text)
        rejected_texts.append(rejected_text)

    return Dataset.from_dict({
        "chosen": chosen_texts,
        "rejected": rejected_texts,
    })


def build_rm_lora_config(config: RMTrainerConfig) -> LoraConfig:
    """Create a LoRA configuration for the reward model."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.SEQ_CLS,
        bias="none",
    )


def run_rm_training(
    preference_pairs: list[dict],
    config: RMTrainerConfig | None = None,
) -> str:
    """
    Run Reward Model training on preference pairs.

    Args:
        preference_pairs: List of preference pair dicts.
        config: RM training configuration.

    Returns:
        Path to the saved model.
    """
    if config is None:
        config = RMTrainerConfig()

    logger.info("Loading tokenizer and model: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Apply LoRA
    lora_config = build_rm_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    dataset = prepare_rm_dataset(preference_pairs, tokenizer)
    logger.info("RM dataset size: %d", len(dataset))

    # Training arguments
    reward_config = RewardConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        max_length=config.max_length,
        report_to="none",
    )

    # Create trainer
    trainer = RewardTrainer(
        model=model,
        args=reward_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting Reward Model training...")
    trainer.train()

    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("Reward Model saved to %s", config.output_dir)

    return config.output_dir
