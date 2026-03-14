"""
Module 3a: SFT (Supervised Fine-Tuning) Trainer.

Fine-tunes Qwen2.5-7B with LoRA on CoI-filtered high-quality dialogue data
using HuggingFace's trl SFTTrainer.
"""

import logging
import os
from dataclasses import dataclass, field

from src.utils.data_utils import format_dialogue_for_sft, load_jsonl

logger = logging.getLogger(__name__)


@dataclass
class SFTTrainerConfig:
    """Configuration for SFT training."""
    model_name: str = "Qwen/Qwen2.5-7B"
    output_dir: str = "models/sft"
    learning_rate: float = 2e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
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


def prepare_sft_dataset(dialogues: list[dict], tokenizer):
    """
    Convert dialogue dicts to a HuggingFace Dataset for SFT training.

    Each dialogue is formatted as a chat conversation and then
    tokenized into a text field.

    Args:
        dialogues: List of dialogue dicts (from Module 1 output).
        tokenizer: The tokenizer to apply chat template.

    Returns:
        HuggingFace Dataset.
    """
    texts = []
    for d in dialogues:
        messages = format_dialogue_for_sft(d)
        if not messages:
            continue
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)

    from datasets import Dataset
    return Dataset.from_dict({"text": texts})


def build_lora_config(config: SFTTrainerConfig):
    """Create a LoRA configuration."""
    from peft import LoraConfig, TaskType
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def run_sft_training(
    dialogues: list[dict],
    config: SFTTrainerConfig | None = None,
) -> str:
    """
    Run SFT training on the provided dialogues.

    Args:
        dialogues: List of dialogue dicts.
        config: SFT training configuration.

    Returns:
        Path to the saved model.
    """
    if config is None:
        config = SFTTrainerConfig()

    import torch
    from peft import get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    logger.info("Loading tokenizer and model: %s", config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = build_lora_config(config)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    dataset = prepare_sft_dataset(dialogues, tokenizer)
    logger.info("SFT dataset size: %d", len(dataset))

    # Training arguments
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        max_seq_length=config.max_seq_length,
        report_to="none",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    # Train
    logger.info("Starting SFT training...")
    trainer.train()

    # Save
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("SFT model saved to %s", config.output_dir)

    return config.output_dir
