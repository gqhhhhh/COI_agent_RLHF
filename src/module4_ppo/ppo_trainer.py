"""
Module 4: PPO (Proximal Policy Optimization) Trainer.

Optimizes the SFT model using PPO with a composite reward:
    R_t = alpha * R_rule + beta * R_model

Where:
- R_rule: Rule-based safety reward (negative constraints)
- R_model: Model-based preference reward from the trained Reward Model
- KL penalty prevents policy from deviating too far from the SFT base
"""

import logging
import os
import re
from dataclasses import dataclass, field

from src.utils.data_utils import load_jsonl

logger = logging.getLogger(__name__)

# Negative patterns for rule-based reward
NEGATIVE_PATTERNS = [
    r"(.{20,})\1{2,}",  # Repetitive content (same text repeated 3+ times)
    r"(你好.*){3,}",  # Repeated greetings
]

TOXIC_KEYWORDS = [
    "stupid", "idiot", "hate", "terrible", "worst",
    "garbage", "useless", "pathetic", "disgusting",
]


@dataclass
class PPOTrainerConfig:
    """Configuration for PPO training."""
    sft_model_path: str = "models/sft"
    reward_model_path: str = "models/reward_model"
    output_dir: str = "models/ppo"
    learning_rate: float = 1e-5
    batch_size: int = 4
    mini_batch_size: int = 2
    ppo_epochs: int = 4
    max_new_tokens: int = 256
    kl_penalty: float = 0.2
    alpha_rule: float = 0.3
    beta_model: float = 0.7
    temperature: float = 0.7
    top_p: float = 0.9
    num_train_steps: int = 100
    log_with: str | None = None


def compute_rule_reward(response: str) -> float:
    """
    Compute rule-based safety reward R_rule.

    Checks for negative patterns (repetition, toxicity).
    Returns 0.0 for clean responses, -1.0 for violations.

    Args:
        response: The generated response text.

    Returns:
        Rule-based reward score.
    """
    response_lower = response.lower()

    # Check for toxic keywords
    for keyword in TOXIC_KEYWORDS:
        if keyword in response_lower:
            return -1.0

    # Check for repetitive patterns
    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, response, re.DOTALL):
            return -1.0

    # Check for very short or empty responses
    if len(response.strip()) < 5:
        return -0.5

    return 0.0


def compute_model_reward(
    response: str,
    context: str,
    reward_model,
    reward_tokenizer,
    device: str = "cuda",
) -> float:
    """
    Compute model-based preference reward R_model using the trained Reward Model.

    Args:
        response: Generated response text.
        context: Conversation context (previous turns).
        reward_model: The trained reward model.
        reward_tokenizer: Tokenizer for the reward model.
        device: Device for inference.

    Returns:
        Model-based reward score.
    """
    full_text = context + response
    inputs = reward_tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
        padding=True,
    ).to(device)

    import torch
    with torch.no_grad():
        outputs = reward_model(**inputs)
        # The reward model outputs logits; take the scalar score
        if hasattr(outputs, "logits"):
            reward = outputs.logits.squeeze().item()
        else:
            reward = outputs[0].squeeze().item()

    return reward


def compute_composite_reward(
    response: str,
    context: str,
    reward_model,
    reward_tokenizer,
    alpha: float = 0.3,
    beta: float = 0.7,
    device: str = "cuda",
) -> float:
    """
    Compute composite reward: R_t = alpha * R_rule + beta * R_model

    Args:
        response: Generated response.
        context: Conversation context.
        reward_model: Trained reward model.
        reward_tokenizer: Reward model tokenizer.
        alpha: Weight for rule-based reward.
        beta: Weight for model-based reward.
        device: Device for inference.

    Returns:
        Composite reward score.
    """
    r_rule = compute_rule_reward(response)
    r_model = compute_model_reward(
        response, context, reward_model, reward_tokenizer, device
    )
    return alpha * r_rule + beta * r_model


def prepare_ppo_dataset(
    dialogues: list[dict], tokenizer
):
    """
    Prepare prompts for PPO training from dialogue data.

    Each entry is a context prompt that the model should respond to.

    Args:
        dialogues: List of dialogue dicts.
        tokenizer: Tokenizer for encoding.

    Returns:
        HuggingFace Dataset with 'query' and 'input_ids' columns.
    """
    from datasets import Dataset

    queries = []
    input_ids_list = []

    for dialogue in dialogues:
        turns = dialogue.get("turns", [])
        if len(turns) < 2:
            continue

        # Use first few turns as context, model generates the next agent response
        context_turns = turns[:-1]
        messages = []
        for t in context_turns:
            role = "assistant" if t["role"] == "agent" else "user"
            messages.append({"role": role, "content": t["content"]})

        query_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(query_text, return_tensors="pt", truncation=True,
                            max_length=1024)
        queries.append(query_text)
        input_ids_list.append(encoded["input_ids"].squeeze(0))

    return Dataset.from_dict({
        "query": queries,
        "input_ids": [ids.tolist() for ids in input_ids_list],
    })


def run_ppo_training(
    dialogues: list[dict],
    config: PPOTrainerConfig | None = None,
) -> str:
    """
    Run PPO training loop.

    Args:
        dialogues: List of dialogue dicts for generating prompts.
        config: PPO training configuration.

    Returns:
        Path to the saved model.
    """
    if config is None:
        config = PPOTrainerConfig()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer

    logger.info("Loading SFT model from: %s", config.sft_model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path, trust_remote_code=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load SFT model with value head for PPO
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        peft_config=LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        ),
    )

    # Load reference model (frozen SFT model for KL penalty)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.sft_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Load reward model
    logger.info("Loading Reward Model from: %s", config.reward_model_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(
        config.reward_model_path, trust_remote_code=True
    )
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        config.reward_model_path,
        num_labels=1,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    reward_model.eval()

    # PPO config
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        mini_batch_size=config.mini_batch_size,
        ppo_epochs=config.ppo_epochs,
        kl_penalty="kl",
        init_kl_coef=config.kl_penalty,
        log_with=config.log_with,
    )

    # Prepare dataset
    dataset = prepare_ppo_dataset(dialogues, tokenizer)
    logger.info("PPO dataset size: %d", len(dataset))

    # Create PPO trainer
    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    device = model.pretrained_model.device

    # Generation kwargs
    generation_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    # PPO training loop
    logger.info("Starting PPO training...")
    for step in range(min(config.num_train_steps, len(dataset))):
        # Get batch
        batch_idx = step % len(dataset)
        query_text = dataset[batch_idx]["query"]
        input_ids = torch.tensor(dataset[batch_idx]["input_ids"]).unsqueeze(0).to(device)

        # Generate response
        with torch.no_grad():
            response_ids = model.generate(input_ids, **generation_kwargs)
            new_tokens = response_ids[0][input_ids.shape[1]:]
            response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Compute composite reward: R_t = alpha * R_rule + beta * R_model
        reward_value = compute_composite_reward(
            response=response_text,
            context=query_text,
            reward_model=reward_model,
            reward_tokenizer=reward_tokenizer,
            alpha=config.alpha_rule,
            beta=config.beta_model,
            device=str(device),
        )
        reward_tensor = [torch.tensor([reward_value], device=device)]

        # PPO step
        query_tensors = [input_ids.squeeze(0)]
        response_tensors = [new_tokens]

        stats = ppo_trainer.step(query_tensors, response_tensors, reward_tensor)

        if step % 10 == 0:
            logger.info(
                "Step %d/%d - reward: %.4f, R_rule: %.4f",
                step, config.num_train_steps,
                reward_value, compute_rule_reward(response_text),
            )

    # Save model
    os.makedirs(config.output_dir, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    logger.info("PPO model saved to %s", config.output_dir)

    return config.output_dir
