#!/usr/bin/env python3
"""
Script to run PPO training (Module 4).

Usage:
    python scripts/run_ppo.py --sft_model models/sft --rm_model models/reward_model --input data/output/dialogues.jsonl
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.module4_ppo.ppo_trainer import PPOTrainerConfig, run_ppo_training
from src.utils.data_utils import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run PPO Training")
    parser.add_argument("--input", type=str, default="data/output/simulated_dialogues.jsonl")
    parser.add_argument("--sft_model", type=str, default="models/sft")
    parser.add_argument("--rm_model", type=str, default="models/reward_model")
    parser.add_argument("--output", type=str, default="models/ppo")
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=0.3, help="Weight for rule-based reward")
    parser.add_argument("--beta", type=float, default=0.7, help="Weight for model-based reward")
    args = parser.parse_args()

    logger.info("Loading dialogues from %s", args.input)
    dialogues = load_jsonl(args.input)
    logger.info("Loaded %d dialogues for PPO", len(dialogues))

    config = PPOTrainerConfig(
        sft_model_path=args.sft_model,
        reward_model_path=args.rm_model,
        output_dir=args.output,
        num_train_steps=args.num_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        alpha_rule=args.alpha,
        beta_model=args.beta,
    )

    output_path = run_ppo_training(dialogues, config)
    logger.info("PPO training complete. Model saved to: %s", output_path)


if __name__ == "__main__":
    main()
