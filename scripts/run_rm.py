#!/usr/bin/env python3
"""
Script to run Reward Model training (Module 3b).

Usage:
    python scripts/run_rm.py --input data/output/preference_data.jsonl --output models/reward_model
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.module3_training.rm_trainer import RMTrainerConfig, run_rm_training
from src.utils.data_utils import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run Reward Model Training")
    parser.add_argument("--input", type=str, default="data/output/preference_data.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output", type=str, default="models/reward_model")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    logger.info("Loading preference pairs from %s", args.input)
    preference_pairs = load_jsonl(args.input)
    logger.info("Loaded %d preference pairs for RM training", len(preference_pairs))

    config = RMTrainerConfig(
        model_name=args.model_name,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    output_path = run_rm_training(preference_pairs, config)
    logger.info("RM training complete. Model saved to: %s", output_path)


if __name__ == "__main__":
    main()
