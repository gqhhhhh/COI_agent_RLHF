#!/usr/bin/env python3
"""
Script to run SFT training (Module 3a).

Usage:
    python scripts/run_sft.py --input data/output/simulated_dialogues.jsonl --output models/sft
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.module3_training.sft_trainer import SFTTrainerConfig, run_sft_training
from src.utils.data_utils import load_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run SFT Training")
    parser.add_argument("--input", type=str, default="data/output/simulated_dialogues.jsonl")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--output", type=str, default="models/sft")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    logger.info("Loading dialogues from %s", args.input)
    dialogues = load_jsonl(args.input)
    logger.info("Loaded %d dialogues for SFT", len(dialogues))

    config = SFTTrainerConfig(
        model_name=args.model_name,
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    output_path = run_sft_training(dialogues, config)
    logger.info("SFT training complete. Model saved to: %s", output_path)


if __name__ == "__main__":
    main()
