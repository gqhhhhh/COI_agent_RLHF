#!/usr/bin/env python3
"""
Script to run the User Simulator (Module 1) for multi-turn dialogue generation.

Usage:
    python scripts/run_simulator.py --num_dialogues 100 --output data/output/dialogues.jsonl
    python scripts/run_simulator.py --dummy  # Generate dummy data without model
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.module1_simulator.user_simulator import (
    SimulatorConfig,
    UserSimulator,
    create_dummy_dialogues,
)
from src.utils.data_utils import save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run User Simulator for dialogue generation")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B")
    parser.add_argument("--num_dialogues", type=int, default=10)
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--output", type=str, default="data/output/simulated_dialogues.jsonl")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy data without model")
    args = parser.parse_args()

    if args.dummy:
        logger.info("Generating %d dummy dialogues...", args.num_dialogues)
        dialogues = create_dummy_dialogues(args.num_dialogues)
        save_jsonl(dialogues, args.output)
        logger.info("Saved %d dummy dialogues to %s", len(dialogues), args.output)
    else:
        config = SimulatorConfig(
            model_name=args.model_name,
            num_dialogues=args.num_dialogues,
            max_turns=args.max_turns,
            output_path=args.output,
        )
        simulator = UserSimulator(config)
        simulator.load_model()
        simulator.generate_and_save()

    logger.info("Done!")


if __name__ == "__main__":
    main()
