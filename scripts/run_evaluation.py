#!/usr/bin/env python3
"""
Script to run the CoI Evaluator (Module 2) for dialogue evaluation and filtering.

Usage:
    python scripts/run_evaluation.py --input data/output/dialogues.jsonl --output data/output/preference_data.jsonl
    python scripts/run_evaluation.py --dummy  # Use dummy evaluation (no model)
"""

import argparse
import logging
import sys

sys.path.insert(0, ".")

from src.module2_coi_evaluator.coi_evaluator import (
    CoIEvaluator,
    CoIEvaluatorConfig,
    evaluate_and_filter_dummy,
)
from src.utils.data_utils import load_jsonl, save_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run CoI Evaluation and Filtering")
    parser.add_argument("--input", type=str, default="data/output/simulated_dialogues.jsonl")
    parser.add_argument("--output", type=str, default="data/output/preference_data.jsonl")
    parser.add_argument("--scored_output", type=str, default="data/output/scored_dialogues.jsonl")
    parser.add_argument("--dummy", action="store_true", help="Use rule-based evaluation (no model)")
    args = parser.parse_args()

    logger.info("Loading dialogues from %s", args.input)
    dialogues = load_jsonl(args.input)
    logger.info("Loaded %d dialogues", len(dialogues))

    if args.dummy:
        scored_dialogues, preference_pairs = evaluate_and_filter_dummy(dialogues)
    else:
        config = CoIEvaluatorConfig()
        evaluator = CoIEvaluator(config)
        evaluator.load_model()
        # Full evaluation would go here with LLM-as-a-Judge
        # For now, fall back to dummy
        scored_dialogues, preference_pairs = evaluate_and_filter_dummy(dialogues)

    # Save scored dialogues
    save_jsonl(scored_dialogues, args.scored_output)
    logger.info("Saved %d scored dialogues to %s", len(scored_dialogues), args.scored_output)

    # Save preference pairs
    save_jsonl(preference_pairs, args.output)
    logger.info("Saved %d preference pairs to %s", len(preference_pairs), args.output)

    logger.info("Done!")


if __name__ == "__main__":
    main()
