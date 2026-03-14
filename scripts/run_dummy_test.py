#!/usr/bin/env python3
"""
End-to-end dummy test script.

Validates the entire pipeline (Simulator -> CoI Evaluation -> SFT -> RM -> PPO)
data flow using 10 dummy dialogues, WITHOUT requiring a GPU or model download.

This script verifies:
1. Dummy dialogue generation produces correct format
2. CoI evaluation and preference pair construction works
3. Data format is compatible with SFT/RM/PPO expectations

Usage:
    python scripts/run_dummy_test.py
"""

import json
import logging
import os
import sys
import tempfile

sys.path.insert(0, ".")

from src.module1_simulator.user_simulator import create_dummy_dialogues
from src.module1_simulator.profiles import SAMPLE_PROFILES, get_profile_prompt
from src.module2_coi_evaluator.coi_evaluator import (
    CoIEvaluator,
    evaluate_and_filter_dummy,
)
from src.module2_coi_evaluator.intent_classifier import (
    INTENT_CATEGORIES,
    classify_intent_rule_based,
)
from src.module4_ppo.ppo_trainer import compute_rule_reward
from src.utils.data_utils import (
    save_jsonl,
    load_jsonl,
    format_dialogue_for_sft,
    format_preference_pair,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_module1_dummy_generation():
    """Test Module 1: Dummy dialogue generation."""
    logger.info("=" * 60)
    logger.info("Testing Module 1: Dummy Dialogue Generation")
    logger.info("=" * 60)

    dialogues = create_dummy_dialogues(10)

    assert len(dialogues) == 10, f"Expected 10 dialogues, got {len(dialogues)}"

    for d in dialogues:
        assert "dialogue_id" in d, "Missing dialogue_id"
        assert "turns" in d, "Missing turns"
        assert "outcome" in d, "Missing outcome"
        assert isinstance(d["turns"], list), "turns should be a list"
        assert len(d["turns"]) > 0, "turns should not be empty"

        for turn in d["turns"]:
            assert "role" in turn, "Missing role in turn"
            assert "content" in turn, "Missing content in turn"
            assert turn["role"] in ("agent", "user"), f"Invalid role: {turn['role']}"

    logger.info("✓ Generated %d dialogues with correct format", len(dialogues))
    logger.info("  Outcomes: %s", [d["outcome"] for d in dialogues])
    return dialogues


def test_module2_evaluation(dialogues):
    """Test Module 2: CoI evaluation and filtering."""
    logger.info("=" * 60)
    logger.info("Testing Module 2: CoI Evaluation & Filtering")
    logger.info("=" * 60)

    # Test intent classification
    for d in dialogues[:3]:
        intents = [classify_intent_rule_based(t) for t in d["turns"]]
        logger.info("  Dialogue %s intents: %s", d["dialogue_id"][:8], intents)
        for intent in intents:
            assert intent in INTENT_CATEGORIES, f"Invalid intent: {intent}"

    # Test evaluation
    scored_dialogues, preference_pairs = evaluate_and_filter_dummy(dialogues)

    assert len(scored_dialogues) == len(dialogues), "Scored dialogues count mismatch"
    for sd in scored_dialogues:
        assert "composite_score" in sd, "Missing composite_score"
        assert "dialogue" in sd, "Missing dialogue"
        assert 0 <= sd["composite_score"] <= 1, f"Score out of range: {sd['composite_score']}"

    assert len(preference_pairs) > 0, "No preference pairs generated"
    for pair in preference_pairs:
        assert "chosen" in pair, "Missing chosen in preference pair"
        assert "rejected" in pair, "Missing rejected in preference pair"
        assert isinstance(pair["chosen"], list), "chosen should be a list of messages"
        assert isinstance(pair["rejected"], list), "rejected should be a list of messages"

    logger.info("✓ Scored %d dialogues", len(scored_dialogues))
    logger.info("✓ Generated %d preference pairs", len(preference_pairs))

    # Test transition matrix
    evaluator = CoIEvaluator()
    all_intents = [[classify_intent_rule_based(t) for t in d["turns"]] for d in dialogues]
    matrix = evaluator.build_transition_matrix(all_intents)
    assert matrix.shape == (len(INTENT_CATEGORIES), len(INTENT_CATEGORIES))
    logger.info("✓ Transition matrix shape: %s", matrix.shape)

    return scored_dialogues, preference_pairs


def test_module3_data_format(dialogues, preference_pairs):
    """Test Module 3: SFT and RM data format compatibility."""
    logger.info("=" * 60)
    logger.info("Testing Module 3: SFT & RM Data Format")
    logger.info("=" * 60)

    # Test SFT format
    for d in dialogues[:3]:
        messages = format_dialogue_for_sft(d)
        assert len(messages) > 0, "SFT messages should not be empty"
        for msg in messages:
            assert "role" in msg, "Missing role in SFT message"
            assert "content" in msg, "Missing content in SFT message"
            assert msg["role"] in ("user", "assistant"), f"Invalid SFT role: {msg['role']}"
        logger.info("  SFT format: %d messages for dialogue %s",
                     len(messages), d["dialogue_id"][:8])

    # Test RM format
    for pair in preference_pairs[:3]:
        assert all("role" in m and "content" in m for m in pair["chosen"])
        assert all("role" in m and "content" in m for m in pair["rejected"])
        logger.info("  RM pair: chosen=%d msgs, rejected=%d msgs",
                     len(pair["chosen"]), len(pair["rejected"]))

    logger.info("✓ SFT data format verified")
    logger.info("✓ RM preference pair format verified")


def test_module4_rule_reward():
    """Test Module 4: Rule-based reward computation."""
    logger.info("=" * 60)
    logger.info("Testing Module 4: Rule-Based Reward")
    logger.info("=" * 60)

    # Clean response
    reward = compute_rule_reward("Thank you for your interest. Let me tell you about the position.")
    assert reward == 0.0, f"Clean response should get 0.0 reward, got {reward}"
    logger.info("✓ Clean response reward: %.2f", reward)

    # Toxic response
    reward = compute_rule_reward("This is stupid and you are an idiot.")
    assert reward == -1.0, f"Toxic response should get -1.0 reward, got {reward}"
    logger.info("✓ Toxic response reward: %.2f", reward)

    # Short response
    reward = compute_rule_reward("Hi")
    assert reward == -0.5, f"Short response should get -0.5 reward, got {reward}"
    logger.info("✓ Short response reward: %.2f", reward)

    logger.info("✓ Rule-based reward computation verified")


def test_jsonl_io(dialogues, preference_pairs):
    """Test JSONL save/load round-trip."""
    logger.info("=" * 60)
    logger.info("Testing JSONL I/O Round-Trip")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test dialogues
        path = os.path.join(tmpdir, "dialogues.jsonl")
        save_jsonl(dialogues, path)
        loaded = load_jsonl(path)
        assert len(loaded) == len(dialogues), "Dialogue count mismatch after round-trip"
        logger.info("✓ Dialogue JSONL round-trip: %d records", len(loaded))

        # Test preference pairs
        path = os.path.join(tmpdir, "preferences.jsonl")
        save_jsonl(preference_pairs, path)
        loaded = load_jsonl(path)
        assert len(loaded) == len(preference_pairs), "Preference pair count mismatch"
        logger.info("✓ Preference JSONL round-trip: %d records", len(loaded))


def main():
    logger.info("🚀 Starting End-to-End Dummy Pipeline Test")
    logger.info("=" * 60)

    # Module 1: Generate dummy dialogues
    dialogues = test_module1_dummy_generation()

    # Module 2: Evaluate and filter
    scored_dialogues, preference_pairs = test_module2_evaluation(dialogues)

    # Module 3: Verify data format compatibility
    test_module3_data_format(dialogues, preference_pairs)

    # Module 4: Test rule-based reward
    test_module4_rule_reward()

    # I/O round-trip
    test_jsonl_io(dialogues, preference_pairs)

    logger.info("=" * 60)
    logger.info("🎉 ALL TESTS PASSED! Pipeline data flow is verified.")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run with real model: python scripts/run_simulator.py")
    logger.info("  2. Evaluate: python scripts/run_evaluation.py --dummy")
    logger.info("  3. SFT: python scripts/run_sft.py")
    logger.info("  4. RM: python scripts/run_rm.py")
    logger.info("  5. PPO: python scripts/run_ppo.py")


if __name__ == "__main__":
    main()
