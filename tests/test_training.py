"""Tests for Module 3 & 4: Training data format and utility functions."""

import pytest

from src.module1_simulator.user_simulator import create_dummy_dialogues
from src.module2_coi_evaluator.coi_evaluator import evaluate_and_filter_dummy
from src.module4_ppo.ppo_trainer import compute_rule_reward
from src.utils.data_utils import (
    format_dialogue_for_sft,
    format_preference_pair,
)


class TestSFTDataFormat:
    """Tests for SFT data formatting."""

    def test_format_dialogue_for_sft(self):
        dialogues = create_dummy_dialogues(1)
        messages = format_dialogue_for_sft(dialogues[0])
        assert len(messages) > 0
        for msg in messages:
            assert "role" in msg
            assert "content" in msg
            assert msg["role"] in ("user", "assistant")

    def test_agent_role_mapped_to_assistant(self):
        dialogue = {
            "turns": [
                {"role": "agent", "content": "Hello"},
                {"role": "user", "content": "Hi"},
            ]
        }
        messages = format_dialogue_for_sft(dialogue)
        assert messages[0]["role"] == "assistant"
        assert messages[1]["role"] == "user"

    def test_empty_turns(self):
        messages = format_dialogue_for_sft({"turns": []})
        assert len(messages) == 0

    def test_missing_turns_key(self):
        messages = format_dialogue_for_sft({})
        assert len(messages) == 0


class TestPreferencePairFormat:
    """Tests for preference pair formatting."""

    def test_format_preference_pair(self):
        context = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        pair = format_preference_pair(context, "Great answer", "Bad answer")
        assert "chosen" in pair
        assert "rejected" in pair
        assert len(pair["chosen"]) == 3
        assert len(pair["rejected"]) == 3
        assert pair["chosen"][-1]["content"] == "Great answer"
        assert pair["rejected"][-1]["content"] == "Bad answer"


class TestRuleReward:
    """Tests for rule-based reward computation."""

    def test_clean_response_reward(self):
        reward = compute_rule_reward(
            "Thank you for your interest. We have a great position available."
        )
        assert reward == 0.0

    def test_toxic_keyword_penalty(self):
        for keyword in ["stupid", "idiot", "hate", "garbage"]:
            reward = compute_rule_reward(f"This is {keyword}!")
            assert reward == -1.0, f"Toxic keyword '{keyword}' should give -1.0"

    def test_short_response_penalty(self):
        reward = compute_rule_reward("Hi")
        assert reward == -0.5

    def test_empty_response_penalty(self):
        reward = compute_rule_reward("")
        assert reward == -0.5

    def test_normal_length_response(self):
        reward = compute_rule_reward("I would be happy to discuss the position details with you.")
        assert reward == 0.0


class TestPipelineDataFlow:
    """Test that data flows correctly between pipeline stages."""

    def test_simulator_to_evaluator(self):
        """Module 1 -> Module 2: Dialogue format compatibility."""
        dialogues = create_dummy_dialogues(10)
        scored, pairs = evaluate_and_filter_dummy(dialogues)
        assert len(scored) == 10
        assert len(pairs) > 0

    def test_evaluator_to_sft(self):
        """Module 2 -> Module 3a: Scored dialogues to SFT format."""
        dialogues = create_dummy_dialogues(5)
        scored, _ = evaluate_and_filter_dummy(dialogues)
        for s in scored:
            messages = format_dialogue_for_sft(s["dialogue"])
            assert len(messages) > 0
            for msg in messages:
                assert msg["role"] in ("user", "assistant")

    def test_evaluator_to_rm(self):
        """Module 2 -> Module 3b: Preference pairs have correct format."""
        dialogues = create_dummy_dialogues(10)
        _, pairs = evaluate_and_filter_dummy(dialogues)
        for pair in pairs:
            assert isinstance(pair["chosen"], list)
            assert isinstance(pair["rejected"], list)
            for msg in pair["chosen"] + pair["rejected"]:
                assert "role" in msg
                assert "content" in msg
