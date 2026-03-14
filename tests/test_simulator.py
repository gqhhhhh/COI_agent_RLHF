"""Tests for Module 1: User Simulator."""

import json
import os
import tempfile
import uuid

import pytest

from src.module1_simulator.profiles import SAMPLE_PROFILES, get_profile_prompt
from src.module1_simulator.user_simulator import (
    END_KEYWORDS_REJECT,
    END_KEYWORDS_SUCCESS,
    UserSimulator,
    create_dummy_dialogues,
)
from src.utils.data_utils import save_jsonl, load_jsonl


class TestProfiles:
    """Tests for job seeker profiles."""

    def test_sample_profiles_not_empty(self):
        assert len(SAMPLE_PROFILES) > 0

    def test_profile_has_required_fields(self):
        required = ["id", "gender", "age", "years_of_experience",
                     "current_role", "desired_role", "skills",
                     "salary_expectation", "location_preference", "personality"]
        for profile in SAMPLE_PROFILES:
            for field in required:
                assert field in profile, f"Missing field '{field}' in profile {profile.get('id')}"

    def test_get_profile_prompt(self):
        profile = SAMPLE_PROFILES[0]
        prompt = get_profile_prompt(profile)
        assert isinstance(prompt, str)
        assert len(prompt) > 50
        assert profile["gender"] in prompt
        assert profile["current_role"] in prompt
        assert profile["desired_role"] in prompt


class TestEndConditionCheck:
    """Tests for the end condition checking logic."""

    def test_success_detection(self):
        sim = UserSimulator()
        for keyword in END_KEYWORDS_SUCCESS:
            result = sim._check_end_condition(f"I think {keyword}, let me know.")
            assert result == "success", f"Failed to detect success for: {keyword}"

    def test_rejection_detection(self):
        sim = UserSimulator()
        for keyword in END_KEYWORDS_REJECT:
            result = sim._check_end_condition(f"Sorry, {keyword}.")
            assert result == "rejection", f"Failed to detect rejection for: {keyword}"

    def test_no_end_condition(self):
        sim = UserSimulator()
        result = sim._check_end_condition("Tell me more about the position.")
        assert result is None

    def test_case_insensitive(self):
        sim = UserSimulator()
        result = sim._check_end_condition("SOUNDS GREAT!")
        assert result == "success"


class TestDummyDialogues:
    """Tests for dummy dialogue generation."""

    def test_creates_correct_count(self):
        dialogues = create_dummy_dialogues(5)
        assert len(dialogues) == 5

    def test_creates_ten_by_default(self):
        dialogues = create_dummy_dialogues()
        assert len(dialogues) == 10

    def test_dialogue_format(self):
        dialogues = create_dummy_dialogues(3)
        for d in dialogues:
            assert "dialogue_id" in d
            assert "profile_id" in d
            assert "turns" in d
            assert "outcome" in d
            assert "num_turns" in d
            assert isinstance(d["turns"], list)
            assert len(d["turns"]) > 0

    def test_turn_format(self):
        dialogues = create_dummy_dialogues(1)
        for turn in dialogues[0]["turns"]:
            assert "role" in turn
            assert "content" in turn
            assert turn["role"] in ("agent", "user")
            assert isinstance(turn["content"], str)
            assert len(turn["content"]) > 0

    def test_outcome_values(self):
        dialogues = create_dummy_dialogues(10)
        valid_outcomes = {"success", "rejection", "max_turns_reached"}
        for d in dialogues:
            assert d["outcome"] in valid_outcomes

    def test_dialogue_id_is_uuid(self):
        dialogues = create_dummy_dialogues(3)
        for d in dialogues:
            uuid.UUID(d["dialogue_id"])  # Should not raise

    def test_jsonl_round_trip(self):
        dialogues = create_dummy_dialogues(5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.jsonl")
            save_jsonl(dialogues, path)
            loaded = load_jsonl(path)
            assert len(loaded) == len(dialogues)
            assert loaded[0]["dialogue_id"] == dialogues[0]["dialogue_id"]
