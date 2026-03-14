"""Tests for Module 2: CoI Evaluator and Intent Classifier."""

import numpy as np
import pytest

from src.module1_simulator.user_simulator import create_dummy_dialogues
from src.module2_coi_evaluator.coi_evaluator import (
    CoIEvaluator,
    evaluate_and_filter_dummy,
)
from src.module2_coi_evaluator.intent_classifier import (
    INTENT_CATEGORIES,
    classify_intent_rule_based,
)


class TestIntentClassifier:
    """Tests for rule-based intent classifier."""

    def test_rejection_intent(self):
        turn = {"role": "user", "content": "I'm not interested in this position."}
        assert classify_intent_rule_based(turn) == "explicit_rejection"

    def test_success_intent(self):
        turn = {"role": "user", "content": "Sounds great! I'm interested."}
        assert classify_intent_rule_based(turn) == "successful_conversion"

    def test_hesitation_intent(self):
        turn = {"role": "user", "content": "Let me think about it."}
        assert classify_intent_rule_based(turn) == "hesitation"

    def test_salary_intent(self):
        turn = {"role": "user", "content": "What's the salary range for this role?"}
        assert classify_intent_rule_based(turn) == "salary_negotiation"

    def test_benefit_intent(self):
        turn = {"role": "user", "content": "Do you offer remote work options?"}
        assert classify_intent_rule_based(turn) == "benefit_discussion"

    def test_job_recommendation_intent(self):
        turn = {"role": "agent", "content": "We have a great position for you."}
        assert classify_intent_rule_based(turn) == "job_recommendation"

    def test_default_intent(self):
        turn = {"role": "user", "content": "Hello, how are you?"}
        assert classify_intent_rule_based(turn) == "information_inquiry"

    def test_all_intents_valid(self):
        turns = [
            {"role": "user", "content": "Tell me about the company"},
            {"role": "agent", "content": "We have a position for you"},
            {"role": "user", "content": "What qualifications do you require?"},
            {"role": "user", "content": "What's the salary?"},
            {"role": "user", "content": "Any remote work benefits?"},
            {"role": "user", "content": "Let me think about it, maybe later"},
            {"role": "user", "content": "No thanks, not interested"},
            {"role": "user", "content": "Sounds great! I accept."},
            {"role": "agent", "content": "Let's schedule the next step interview."},
        ]
        for turn in turns:
            intent = classify_intent_rule_based(turn)
            assert intent in INTENT_CATEGORIES, f"Invalid intent: {intent}"

    def test_intent_categories_count(self):
        assert len(INTENT_CATEGORIES) == 9


class TestCoIEvaluator:
    """Tests for CoI Evaluator."""

    def test_build_transition_matrix(self):
        sequences = [
            ["information_inquiry", "job_recommendation", "salary_negotiation"],
            ["information_inquiry", "hesitation", "explicit_rejection"],
        ]
        matrix = CoIEvaluator.build_transition_matrix(sequences)
        assert matrix.shape == (9, 9)
        # Each row should sum to ~1 (probability distribution)
        for row in matrix:
            assert abs(row.sum() - 1.0) < 1e-6

    def test_kl_divergence_same_distribution(self):
        p = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        kl = CoIEvaluator.kl_divergence(p, p)
        assert abs(kl) < 1e-6, f"KL divergence of same distribution should be ~0, got {kl}"

    def test_kl_divergence_different_distributions(self):
        p = np.array([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        q = np.array([[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]])
        kl = CoIEvaluator.kl_divergence(p, q)
        assert kl > 0, "KL divergence of different distributions should be > 0"

    def test_js_divergence_same_distribution(self):
        p = np.array([[0.5, 0.3, 0.2]])
        js = CoIEvaluator.js_divergence(p, p)
        assert abs(js) < 1e-6, f"JS divergence of same distribution should be ~0, got {js}"

    def test_js_divergence_symmetric(self):
        p = np.array([[0.5, 0.3, 0.2]])
        q = np.array([[0.2, 0.5, 0.3]])
        js_pq = CoIEvaluator.js_divergence(p, q)
        js_qp = CoIEvaluator.js_divergence(q, p)
        assert abs(js_pq - js_qp) < 1e-6, "JS divergence should be symmetric"

    def test_js_divergence_bounded(self):
        p = np.array([[0.9, 0.05, 0.05]])
        q = np.array([[0.05, 0.05, 0.9]])
        js = CoIEvaluator.js_divergence(p, q)
        assert 0 <= js <= np.log(2) + 1e-6, f"JS divergence should be bounded, got {js}"

    def test_evaluate_global(self):
        evaluator = CoIEvaluator()
        real = [
            ["information_inquiry", "job_recommendation", "successful_conversion"],
            ["information_inquiry", "salary_negotiation", "hesitation"],
        ]
        synth = [
            ["information_inquiry", "job_recommendation", "hesitation"],
            ["information_inquiry", "benefit_discussion", "successful_conversion"],
        ]
        result = evaluator.evaluate_global(synth, real)
        assert "kl_divergence" in result
        assert "js_divergence" in result
        assert "global_score" in result
        assert 0 <= result["global_score"] <= 1

    def test_compute_composite_score(self):
        evaluator = CoIEvaluator()
        instance_scores = {"style_sim": 0.8, "result_f1": 0.7}
        composite = evaluator.compute_composite_score(instance_scores, 0.9)
        assert 0 <= composite <= 1

    def test_construct_preference_pairs(self):
        scored = [
            {"profile_id": "p1", "composite_score": 0.9,
             "dialogue": {"turns": [{"role": "agent", "content": "Good dialogue"}]}},
            {"profile_id": "p1", "composite_score": 0.3,
             "dialogue": {"turns": [{"role": "agent", "content": "Bad dialogue"}]}},
        ]
        pairs = CoIEvaluator.construct_preference_pairs(scored)
        assert len(pairs) == 1
        assert pairs[0]["chosen_score"] > pairs[0]["rejected_score"]

    def test_construct_preference_pairs_multiple_profiles(self):
        scored = [
            {"profile_id": "p1", "composite_score": 0.9,
             "dialogue": {"turns": [{"role": "agent", "content": "A"}]}},
            {"profile_id": "p1", "composite_score": 0.3,
             "dialogue": {"turns": [{"role": "agent", "content": "B"}]}},
            {"profile_id": "p2", "composite_score": 0.8,
             "dialogue": {"turns": [{"role": "agent", "content": "C"}]}},
            {"profile_id": "p2", "composite_score": 0.5,
             "dialogue": {"turns": [{"role": "agent", "content": "D"}]}},
        ]
        pairs = CoIEvaluator.construct_preference_pairs(scored)
        assert len(pairs) == 2  # One pair per profile


class TestEvaluateAndFilterDummy:
    """Tests for the dummy evaluation pipeline."""

    def test_dummy_evaluation(self):
        dialogues = create_dummy_dialogues(10)
        scored, pairs = evaluate_and_filter_dummy(dialogues)
        assert len(scored) == 10
        assert len(pairs) > 0

    def test_scored_dialogues_have_required_fields(self):
        dialogues = create_dummy_dialogues(5)
        scored, _ = evaluate_and_filter_dummy(dialogues)
        for s in scored:
            assert "composite_score" in s
            assert "dialogue" in s
            assert "instance_scores" in s
            assert "global_score" in s

    def test_preference_pairs_have_required_fields(self):
        dialogues = create_dummy_dialogues(10)
        _, pairs = evaluate_and_filter_dummy(dialogues)
        for p in pairs:
            assert "chosen" in p
            assert "rejected" in p
            assert "chosen_score" in p
            assert "rejected_score" in p
            assert p["chosen_score"] >= p["rejected_score"]
