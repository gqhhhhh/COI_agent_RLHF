"""
Module 2: CoI (Chain-of-Intention) Evaluator for dialogue quality assessment.

Implements:
- Instance-level evaluation: Style Similarity and Result F1 via LLM-as-a-Judge
- Global-level evaluation: KL/JS divergence of intent transition matrices
- Preference data construction for Reward Model training
"""

import logging
import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.module2_coi_evaluator.intent_classifier import (
    INTENT_CATEGORIES,
    classify_intent_rule_based,
)
from src.utils.data_utils import save_jsonl

logger = logging.getLogger(__name__)


@dataclass
class CoIEvaluatorConfig:
    """Configuration for the CoI Evaluator."""
    model_name: str = "Qwen/Qwen2.5-7B"
    device: str = "auto"
    style_sim_weight: float = 0.3
    result_f1_weight: float = 0.3
    coi_kl_weight: float = 0.4


STYLE_SIM_PROMPT = """You are an expert dialogue evaluator.
Compare the following two dialogues and rate their STYLE SIMILARITY on a scale of 0.0 to 1.0.
Style includes: tone, formality, sentence structure, and language patterns.

Dialogue A:
{dialogue_a}

Dialogue B:
{dialogue_b}

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.75"). Do not explain."""


RESULT_F1_PROMPT = """You are an expert dialogue evaluator.
Given the following dialogue, evaluate whether the outcome is consistent with the expected outcome.

Dialogue:
{dialogue}

Expected outcome: {expected_outcome}
Actual outcome: {actual_outcome}

Rate the RESULT CONSISTENCY (F1 score) on a scale of 0.0 to 1.0.
Consider: Was the dialogue's conclusion logically consistent? Did the intent flow make sense?

Respond with ONLY a number between 0.0 and 1.0 (e.g., "0.80"). Do not explain."""


class CoIEvaluator:
    """
    Chain-of-Intention Evaluator for multi-dimensional dialogue assessment.

    Evaluates synthetic dialogues at both instance and global levels,
    and constructs preference data for reward model training.
    """

    def __init__(self, config: CoIEvaluatorConfig | None = None):
        self.config = config or CoIEvaluatorConfig()
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load the LLM for evaluation (LLM-as-a-Judge)."""
        logger.info("Loading evaluator model: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def _llm_judge(self, prompt: str) -> float:
        """Call LLM-as-a-Judge and parse a float score."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=16,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return self._parse_score(raw)

    @staticmethod
    def _parse_score(raw: str) -> float:
        """Extract a float score from model output."""
        try:
            # Try to find a float in the output
            for token in raw.split():
                try:
                    score = float(token)
                    return max(0.0, min(1.0, score))
                except ValueError:
                    continue
        except Exception:
            pass
        return 0.5  # default fallback

    # ----------------------------------------------------------------
    # Instance-Level Evaluation
    # ----------------------------------------------------------------

    def evaluate_style_similarity(
        self, dialogue_a: dict, dialogue_b: dict
    ) -> float:
        """
        Evaluate style similarity between two dialogues using LLM-as-a-Judge.

        Args:
            dialogue_a: First dialogue dict with 'turns'.
            dialogue_b: Second dialogue dict with 'turns'.

        Returns:
            Style similarity score in [0, 1].
        """
        text_a = self._dialogue_to_text(dialogue_a)
        text_b = self._dialogue_to_text(dialogue_b)
        prompt = STYLE_SIM_PROMPT.format(dialogue_a=text_a, dialogue_b=text_b)
        return self._llm_judge(prompt)

    def evaluate_result_f1(
        self, dialogue: dict, expected_outcome: str = "success"
    ) -> float:
        """
        Evaluate result consistency (F1) of a dialogue using LLM-as-a-Judge.

        Args:
            dialogue: Dialogue dict with 'turns' and 'outcome'.
            expected_outcome: The expected outcome label.

        Returns:
            Result F1 score in [0, 1].
        """
        text = self._dialogue_to_text(dialogue)
        actual_outcome = dialogue.get("outcome", "unknown")
        prompt = RESULT_F1_PROMPT.format(
            dialogue=text,
            expected_outcome=expected_outcome,
            actual_outcome=actual_outcome,
        )
        return self._llm_judge(prompt)

    def evaluate_instance(
        self,
        dialogue: dict,
        reference_dialogue: dict | None = None,
        expected_outcome: str = "success",
    ) -> dict[str, float]:
        """
        Perform instance-level evaluation combining style similarity and result F1.

        Args:
            dialogue: The dialogue to evaluate.
            reference_dialogue: A reference dialogue for style comparison.
            expected_outcome: Expected outcome for result F1.

        Returns:
            Dict with 'style_sim', 'result_f1', and 'instance_score'.
        """
        style_sim = 0.5  # default if no reference
        if reference_dialogue is not None:
            style_sim = self.evaluate_style_similarity(dialogue, reference_dialogue)

        result_f1 = self.evaluate_result_f1(dialogue, expected_outcome)

        instance_score = (
            self.config.style_sim_weight * style_sim
            + self.config.result_f1_weight * result_f1
        )
        return {
            "style_sim": style_sim,
            "result_f1": result_f1,
            "instance_score": instance_score,
        }

    # ----------------------------------------------------------------
    # Global-Level Evaluation
    # ----------------------------------------------------------------

    @staticmethod
    def build_transition_matrix(
        intent_sequences: list[list[str]],
    ) -> np.ndarray:
        """
        Build a CoI transition probability matrix from intent sequences.

        Args:
            intent_sequences: List of intent label sequences.

        Returns:
            Transition probability matrix of shape (num_intents, num_intents).
        """
        n = len(INTENT_CATEGORIES)
        idx_map = {cat: i for i, cat in enumerate(INTENT_CATEGORIES)}
        count_matrix = np.zeros((n, n), dtype=np.float64)

        for seq in intent_sequences:
            for i in range(len(seq) - 1):
                from_idx = idx_map.get(seq[i])
                to_idx = idx_map.get(seq[i + 1])
                if from_idx is not None and to_idx is not None:
                    count_matrix[from_idx, to_idx] += 1

        # Normalize rows to probabilities (add smoothing to avoid zeros)
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        # Laplace smoothing
        count_matrix += 1e-10
        row_sums = count_matrix.sum(axis=1, keepdims=True)
        transition_matrix = count_matrix / row_sums

        return transition_matrix

    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute KL divergence KL(P || Q) between two probability matrices.

        Args:
            p: Reference distribution matrix.
            q: Approximating distribution matrix.

        Returns:
            KL divergence (scalar, averaged over rows).
        """
        # Ensure no zeros
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        # Re-normalize
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)

        kl_per_row = np.sum(p * np.log(p / q), axis=1)
        return float(np.mean(kl_per_row))

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Compute JS divergence between two probability matrices.

        JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M), where M = 0.5*(P+Q)

        Args:
            p: First distribution matrix.
            q: Second distribution matrix.

        Returns:
            JS divergence (scalar).
        """
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        p = p / p.sum(axis=1, keepdims=True)
        q = q / q.sum(axis=1, keepdims=True)

        m = 0.5 * (p + q)
        return float(
            0.5 * CoIEvaluator.kl_divergence(p, m)
            + 0.5 * CoIEvaluator.kl_divergence(q, m)
        )

    def evaluate_global(
        self,
        synthetic_intents: list[list[str]],
        real_intents: list[list[str]],
    ) -> dict[str, float]:
        """
        Perform global-level evaluation: compare CoI transition matrices.

        Args:
            synthetic_intents: Intent sequences from synthetic dialogues.
            real_intents: Intent sequences from real dialogues.

        Returns:
            Dict with 'kl_divergence', 'js_divergence', and 'global_score'.
        """
        p_real = self.build_transition_matrix(real_intents)
        q_synth = self.build_transition_matrix(synthetic_intents)

        kl = self.kl_divergence(p_real, q_synth)
        js = self.js_divergence(p_real, q_synth)

        # Lower divergence is better; convert to a score in [0, 1]
        global_score = math.exp(-js)  # e^(-JS) gives high score for low divergence

        return {
            "kl_divergence": kl,
            "js_divergence": js,
            "global_score": global_score,
        }

    # ----------------------------------------------------------------
    # Preference Data Construction
    # ----------------------------------------------------------------

    def compute_composite_score(
        self,
        instance_scores: dict[str, float],
        global_score: float,
    ) -> float:
        """
        Compute composite score combining instance and global evaluations.

        Args:
            instance_scores: Dict with 'style_sim' and 'result_f1'.
            global_score: Global-level score.

        Returns:
            Composite score in [0, 1].
        """
        instance_part = (
            self.config.style_sim_weight * instance_scores.get("style_sim", 0.5)
            + self.config.result_f1_weight * instance_scores.get("result_f1", 0.5)
        )
        global_part = self.config.coi_kl_weight * global_score
        return instance_part + global_part

    @staticmethod
    def construct_preference_pairs(
        scored_dialogues: list[dict],
    ) -> list[dict]:
        """
        Construct preference pairs from scored dialogues.

        Groups dialogues by profile_id (context), and for each pair,
        the higher-scored dialogue becomes 'chosen' and the lower becomes 'rejected'.

        Args:
            scored_dialogues: List of dicts with 'dialogue', 'composite_score', etc.

        Returns:
            List of preference pair dicts with 'chosen' and 'rejected'.
        """
        from collections import defaultdict

        grouped = defaultdict(list)
        for item in scored_dialogues:
            profile_id = item.get("profile_id", "default")
            grouped[profile_id].append(item)

        pairs = []
        for profile_id, items in grouped.items():
            items_sorted = sorted(
                items, key=lambda x: x["composite_score"], reverse=True
            )
            for i in range(len(items_sorted)):
                for j in range(i + 1, len(items_sorted)):
                    chosen = items_sorted[i]
                    rejected = items_sorted[j]

                    # Build the context (first few turns shared between both)
                    chosen_turns = chosen["dialogue"].get("turns", [])
                    rejected_turns = rejected["dialogue"].get("turns", [])

                    # Format as chat messages
                    chosen_messages = []
                    for t in chosen_turns:
                        role = "assistant" if t["role"] == "agent" else "user"
                        chosen_messages.append(
                            {"role": role, "content": t["content"]}
                        )

                    rejected_messages = []
                    for t in rejected_turns:
                        role = "assistant" if t["role"] == "agent" else "user"
                        rejected_messages.append(
                            {"role": role, "content": t["content"]}
                        )

                    pairs.append({
                        "profile_id": profile_id,
                        "chosen": chosen_messages,
                        "rejected": rejected_messages,
                        "chosen_score": chosen["composite_score"],
                        "rejected_score": rejected["composite_score"],
                    })
        return pairs

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _dialogue_to_text(dialogue: dict) -> str:
        """Convert dialogue turns to readable text."""
        lines = []
        for turn in dialogue.get("turns", []):
            lines.append(f"{turn['role']}: {turn['content']}")
        return "\n".join(lines)


def evaluate_and_filter_dummy(dialogues: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Evaluate dialogues using rule-based intent classification (no model needed).
    Returns scored dialogues and preference pairs.

    This is used for testing the pipeline without GPU.
    """
    evaluator = CoIEvaluator()

    # Classify intents using rule-based classifier
    all_intents = []
    for d in dialogues:
        intents = [classify_intent_rule_based(t) for t in d.get("turns", [])]
        all_intents.append(intents)

    # Use half as "real" and half as "synthetic" for global evaluation
    mid = max(1, len(all_intents) // 2)
    real_intents = all_intents[:mid]
    synth_intents = all_intents[mid:]

    if not synth_intents:
        synth_intents = real_intents

    global_eval = evaluator.evaluate_global(synth_intents, real_intents)
    global_score = global_eval["global_score"]

    scored_dialogues = []
    for i, dialogue in enumerate(dialogues):
        # Simple instance-level scoring based on outcome
        outcome = dialogue.get("outcome", "max_turns_reached")
        if outcome == "success":
            instance_scores = {"style_sim": 0.8, "result_f1": 0.9}
        elif outcome == "rejection":
            instance_scores = {"style_sim": 0.6, "result_f1": 0.4}
        else:
            instance_scores = {"style_sim": 0.5, "result_f1": 0.5}

        composite = evaluator.compute_composite_score(instance_scores, global_score)

        scored_dialogues.append({
            "dialogue": dialogue,
            "profile_id": dialogue.get("profile_id", "default"),
            "instance_scores": instance_scores,
            "global_score": global_score,
            "composite_score": composite,
        })

    preference_pairs = evaluator.construct_preference_pairs(scored_dialogues)
    return scored_dialogues, preference_pairs
