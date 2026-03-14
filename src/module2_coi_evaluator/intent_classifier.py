"""
Zero-shot intent classifier using LLM for labeling dialogue turns.

Classifies each turn in a dialogue into one of 9 predefined intent categories
from the CoI (Chain-of-Intention) framework.
"""

import logging
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

INTENT_CATEGORIES = [
    "information_inquiry",
    "job_recommendation",
    "requirement_clarification",
    "salary_negotiation",
    "benefit_discussion",
    "hesitation",
    "explicit_rejection",
    "successful_conversion",
    "follow_up",
]

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for recruitment dialogues.
Given the following dialogue turn, classify it into exactly ONE of these intent categories:

Categories:
1. information_inquiry - Asking for or providing factual information
2. job_recommendation - Recommending or discussing specific job positions
3. requirement_clarification - Clarifying job requirements or candidate qualifications
4. salary_negotiation - Discussing or negotiating salary and compensation
5. benefit_discussion - Discussing benefits, perks, or working conditions
6. hesitation - Expressing uncertainty or asking for time to think
7. explicit_rejection - Clearly declining an offer or ending the conversation
8. successful_conversion - Accepting an offer or expressing strong interest
9. follow_up - Scheduling next steps, follow-up meetings, or closing remarks

Dialogue context (previous turns):
{context}

Current turn:
Role: {role}
Content: {content}

Respond with ONLY the category name (e.g., "information_inquiry"). Do not explain."""


class IntentClassifier:
    """Zero-shot LLM-based intent classifier for dialogue turns."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load the LLM for classification."""
        logger.info("Loading intent classifier model: %s", self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True,
        )
        self.model.eval()

    def _generate(self, prompt: str) -> str:
        """Generate a response from the model."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def classify_turn(
        self, turn: dict[str, str], context: list[dict[str, str]] | None = None
    ) -> str:
        """
        Classify a single dialogue turn.

        Args:
            turn: Dict with 'role' and 'content'.
            context: Previous turns for context.

        Returns:
            One of the INTENT_CATEGORIES.
        """
        context_str = ""
        if context:
            context_str = "\n".join(
                f"{t['role']}: {t['content']}" for t in context[-4:]
            )

        prompt = INTENT_CLASSIFICATION_PROMPT.format(
            context=context_str or "(start of conversation)",
            role=turn["role"],
            content=turn["content"],
        )
        raw_output = self._generate(prompt)
        return self._parse_intent(raw_output)

    def classify_dialogue(self, dialogue: dict) -> list[str]:
        """
        Classify all turns in a dialogue, returning intent sequence.

        Args:
            dialogue: Dict with 'turns' list.

        Returns:
            List of intent labels, one per turn.
        """
        turns = dialogue.get("turns", [])
        intents = []
        for i, turn in enumerate(turns):
            context = turns[:i]
            intent = self.classify_turn(turn, context)
            intents.append(intent)
        return intents

    @staticmethod
    def _parse_intent(raw_output: str) -> str:
        """Parse model output to extract a valid intent category."""
        cleaned = raw_output.strip().lower().replace(" ", "_")
        # Direct match
        if cleaned in INTENT_CATEGORIES:
            return cleaned
        # Partial match
        for cat in INTENT_CATEGORIES:
            if cat in cleaned:
                return cat
        # Default fallback
        return "information_inquiry"


def classify_intent_rule_based(turn: dict[str, str]) -> str:
    """
    Simple rule-based intent classifier as fallback (no model needed).

    Args:
        turn: Dict with 'role' and 'content'.

    Returns:
        One of the INTENT_CATEGORIES.
    """
    content = turn.get("content", "").lower()

    rejection_kw = ["not interested", "no thanks", "decline", "reject", "too low", "can't accept"]
    success_kw = ["accept", "sounds great", "i'm interested", "let's proceed", "deal", "sign me up"]
    hesitation_kw = ["think about it", "not sure", "maybe", "let me consider", "hesitant"]
    salary_kw = ["salary", "compensation", "pay", "money", "offer", "package"]
    benefit_kw = ["benefit", "remote", "vacation", "insurance", "perk", "work-life"]
    recommend_kw = ["position", "role", "opening", "opportunity", "job"]
    clarify_kw = ["require", "qualification", "experience needed", "what do you need"]
    followup_kw = ["next step", "follow up", "schedule", "interview", "call back"]

    for kw in rejection_kw:
        if kw in content:
            return "explicit_rejection"
    for kw in success_kw:
        if kw in content:
            return "successful_conversion"
    for kw in hesitation_kw:
        if kw in content:
            return "hesitation"
    for kw in salary_kw:
        if kw in content:
            return "salary_negotiation"
    for kw in benefit_kw:
        if kw in content:
            return "benefit_discussion"
    for kw in recommend_kw:
        if kw in content:
            return "job_recommendation"
    for kw in clarify_kw:
        if kw in content:
            return "requirement_clarification"
    for kw in followup_kw:
        if kw in content:
            return "follow_up"

    return "information_inquiry"
