"""
Module 1: User Simulator for multi-turn dialogue data synthesis.

Generates synthetic recruitment dialogues between an Agent (recruiter)
and a Simulator (job seeker) based on predefined profiles.
Uses Qwen2.5-7B for both Agent and Simulator response generation.
"""

import json
import logging
import os
import uuid
from dataclasses import dataclass, field

from src.module1_simulator.profiles import SAMPLE_PROFILES, get_profile_prompt
from src.utils.data_utils import save_jsonl

logger = logging.getLogger(__name__)

END_KEYWORDS_SUCCESS = ["accept", "I'll take it", "sounds great", "let's proceed",
                        "deal", "I'm interested", "sign me up"]
END_KEYWORDS_REJECT = ["not interested", "no thanks", "I'll pass", "decline",
                       "reject", "too low", "can't accept"]


@dataclass
class SimulatorConfig:
    """Configuration for the User Simulator."""
    model_name: str = "Qwen/Qwen2.5-7B"
    max_turns: int = 10
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"
    num_dialogues: int = 100
    output_path: str = "data/output/simulated_dialogues.jsonl"


AGENT_SYSTEM_PROMPT = (
    "You are an experienced recruitment agent. Your goal is to guide the "
    "conversation to understand the candidate's background, match them with "
    "suitable positions, and ultimately achieve a successful placement. "
    "Be professional, persuasive, and attentive to the candidate's needs. "
    "Start by greeting the candidate and asking about their career goals."
)


class UserSimulator:
    """
    Multi-turn dialogue simulator that generates synthetic recruitment
    conversations between an Agent and a User (job seeker).
    """

    def __init__(self, config: SimulatorConfig | None = None):
        self.config = config or SimulatorConfig()
        self.tokenizer = None
        self.model = None

    def load_model(self) -> None:
        """Load the LLM for dialogue generation."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading model: %s", self.config.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="left",
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
        logger.info("Model loaded successfully.")

    def _generate_response(self, messages: list[dict[str, str]]) -> str:
        """Generate a single response given a conversation history."""
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @staticmethod
    def _check_end_condition(response: str) -> str | None:
        """
        Check if the response triggers an end condition.

        Returns:
            "success" if a success keyword is found,
            "rejection" if a rejection keyword is found,
            None otherwise.
        """
        response_lower = response.lower()
        # Check rejection before success to handle cases like "can't accept"
        for kw in END_KEYWORDS_REJECT:
            if kw.lower() in response_lower:
                return "rejection"
        for kw in END_KEYWORDS_SUCCESS:
            if kw.lower() in response_lower:
                return "success"
        return None

    def generate_dialogue(self, profile: dict) -> dict:
        """
        Generate a single multi-turn dialogue for a given job seeker profile.

        The loop:
        1. Agent sends a message (using agent system prompt + history)
        2. Simulator replies (using profile prompt + history)
        3. Check end conditions; repeat or stop.

        Returns:
            A dict with dialogue_id, turns, and outcome.
        """
        dialogue_id = str(uuid.uuid4())
        user_system_prompt = get_profile_prompt(profile)
        turns: list[dict[str, str]] = []
        outcome = "max_turns_reached"

        # Build agent messages and user messages separately
        agent_messages = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        user_messages = [{"role": "system", "content": user_system_prompt}]

        for turn_idx in range(self.config.max_turns):
            # --- Agent turn ---
            agent_response = self._generate_response(agent_messages)
            turns.append({"role": "agent", "content": agent_response})

            # Update both conversation histories
            agent_messages.append({"role": "assistant", "content": agent_response})
            user_messages.append({"role": "user", "content": agent_response})

            # Check if agent triggers end condition
            end = self._check_end_condition(agent_response)
            if end:
                outcome = end
                break

            # --- User (Simulator) turn ---
            user_response = self._generate_response(user_messages)
            turns.append({"role": "user", "content": user_response})

            # Update both conversation histories
            agent_messages.append({"role": "user", "content": user_response})
            user_messages.append({"role": "assistant", "content": user_response})

            # Check if user triggers end condition
            end = self._check_end_condition(user_response)
            if end:
                outcome = end
                break

        return {
            "dialogue_id": dialogue_id,
            "profile_id": profile.get("id", "unknown"),
            "turns": turns,
            "outcome": outcome,
            "num_turns": len(turns),
        }

    def generate_dataset(
        self, profiles: list[dict] | None = None, num_dialogues: int | None = None
    ) -> list[dict]:
        """
        Generate a dataset of dialogues across multiple profiles.

        Args:
            profiles: List of job seeker profiles. Defaults to SAMPLE_PROFILES.
            num_dialogues: Total number of dialogues to generate.

        Returns:
            List of dialogue dicts.
        """
        if profiles is None:
            profiles = SAMPLE_PROFILES
        if num_dialogues is None:
            num_dialogues = self.config.num_dialogues

        dialogues = []
        for i in range(num_dialogues):
            profile = profiles[i % len(profiles)]
            logger.info(
                "Generating dialogue %d/%d (profile: %s)",
                i + 1, num_dialogues, profile.get("id", "unknown"),
            )
            dialogue = self.generate_dialogue(profile)
            dialogues.append(dialogue)

        return dialogues

    def generate_and_save(
        self,
        profiles: list[dict] | None = None,
        num_dialogues: int | None = None,
        output_path: str | None = None,
    ) -> list[dict]:
        """Generate dialogues and save to JSONL."""
        dialogues = self.generate_dataset(profiles, num_dialogues)
        path = output_path or self.config.output_path
        save_jsonl(dialogues, path)
        logger.info("Saved %d dialogues to %s", len(dialogues), path)
        return dialogues


def create_dummy_dialogues(num_dialogues: int = 10) -> list[dict]:
    """
    Create dummy dialogue data for testing without a GPU/model.

    Returns a list of dialogue dicts in the same format as generate_dialogue().
    """
    dialogues = []
    outcomes = ["success", "rejection", "max_turns_reached"]
    for i in range(num_dialogues):
        profile = SAMPLE_PROFILES[i % len(SAMPLE_PROFILES)]
        dialogue_id = str(uuid.uuid4())
        outcome = outcomes[i % len(outcomes)]
        turns = [
            {"role": "agent", "content": f"Hello! I'm a recruitment consultant. How can I help you today?"},
            {"role": "user", "content": f"Hi, I'm looking for a {profile['desired_role']} position."},
            {"role": "agent", "content": f"Great! Can you tell me about your experience in {profile['current_role']}?"},
            {"role": "user", "content": f"I have {profile['years_of_experience']} years of experience. My key skills are {', '.join(profile['skills'])}."},
            {"role": "agent", "content": f"We have some positions in {profile['location_preference']}. The salary range is around {profile['salary_expectation']}."},
        ]
        if outcome == "success":
            turns.append({"role": "user", "content": "Sounds great! I'm interested in this position. Let's proceed."})
        elif outcome == "rejection":
            turns.append({"role": "user", "content": "I'm not interested. The offer is too low for my expectations."})
        else:
            turns.append({"role": "user", "content": "Let me think about it and get back to you."})

        dialogues.append({
            "dialogue_id": dialogue_id,
            "profile_id": profile["id"],
            "turns": turns,
            "outcome": outcome,
            "num_turns": len(turns),
        })
    return dialogues
