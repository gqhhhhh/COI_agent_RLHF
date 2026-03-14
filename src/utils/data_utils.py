"""
Utility functions for data loading, saving, and format conversion.
"""

import json
import os
from typing import Any


def save_jsonl(data: list[dict], output_path: str) -> None:
    """Save a list of dicts to a JSONL file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(input_path: str) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def format_dialogue_for_sft(dialogue: dict) -> list[dict[str, str]]:
    """
    Convert a dialogue record to the chat format expected by SFTTrainer.

    Returns a list of {"role": ..., "content": ...} messages.
    """
    messages = []
    for turn in dialogue.get("turns", []):
        role = "assistant" if turn["role"] == "agent" else "user"
        messages.append({"role": role, "content": turn["content"]})
    return messages


def format_preference_pair(
    context: list[dict[str, str]],
    chosen_response: str,
    rejected_response: str,
) -> dict[str, Any]:
    """
    Format a preference pair for Reward Model training.

    Returns a dict with 'chosen' and 'rejected' conversation lists.
    """
    chosen = context + [{"role": "assistant", "content": chosen_response}]
    rejected = context + [{"role": "assistant", "content": rejected_response}]
    return {"chosen": chosen, "rejected": rejected}
