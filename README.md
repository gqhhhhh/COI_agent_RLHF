# COI_agent_RLHF

A reproduction of the **SimRPD** paper ([arXiv:2601.02871](https://arxiv.org/html/2601.02871v1)) — an optimization framework for proactive dialogue agents in recruitment scenarios using **Chain-of-Intention (CoI)** evaluation and **RLHF** alignment.

## Overview

This project addresses **unstable intent tracking in multi-turn proactive dialogue** (e.g., recruitment guidance) through a four-stage pipeline:

1. **User Simulator** — synthesizes multi-turn recruitment dialogues using LLM-based simulation
2. **CoI Evaluator** — multi-dimensional quality assessment with instance-level and global-level metrics
3. **SFT & Reward Model Training** — supervised fine-tuning and preference-based reward model training with LoRA
4. **PPO Alignment** — reinforcement learning optimization with composite rewards (rule-based + model-based)

**Base Model:** Qwen2.5-7B  
**Tech Stack:** Python, PyTorch, HuggingFace Transformers, TRL, PEFT (LoRA)

## Project Structure

```
COI_agent_RLHF/
├── configs/
│   └── default_config.yaml        # Default hyperparameters and settings
├── data/
│   └── sample/                    # Sample data directory
├── scripts/
│   ├── run_simulator.py           # Run Module 1: dialogue generation
│   ├── run_evaluation.py          # Run Module 2: CoI evaluation & filtering
│   ├── run_sft.py                 # Run Module 3a: SFT training
│   ├── run_rm.py                  # Run Module 3b: Reward Model training
│   ├── run_ppo.py                 # Run Module 4: PPO training
│   └── run_dummy_test.py          # End-to-end pipeline validation (no GPU)
├── src/
│   ├── module1_simulator/
│   │   ├── user_simulator.py      # Multi-turn dialogue simulator
│   │   └── profiles.py            # Job seeker profile definitions
│   ├── module2_coi_evaluator/
│   │   ├── coi_evaluator.py       # CoI evaluation, filtering, preference construction
│   │   └── intent_classifier.py   # Zero-shot and rule-based intent classification
│   ├── module3_training/
│   │   ├── sft_trainer.py         # SFT with LoRA on Qwen2.5-7B
│   │   └── rm_trainer.py          # Reward Model training on preference pairs
│   ├── module4_ppo/
│   │   └── ppo_trainer.py         # PPO with composite reward (R_rule + R_model)
│   └── utils/
│       └── data_utils.py          # JSONL I/O, data formatting utilities
├── tests/
│   ├── test_simulator.py          # Tests for Module 1
│   ├── test_coi_evaluator.py      # Tests for Module 2
│   └── test_training.py           # Tests for Modules 3 & 4 data flow
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start: Dummy Pipeline Test

Validate the entire data flow without GPU or model download:

```bash
python scripts/run_dummy_test.py
```

This generates 10 dummy dialogues, runs evaluation, constructs preference pairs, and verifies all data formats.

## Full Pipeline

### Step 1: Generate Dialogues (Module 1)

```bash
# With model (requires GPU + Qwen2.5-7B)
python scripts/run_simulator.py --num_dialogues 100 --output data/output/dialogues.jsonl

# Dummy mode (no GPU needed)
python scripts/run_simulator.py --dummy --num_dialogues 100 --output data/output/dialogues.jsonl
```

**Output format** (JSONL):
```json
{
  "dialogue_id": "uuid",
  "profile_id": "profile_001",
  "turns": [
    {"role": "agent", "content": "Hello! How can I help you?"},
    {"role": "user", "content": "I'm looking for a Senior Engineer role."}
  ],
  "outcome": "success",
  "num_turns": 6
}
```

### Step 2: CoI Evaluation & Filtering (Module 2)

```bash
python scripts/run_evaluation.py \
  --input data/output/dialogues.jsonl \
  --output data/output/preference_data.jsonl \
  --dummy
```

The evaluator:
- Classifies each turn into one of **9 intent categories** (information_inquiry, job_recommendation, requirement_clarification, salary_negotiation, benefit_discussion, hesitation, explicit_rejection, successful_conversion, follow_up)
- Computes **instance-level** metrics: Style Similarity and Result F1 (via LLM-as-a-Judge)
- Computes **global-level** metrics: KL/JS divergence of intent transition matrices
- Constructs **preference pairs** (chosen/rejected) for RM training

### Step 3a: SFT Training (Module 3)

```bash
python scripts/run_sft.py \
  --input data/output/dialogues.jsonl \
  --model_name Qwen/Qwen2.5-7B \
  --output models/sft \
  --epochs 3
```

### Step 3b: Reward Model Training (Module 3)

```bash
python scripts/run_rm.py \
  --input data/output/preference_data.jsonl \
  --model_name Qwen/Qwen2.5-7B \
  --output models/reward_model \
  --epochs 1
```

### Step 4: PPO Training (Module 4)

```bash
python scripts/run_ppo.py \
  --input data/output/dialogues.jsonl \
  --sft_model models/sft \
  --rm_model models/reward_model \
  --output models/ppo \
  --alpha 0.3 \
  --beta 0.7
```

The PPO reward is computed as:

$$R_t = \alpha \cdot R_{rule} + \beta \cdot R_{model}$$

Where:
- **R_rule**: Rule-based safety reward (penalizes repetition, toxicity, empty responses)
- **R_model**: Preference reward from the trained Reward Model
- **KL penalty**: Prevents policy from deviating too far from the SFT base model

## Running Tests

```bash
python -m pytest tests/ -v
```

## Key Design Decisions

- **Lazy imports**: GPU dependencies (torch, transformers, trl, peft) are imported only when needed, allowing tests and dummy runs without GPU
- **Rule-based fallback**: Intent classifier has both LLM-based and rule-based modes for flexibility
- **Modular architecture**: Each pipeline stage is independently testable with its own config
- **JSONL format**: All intermediate data uses JSONL for streaming-friendly I/O

## References

- [SimRPD: A Simulated Recruitment Proactive Dialogue System](https://arxiv.org/html/2601.02871v1)
- [Qwen2.5](https://huggingface.co/Qwen/Qwen2.5-7B)
- [TRL: Transformer Reinforcement Learning](https://huggingface.co/docs/trl)
- [PEFT: Parameter-Efficient Fine-Tuning](https://huggingface.co/docs/peft)

