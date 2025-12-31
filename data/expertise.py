"""
Expertise level probing dataset.

Loads conversation data from the opus_expertise folder.
Each conversation is labeled by the user's expertise level:
- novice: Basic questions, unfamiliar with technical concepts
- medium: Moderate technical understanding
- expert: Advanced technical discussions

Uses one-vs-rest strategy for binary classification:
- expertise_novice_prompts, expertise_novice_labels: novice=1, others=0
- expertise_medium_prompts, expertise_medium_labels: medium=1, others=0
- expertise_expert_prompts, expertise_expert_labels: expert=1, others=0
"""

import os
from pathlib import Path
from typing import Literal


# Data directory
DATA_DIR = Path(__file__).parent / "opus_expertise"


def load_all_conversations() -> tuple[list[str], list[str]]:
    """
    Load all expertise conversations.

    Returns:
        Tuple of (prompts, level_names) where level_names is "novice", "medium", or "expert"
    """
    levels = ["novice", "medium", "expert"]

    prompts = []
    level_names = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Expertise data directory not found: {DATA_DIR}")

    files = list(DATA_DIR.glob("*.txt"))

    # Group files by level
    files_by_level: dict[str, list[Path]] = {level: [] for level in levels}

    for f in files:
        for level in levels:
            if f"_expertise_{level}.txt" in f.name:
                files_by_level[level].append(f)
                break

    # Sort and load each level's files
    for level in levels:
        files_by_level[level].sort(key=lambda x: x.name)
        for filepath in files_by_level[level]:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                prompts.append(content)
                level_names.append(level)
            except Exception as e:
                print(f"Warning: Could not read {filepath}: {e}")

    return prompts, level_names


def make_one_vs_rest_labels(
    level_names: list[str],
    target_level: Literal["novice", "medium", "expert"],
) -> list[int]:
    """
    Create one-vs-rest binary labels.

    Args:
        level_names: List of expertise level names
        target_level: The "positive" class (gets label 1)

    Returns:
        List of binary labels (1 for target_level, 0 for others)
    """
    return [1 if name == target_level else 0 for name in level_names]


# Load all conversations once
_all_prompts, _all_level_names = load_all_conversations()

# One-vs-rest datasets
# All prompts are the same, only labels differ
expertise_prompts = _all_prompts

# Novice vs rest (novice=1, medium+expert=0)
expertise_novice_labels = make_one_vs_rest_labels(_all_level_names, "novice")

# Medium vs rest (medium=1, novice+expert=0)
expertise_medium_labels = make_one_vs_rest_labels(_all_level_names, "medium")

# Expert vs rest (expert=1, novice+medium=0)
expertise_expert_labels = make_one_vs_rest_labels(_all_level_names, "expert")

# Raw level names for reference
expertise_level_names = _all_level_names


# Export summary
if __name__ == "__main__":
    print("Expertise Dataset Summary (One-vs-Rest)")
    print("=" * 50)
    print(f"\nTotal conversations: {len(expertise_prompts)}")

    print(f"\nNovice vs Rest:")
    print(f"  Novice (1): {sum(expertise_novice_labels)}")
    print(f"  Rest (0): {len(expertise_novice_labels) - sum(expertise_novice_labels)}")

    print(f"\nMedium vs Rest:")
    print(f"  Medium (1): {sum(expertise_medium_labels)}")
    print(f"  Rest (0): {len(expertise_medium_labels) - sum(expertise_medium_labels)}")

    print(f"\nExpert vs Rest:")
    print(f"  Expert (1): {sum(expertise_expert_labels)}")
    print(f"  Rest (0): {len(expertise_expert_labels) - sum(expertise_expert_labels)}")

    print(f"\nLevel distribution:")
    for level in ["novice", "medium", "expert"]:
        count = sum(1 for n in expertise_level_names if n == level)
        print(f"  {level}: {count}")

    print(f"\nSample novice conversation (first 200 chars):")
    idx = expertise_level_names.index("novice")
    print(f"  {expertise_prompts[idx][:200]}...")

    print(f"\nSample expert conversation (first 200 chars):")
    idx = expertise_level_names.index("expert")
    print(f"  {expertise_prompts[idx][:200]}...")