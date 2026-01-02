"""
Expertise level probing dataset.

Loads conversation data from the opus_expertise folder.
Only loads MEDIUM and EXPERT levels to match llama_gender data distribution.

Each conversation is labeled by the user's expertise level:
- medium: Moderate technical understanding
- expert: Advanced technical discussions

Binary classification: expert=1, medium=0
"""

from pathlib import Path
from typing import Literal


# Data directory
DATA_DIR = Path(__file__).parent / "opus_expertise"


def load_all_conversations(levels: list[str] = None) -> tuple[list[str], list[str]]:
    """
    Load expertise conversations for specified levels.

    Args:
        levels: List of levels to load. If None, loads ["medium", "expert"]

    Returns:
        Tuple of (prompts, level_names) where level_names is "medium" or "expert"
    """
    if levels is None:
        levels = ["medium", "expert"]

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


def make_binary_labels(
    level_names: list[str],
    positive_level: Literal["medium", "expert"] = "expert",
) -> list[int]:
    """
    Create binary labels for expertise classification.

    Args:
        level_names: List of expertise level names
        positive_level: The "positive" class (gets label 1), default is "expert"

    Returns:
        List of binary labels (1 for positive_level, 0 for others)
    """
    return [1 if name == positive_level else 0 for name in level_names]


# Load medium and expert conversations only
_all_prompts, _all_level_names = load_all_conversations(["medium", "expert"])

# Binary classification: expert=1, medium=0
expertise_prompts = _all_prompts
expertise_labels = make_binary_labels(_all_level_names, "expert")

# Raw level names for reference
expertise_level_names = _all_level_names


# Export summary
if __name__ == "__main__":
    print("Expertise Dataset Summary (Medium vs Expert)")
    print("=" * 50)
    print(f"\nTotal conversations: {len(expertise_prompts)}")
    print(f"Expert (1): {sum(expertise_labels)}")
    print(f"Medium (0): {len(expertise_labels) - sum(expertise_labels)}")

    print(f"\nLevel distribution:")
    for level in ["medium", "expert"]:
        count = sum(1 for n in expertise_level_names if n == level)
        print(f"  {level}: {count}")

    print(f"\nSample medium conversation (first 200 chars):")
    idx = expertise_level_names.index("medium")
    print(f"  {expertise_prompts[idx][:200]}...")

    print(f"\nSample expert conversation (first 200 chars):")
    idx = expertise_level_names.index("expert")
    print(f"  {expertise_prompts[idx][:200]}...")