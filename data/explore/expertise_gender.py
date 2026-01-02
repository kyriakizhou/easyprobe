"""
Expertise-Gender probing dataset.

Loads conversation data from the opus_expertise_gender folder.
Each conversation is labeled by both gender and expertise level:
- Gender: female=1, male=0
- Expertise: expert=1, medium=0

File naming: conversation_X_{gender}_{expertise}.txt
Only loads MEDIUM and EXPERT levels (excludes novice) to match llama_gender distribution.

Provides separate label sets for multi-feature probing:
- expertise_gender_prompts: All prompts (medium+expert only)
- expertise_gender_gender_labels: female=1, male=0
- expertise_gender_expertise_labels: expert=1, medium=0
"""

from pathlib import Path
from typing import Literal


# Data directory
DATA_DIR = Path(__file__).parent / "opus_expertise_gender"


def load_all_conversations(
    expertise_levels: list[str] = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Load expertise-gender conversations for specified expertise levels.

    Args:
        expertise_levels: List of expertise levels to load. If None, loads ["medium", "expert"]

    Returns:
        Tuple of (prompts, genders, expertise_levels) where:
        - genders is "female" or "male"
        - expertise_levels is "medium" or "expert"
    """
    if expertise_levels is None:
        expertise_levels = ["medium", "expert"]

    prompts = []
    genders = []
    expertises = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Expertise-gender data directory not found: {DATA_DIR}")

    files = list(DATA_DIR.glob("*.txt"))

    # Filter and group files by gender and expertise
    valid_files = []
    for f in files:
        # Parse filename: conversation_X_{gender}_{expertise}.txt
        name = f.stem  # e.g., "conversation_0_female_expert"
        parts = name.split("_")
        if len(parts) >= 4:
            gender = parts[-2]  # e.g., "female"
            expertise = parts[-1]  # e.g., "expert"

            # Only include specified expertise levels
            if expertise in expertise_levels and gender in ["female", "male"]:
                valid_files.append((f, gender, expertise))

    # Sort files for consistent ordering
    valid_files.sort(key=lambda x: x[0].name)

    # Load files
    for filepath, gender, expertise in valid_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            prompts.append(content)
            genders.append(gender)
            expertises.append(expertise)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")

    return prompts, genders, expertises


def make_binary_labels(
    names: list[str],
    positive_value: str,
) -> list[int]:
    """
    Create binary labels.

    Args:
        names: List of string values
        positive_value: The "positive" class (gets label 1)

    Returns:
        List of binary labels (1 for positive_value, 0 for others)
    """
    return [1 if name == positive_value else 0 for name in names]


# Load medium and expert conversations only
_all_prompts, _all_genders, _all_expertises = load_all_conversations(["medium", "expert"])

# Shared prompts for multi-feature probing
expertise_gender_prompts = _all_prompts

# Gender labels: female=1, male=0
expertise_gender_gender_labels = make_binary_labels(_all_genders, "female")

# Expertise labels: expert=1, medium=0
expertise_gender_expertise_labels = make_binary_labels(_all_expertises, "expert")

# Raw names for reference
expertise_gender_genders = _all_genders
expertise_gender_expertises = _all_expertises


# Export summary
if __name__ == "__main__":
    print("Expertise-Gender Dataset Summary (Medium + Expert only)")
    print("=" * 60)
    print(f"\nTotal conversations: {len(expertise_gender_prompts)}")

    print(f"\nGender distribution:")
    print(f"  Female (1): {sum(expertise_gender_gender_labels)}")
    print(f"  Male (0): {len(expertise_gender_gender_labels) - sum(expertise_gender_gender_labels)}")

    print(f"\nExpertise distribution:")
    print(f"  Expert (1): {sum(expertise_gender_expertise_labels)}")
    print(f"  Medium (0): {len(expertise_gender_expertise_labels) - sum(expertise_gender_expertise_labels)}")

    print(f"\nCross-tabulation:")
    for gender in ["female", "male"]:
        for expertise in ["medium", "expert"]:
            count = sum(1 for g, e in zip(_all_genders, _all_expertises)
                       if g == gender and e == expertise)
            print(f"  {gender}_{expertise}: {count}")

    print(f"\nSample female-expert conversation (first 200 chars):")
    for i, (g, e) in enumerate(zip(_all_genders, _all_expertises)):
        if g == "female" and e == "expert":
            print(f"  {expertise_gender_prompts[i][:200]}...")
            break

    print(f"\nSample male-medium conversation (first 200 chars):")
    for i, (g, e) in enumerate(zip(_all_genders, _all_expertises)):
        if g == "male" and e == "medium":
            print(f"  {expertise_gender_prompts[i][:200]}...")
            break
