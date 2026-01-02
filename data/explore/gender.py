"""
Gender probing dataset.

Loads conversation data from the llama_gender folder.
Each conversation is labeled by the simulated user's gender:
- female: Conversations with female user
- male: Conversations with male user

Binary classification: female=1, male=0
"""

from pathlib import Path


# Data directory
DATA_DIR = Path(__file__).parent / "llama_gender"


def load_all_conversations() -> tuple[list[str], list[int]]:
    """
    Load all gender conversations.

    Returns:
        Tuple of (prompts, labels) where labels are 1 for female, 0 for male
    """
    prompts = []
    labels = []

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Gender data directory not found: {DATA_DIR}")

    files = list(DATA_DIR.glob("*.txt"))

    # Group files by gender
    female_files = []
    male_files = []

    for f in files:
        if "_gender_female.txt" in f.name:
            female_files.append(f)
        elif "_gender_male.txt" in f.name:
            male_files.append(f)

    # Sort files for consistent ordering
    female_files.sort(key=lambda x: x.name)
    male_files.sort(key=lambda x: x.name)

    # Load female conversations (label=1)
    for filepath in female_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            prompts.append(content)
            labels.append(1)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")

    # Load male conversations (label=0)
    for filepath in male_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            prompts.append(content)
            labels.append(0)
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")

    return prompts, labels


# Load all conversations once
gender_prompts, gender_labels = load_all_conversations()


# Export summary
if __name__ == "__main__":
    print("Gender Dataset Summary")
    print("=" * 50)
    print(f"\nTotal conversations: {len(gender_prompts)}")
    print(f"Female (1): {sum(gender_labels)}")
    print(f"Male (0): {len(gender_labels) - sum(gender_labels)}")

    print(f"\nSample female conversation (first 200 chars):")
    idx = gender_labels.index(1)
    print(f"  {gender_prompts[idx][:200]}...")

    print(f"\nSample male conversation (first 200 chars):")
    idx = gender_labels.index(0)
    print(f"  {gender_prompts[idx][:200]}...")