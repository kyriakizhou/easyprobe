"""
Factuality probing datasets.

Loads factuality data from JSON files in the datasets/ folder.

Available datasets:
- factuality_large: 800 prompts (400 true, 400 false) - basic factuality
- factuality_small: 200 prompts (100 true, 100 false) - for testing
- topics_large: 1000 prompts (500 math, 500 climate) - topic classification
- topics_small: 200 prompts (100 math, 100 climate) - for testing
- factuality_topic_shared: 800 prompts with 2 labels (factuality + topic) - multi-label
- factuality_uniform: 800 prompts with uniform sentence structure
- factuality_diverse: 800 prompts with diverse sentence structures
- factuality_extended: 1600 prompts (uniform + diverse combined)
"""

from pathlib import Path
from .json_loader import load_json_dataset

# Data directory
DATASETS_DIR = Path(__file__).parent / "datasets"


# Single-label datasets
fact_prompts_large, fact_labels_large = load_json_dataset(
    DATASETS_DIR / "factuality_large.json"
)

fact_prompts_small, fact_labels_small = load_json_dataset(
    DATASETS_DIR / "factuality_small.json"
)

topics_prompts_large, topics_labels_large = load_json_dataset(
    DATASETS_DIR / "topics_large.json"
)

topics_prompts_small, topics_labels_small = load_json_dataset(
    DATASETS_DIR / "topics_small.json"
)

# Extended datasets
scenario6_prompts, scenario6_labels = load_json_dataset(
    DATASETS_DIR / "factuality_uniform.json"
)

scenario7_prompts, scenario7_labels = load_json_dataset(
    DATASETS_DIR / "factuality_diverse.json"
)

extended_prompts, extended_labels = load_json_dataset(
    DATASETS_DIR / "factuality_extended.json"
)

# Multi-label dataset (scenario 4: shared prompts with factuality + topic labels)
# label1 = factuality (1=true, 0=false)
# label2 = topic (1=math, 0=climate)
scenario4_prompts, scenario4_labels_dict = load_json_dataset(
    DATASETS_DIR / "factuality_topic_shared.json",
    multi_label=True
)
scenario4_factuality_labels = scenario4_labels_dict["label1"]
scenario4_topic_labels = scenario4_labels_dict["label2"]


# Export summary
if __name__ == "__main__":
    print("Factuality Datasets Summary")
    print("=" * 60)

    print(f"\nfactuality_large: {len(fact_prompts_large)} prompts")
    print(f"  True (1): {sum(fact_labels_large)}")
    print(f"  False (0): {len(fact_labels_large) - sum(fact_labels_large)}")

    print(f"\nfactuality_small: {len(fact_prompts_small)} prompts")
    print(f"  True (1): {sum(fact_labels_small)}")
    print(f"  False (0): {len(fact_labels_small) - sum(fact_labels_small)}")

    print(f"\ntopics_large: {len(topics_prompts_large)} prompts")
    print(f"  Climate (1): {sum(topics_labels_large)}")
    print(f"  Math (0): {len(topics_labels_large) - sum(topics_labels_large)}")

    print(f"\nfactuality_topic_shared (multi-label): {len(scenario4_prompts)} prompts")
    print(f"  Factuality - True (1): {sum(scenario4_factuality_labels)}")
    print(f"  Topic - Math (1): {sum(scenario4_topic_labels)}")

    print(f"\nfactuality_uniform: {len(scenario6_prompts)} prompts")
    print(f"\nfactuality_diverse: {len(scenario7_prompts)} prompts")
    print(f"\nfactuality_extended: {len(extended_prompts)} prompts")
