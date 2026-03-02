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

import logging
from pathlib import Path
from .json_loader import load_json_dataset

logger = logging.getLogger(__name__)

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
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')

    summary = (
        "Factuality Datasets Summary\n"
        f"{'=' * 60}\n"
        f"factuality_large: {len(fact_prompts_large)} prompts (True: {sum(fact_labels_large)}, False: {len(fact_labels_large) - sum(fact_labels_large)})\n"
        f"factuality_small: {len(fact_prompts_small)} prompts (True: {sum(fact_labels_small)}, False: {len(fact_labels_small) - sum(fact_labels_small)})\n"
        f"topics_large: {len(topics_prompts_large)} prompts (Climate: {sum(topics_labels_large)}, Math: {len(topics_labels_large) - sum(topics_labels_large)})\n"
        f"factuality_topic_shared (multi-label): {len(scenario4_prompts)} prompts (Factuality True: {sum(scenario4_factuality_labels)}, Topic Math: {sum(scenario4_topic_labels)})\n"
        f"factuality_uniform: {len(scenario6_prompts)} prompts\n"
        f"factuality_diverse: {len(scenario7_prompts)} prompts\n"
        f"factuality_extended: {len(extended_prompts)} prompts"
    )
    logger.info(summary)
