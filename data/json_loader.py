"""
General JSON data loader for easyprobe.

Supports two JSON formats:

1. Single-label format:
   [
     {"prompt": "...", "label": 0},
     {"prompt": "...", "label": 1}
   ]

2. Multi-label format (for shared prompts with multiple features):
   {
     "_metadata": {
       "description": "...",
       "label1": "description of label1",
       "label2": "description of label2"
     },
     "samples": [
       {"prompt": "...", "label1": 0, "label2": 1},
       {"prompt": "...", "label1": 1, "label2": 0}
     ]
   }

Usage:
    # Single-label
    prompts, labels = load_json_dataset("path/to/data.json")

    # Multi-label
    prompts, labels_dict = load_json_dataset("path/to/data.json", multi_label=True)
    # labels_dict = {"label1": [...], "label2": [...]}
"""

import json
import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def load_json_dataset(
    path: Union[str, Path],
    multi_label: bool = False,
) -> Union[tuple[list[str], list[int]], tuple[list[str], dict[str, list[int]]]]:
    """
    Load a JSON dataset.

    Args:
        path: Path to JSON file
        multi_label: If True, returns dict of label lists for multi-feature probing

    Returns:
        If multi_label=False: (prompts, labels)
        If multi_label=True: (prompts, {"label1": labels, "label2": labels, ...})

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON format is invalid
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if multi_label:
        # Multi-label format: {"_metadata": {...}, "samples": [...]}
        if isinstance(data, dict) and "samples" in data:
            samples = data["samples"]
        elif isinstance(data, list):
            # Legacy format: direct list of samples
            samples = data
        else:
            raise ValueError("Multi-label format requires 'samples' key or a list")

        if len(samples) == 0:
            raise ValueError("Dataset is empty")

        first_item = samples[0]
        # Find all label fields (anything that's not 'prompt'), there might be multiple label fields 
        label_fields = [k for k in first_item.keys() if k != "prompt"]
        if len(label_fields) == 0:
            raise ValueError("No label fields found in data")

        # Data extraction:
        prompts = []
        labels_dict = {field: [] for field in label_fields}

        for i, item in enumerate(samples):
            if "prompt" not in item:
                raise ValueError(f"Item {i} missing 'prompt' field")
            prompts.append(item["prompt"])

            for field in label_fields:
                if field not in item:
                    raise ValueError(f"Item {i} missing '{field}' field")
                labels_dict[field].append(item[field])

        return prompts, labels_dict
    else:
        # Single label format: [{"prompt": ..., "label": ...}, ...]
        if not isinstance(data, list):
            raise ValueError("Single-label JSON data must be a list of objects")

        if len(data) == 0:
            raise ValueError("Dataset is empty")

        prompts = []
        labels = []
        for i, item in enumerate(data):
            if "prompt" not in item:
                raise ValueError(f"Item {i} missing 'prompt' field")
            if "label" not in item:
                raise ValueError(f"Item {i} missing 'label' field")
            prompts.append(item["prompt"])
            labels.append(item["label"])

        return prompts, labels

# Export summary
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')

    summary = (
        "JSON Data Loader\n"
        f"{'=' * 50}\n"
        "Supported formats:\n"
        '  1. Single-label: [{"prompt": "...", "label": 0}, ...]\n'
        '  2. Multi-label:  [{"prompt": "...", "label1": 0, "label2": 1}, ...]'
    )
    logger.info(summary)
