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
from pathlib import Path
from typing import Union


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
        if "prompt" not in first_item:
            raise ValueError("Each item must have a 'prompt' field")

        # Find all label fields (anything that's not 'prompt')
        label_fields = [k for k in first_item.keys() if k != "prompt"]
        if len(label_fields) == 0:
            raise ValueError("No label fields found in data")

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

        first_item = data[0]
        if "prompt" not in first_item:
            raise ValueError("Each item must have a 'prompt' field")
        if "label" not in first_item:
            raise ValueError("Single-label format requires 'label' field")

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


def save_json_dataset(
    path: Union[str, Path],
    prompts: list[str],
    labels: Union[list[int], dict[str, list[int]]],
) -> None:
    """
    Save a dataset to JSON format.

    Args:
        path: Output path for JSON file
        prompts: List of prompt strings
        labels: Either a list of labels (single-label) or dict of label lists (multi-label)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []

    if isinstance(labels, dict):
        # Multi-label format
        label_fields = list(labels.keys())
        n_samples = len(prompts)

        for field in label_fields:
            if len(labels[field]) != n_samples:
                raise ValueError(f"Label '{field}' has {len(labels[field])} items, expected {n_samples}")

        for i, prompt in enumerate(prompts):
            item = {"prompt": prompt}
            for field in label_fields:
                item[field] = labels[field][i]
            data.append(item)
    else:
        # Single-label format
        if len(labels) != len(prompts):
            raise ValueError(f"Labels has {len(labels)} items, prompts has {len(prompts)}")

        for prompt, label in zip(prompts, labels):
            data.append({"prompt": prompt, "label": label})

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# Export summary
if __name__ == "__main__":
    print("JSON Data Loader")
    print("=" * 50)
    print("\nSupported formats:")
    print("\n1. Single-label:")
    print('   [{"prompt": "...", "label": 0}, ...]')
    print("\n2. Multi-label:")
    print('   [{"prompt": "...", "label1": 0, "label2": 1}, ...]')
