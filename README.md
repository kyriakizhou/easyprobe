# easyprobe

**Train linear probes in 3 lines. Steer models in 1.**

A Python library for AI safety researchers to quickly train linear probes across all layers of a transformer model, use those probes for classification, and steer model behavior — all with minimal boilerplate.

```python
from easyprobe import ProbeOrchestrator, SingleFeatureData

orchestrator = ProbeOrchestrator("gpt2")
results = orchestrator.probe(
    data=SingleFeatureData(
        prompts=["The sky is blue", "Cats can fly"],
        labels=[1, 0]  # 1 = true, 0 = false
    ),
    layers="all",
)
```


## Installation

```bash
pip install -e ".[all]"
```

## Quick Start

### 1. Train probes and find where a feature lives

```python
from easyprobe import ProbeOrchestrator, SingleFeatureData
from easyprobe.models.data_models import PositionOption

orchestrator = ProbeOrchestrator("gpt2")

data = SingleFeatureData(
    prompts=[
        "Paris is the capital of France",       # True
        "Tokyo is the capital of Japan",        # True
        "Berlin is the capital of France",      # False
        "London is the capital of Japan",       # False
    ],
    labels=[1, 1, 0, 0]
)

results = orchestrator.probe(data=data, layers="all")

print(results.best_layer)       # e.g., 8
print(results.best_accuracy)    # e.g., 0.85

# Interactive heatmap + full training report
results.plot_heatmap_interactive(path="heatmap.html")
results.generate_report(path="report.html")

# Visualize position-specific probe scores on text
from easyprobe.visualization.text_highlight import generate_highlight_map_from_results
generate_highlight_map_from_results(
    results, orchestrator, "Paris is the capital of France", 
    method="best_global_layer", save_path="highlight.html"
)
```

### 2. Classify new text with a trained probe

```python
from easyprobe.models.data_models import AggregationMethod

# Use the best probe from training
best_probe = results.best_result

# Classify new examples
score, label = orchestrator.predict(
    text="The Earth orbits around the Sun",
    probe=best_probe,
    aggregation=AggregationMethod.LAST,
    threshold=0.5,
)
print(f"Score: {score:.3f}, Predicted: {'TRUE' if label == 1 else 'FALSE'}")
```

### 3. Save and load probes

```python
# Save specific probe
best_probe.save("my_probe.pkl")

# Save all probes / full results
results.save_probes("trained_probes/")
results.save("results.pkl")

# Load later (no retraining needed)
from easyprobe.models.linear_probe import LinearProbe
probe = LinearProbe.load("my_probe.pkl")
```

### 4. Steer model behavior

```python
model = orchestrator.extractor.model
best_probe = results.best_result

# Create a steering context — amplify the "truthfulness" direction
ctx = best_probe.steer(model, multiplier=3.0)

# Generate steered text
steered_text = ctx.generate("The capital of France is", max_new_tokens=15)
print(steered_text)
```

## API Parameters Explanation

### `ProbeOrchestrator()`

The main entry point. Handles model loading, activation extraction, probe training, and inference.

- **`model`** (`str`): Model identifier (e.g., `"gpt2"`, `"pythia-410m"`, `"allenai/OLMo-2-7B-1124"`).
- **`backend`** (`BackendOption`, default `BackendOption.AUTO`): Which backend to use.
  - `BackendOption.AUTO` / `BackendOption.TRANSFORMERLENS`: TransformerLens
  - `BackendOption.NNSIGHT`: NNSight (for HuggingFace models not supported by TransformerLens)
- **`device`** (`DeviceOption`, default `DeviceOption.AUTO`): Device for computation (`AUTO`, `CPU`, `CUDA`, `MPS`).
- **`revision`** (`Optional[str]`, default `None`): Git revision for HuggingFace models (branch, tag, or commit). Used with NNSight backend.
- **`remote`** (`bool`, default `False`): Unused (kept for backward compatibility).
- **`revisions`** (`Optional[...]`, default `None`): Multiple revisions for multi-model comparison mode.
- **`torch_dtype`** (`Optional[torch.dtype]`, default `None`): Torch dtype for model loading (e.g., `torch.bfloat16`, `torch.float16`). If `None`, uses the model's default dtype. Important for large models to avoid loading in fp32.

### `orchestrator.probe()`

Trains linear probes on model activations.

- **`data`** (`ProbeData`): The dataset to train on. Supports `SingleFeatureData`, `MultiFeatureSharedPromptsData`, and `MultiFeatureSeparatePromptsData`.
- **`layers`** (`LayerSpec`, default `"all"`): Which layers to probe. Can be `"all"`, a list like `[0, 5, 10]`, or a `range`.
- **`components`** (`ComponentSpec`, default `None`): Which components to probe. Defaults to residual stream (`[ComponentOption.RESID]`). Can also include `ATTN` and `MLP`.
- **`position`** (`PositionSpec`, default `PositionOption.LAST`): Which token position to extract. Use `LAST` for the final token, `MEAN` for average over all tokens, `ALL` for every token position, or a specific index like `-1`.
- **`regularization`** (`float`, default `1.0`): L2 regularization strength for the logistic regression model.
- **`probe_type`** (`ProbeType`, default `ProbeType.CLASSIFICATION`): Currently supports classification probes.
- **`include_selectivity`** (`bool`, default `True`): Whether to compute a random baseline (by shuffling labels) to calculate true selectivity (accuracy - baseline).
- **`random_trials`** (`int`, default `5`): Number of random shuffles for the selectivity baseline.
- **`batch_size`** (`int`, default `8`): Batch size for activation extraction.
- **`max_workers`** (`Optional[int]`, default `None`): Number of parallel workers for probe training. `None` uses all available CPUs.
- **`activation_checkpoint_path`** (`Optional[str]`, default `None`): Path to save activation checkpoints (prevents re-extracting if training fails).
- **`auto_cleanup`** (`bool`, default `True`): Whether to automatically clean up activation checkpoints.

### `orchestrator.predict()`

Runs inference with a trained probe on a single text.

- **`text`** (`str`): The input text to classify.
- **`probe`** (`LinearProbe`): The trained probe object to apply.
- **`aggregation`** (`AggregationMethod`, default `AggregationMethod.LAST`): How to aggregate scores across tokens. Supports `LAST`, `MEAN`, `MAX`, `FIRST`.
- **`threshold`** (`float`, default `0.5`): The decision threshold for classification.

### `probe.steer()`

Creates a `SteeringContext` to steer model behavior using the trained probe's direction.

- **`model`**: The model to steer (HookedTransformer or NNSight LanguageModel).
- **`multiplier`** (`float`, default `1.0`): Strength of steering. Positive values amplify the feature, negative values suppress it.
- **`method`** (`str`, default `"standard"`): Steering method to use. `"standard"` performs simple addition. `"dual"` performs regularized Newton updates (Dual Steering) for more robust steering with fewer off-target effects. (`"dual"` is only available for NNSight backend.)
- **`**kwargs`**: Additional parameters for Dual Steering (when `method="dual"`):
  - **`iterations`** (`int`, default `3`): Number of Newton iterations.
  - **`lambda_reg`** (`float`, default `0.1`): Regularization strength for the covariance matrix.
  - **`top_k_cov`** (`int`, default `100`): Number of top activations to use for covariance sampling.

## Usage Workflows

### Workflow 1: Basic Layer Sweep

> **Question:** "At which layer does my model encode factuality?"

```python
from easyprobe import ProbeOrchestrator, SingleFeatureData
from easyprobe.models.data_models import PositionOption, BackendOption
from easyprobe.data.json_loader import load_json_dataset
import torch

# Using OLMo-3 7B via NNSight with bfloat16 to save memory
orchestrator = ProbeOrchestrator(
    "allenai/Olmo-3-1025-7B", 
    backend=BackendOption.NNSIGHT, 
    revision="stage3-step9000",
    torch_dtype=torch.bfloat16,
)

prompts, labels = load_json_dataset("factuality_dataset.json")
data = SingleFeatureData(prompts=prompts, labels=labels)

results = orchestrator.probe(
    data=data,
    layers="all",
    position=PositionOption.LAST,
    components=None,
    include_selectivity=True,
    random_trials=2,
    max_workers=11,
    batch_size=1,
)

print(f"Best layer: {results.best_layer}")
print(f"Best accuracy: {results.best_accuracy:.1%}")
print(f"Selectivity: {results.best_result.selectivity:.1%}")

results.plot_heatmap_interactive(path="sweep.html")
```

### Workflow 2: Component Comparison

> **Question:** "Is factuality better captured in attention, MLP, or the residual stream?"

```python
from easyprobe.models.data_models import ComponentOption

results = orchestrator.probe(
    data=data,
    layers="all",
    components=[ComponentOption.RESID, ComponentOption.ATTN, ComponentOption.MLP],
)
results.plot_heatmap_interactive(path="components.html")
```

### Workflow 3: Multi-Feature Analysis

> **Question:** "Where does the model encode factuality vs. topic?"

```python
from easyprobe import MultiFeatureSharedPromptsData

data = MultiFeatureSharedPromptsData(
    prompts=shared_prompts,
    features={
        "factuality": factuality_labels,
        "topic": topic_labels,
    }
)

results = orchestrator.probe(data=data, layers="all")

# Compare features
for name in results.feature_names:
    r = results[name]
    print(f"{name}: best layer {r.best_layer}, accuracy {r.best_accuracy:.1%}")

results.plot_heatmap_interactive(path="multi_feature.html")
```

### Workflow 4: Model / Checkpoint Comparison

> **Question:** "How does factuality encoding evolve across training stages?"

```python
from easyprobe.visualization import plot_multi_model_heatmap

stages = [
    ("stage1-step1413814", "Stage 1 (Pretraining)"),
    ("stage2-step47684", "Stage 2 (Mid-training)"),
    ("stage3-step11921", "Stage 3 (Long Context)"),
]

results_dict = {}
for revision, label in stages:
    orch = ProbeOrchestrator(
        "allenai/Olmo-3-1025-7B", 
        backend=BackendOption.NNSIGHT, 
        revision=revision,
        torch_dtype=torch.bfloat16,
    )
    results_dict[label] = orch.probe(data=data, layers="all")

plot_multi_model_heatmap(results_dict, path="training_stages.html")
```

### Workflow 5: Dual Steering

> **Goal:** Steer model behavior while minimizing off-target effects using information geometry.

```python
# Standard steering — simple addition of the direction vector
ctx = probe.steer(model, multiplier=3.0, method="standard")
text = ctx.generate("The capital of France is", max_new_tokens=15)

# Dual steering — regularized Newton updates for more robust steering
# (NNSight backend only)
ctx = probe.steer(
    model,
    multiplier=3.0,
    method="dual",
    iterations=3,
    lambda_reg=0.1,
    top_k_cov=100
)
text = ctx.generate("The capital of France is", max_new_tokens=15)
```

## API Reference

### `ProbeOrchestrator`

The main entry point. Handles model loading, activation extraction, probe training, and inference.

| Method | Description |
|--------|-------------|
| `probe(data, layers, position, components, ...)` | Train probes and return `ProbeResults` |
| `predict(text, probe, aggregation, threshold)` | Classify a single text with a trained probe |

### `ProbeResults`

Returned by `probe()`. Contains all trained probes with analysis and visualization methods.

| Property / Method | Description |
|-------------------|-------------|
| `.best_layer` | Layer with highest accuracy |
| `.best_result` | Best `LinearProbe` object |
| `.best_accuracy` | Highest accuracy achieved |
| `.mean_selectivity` | Average selectivity across all probes |
| `.trained_probes` | List of all trained `LinearProbe` objects |
| `.layers` | List of unique layers probed |
| `.components` | List of unique components probed |
| `.to_dataframe()` | Export to pandas DataFrame |
| `.to_numpy()` | Accuracy values as numpy array (resid only) |
| `.filter(component, layer)` | Filter results by component or layer |
| `.plot_heatmap_interactive(title, path, show)` | Interactive Plotly heatmap |
| `.plot_layer_position_heatmap(component, title, path, show)` | Layer × Position heatmap |
| `.generate_report(path, show)` | Full HTML training report |
| `.summary()` | Text summary of results |
| `.show()` | Display summary + heatmap |
| `.save(path)` / `.load(path)` | Persist/reload full results object |
| `.save_probes(dir)` | Save all trained linear probes to directory |

### `MultiFeatureProbeResults`

Returned by `probe()` for multi-feature data. Supports dict-like access by feature name.

| Property / Method | Description |
|-------------------|-------------|
| `[feature_name]` | Get `ProbeResults` for a specific feature |
| `.feature_names` | List of feature names |
| `.num_features` | Number of features |
| `.items()` / `.keys()` / `.values()` | Dict-like iteration |
| `.to_dataframe()` | Combined DataFrame with `feature` column |
| `.plot_heatmap_interactive(title, path, show)` | Multi-subplot heatmap |
| `.generate_report(path, show)` | Full HTML training report |
| `.summary()` | Text summary of all features |
| `.show()` | Display summary + heatmap |
| `.save(path)` / `.load(path)` | Persist/reload full results object |
| `.save_probes(dir)` | Save all probes, organized by feature |

### `LinearProbe`

A trained probe with inference and steering capabilities.

| Method | Description |
|--------|-------------|
| `.predict(activations, threshold)` | Predict class labels |
| `.predict_probability(activations)` | Predict probabilities |
| `.predict_logits(activations)` | Compute raw logits |
| `.predict_on_sequence(activations, aggregation, threshold)` | Predict on a full sequence |
| `.steer(model, multiplier, method, **kwargs)` | Create a `SteeringContext` for steering |
| `.save(path)` / `.load(path)` | Persist/reload a single probe |

### `quick_probe()`

One-liner convenience function:

```python
from easyprobe import quick_probe, SingleFeatureData

results = quick_probe(
    model="gpt2",
    data=SingleFeatureData(prompts=[...], labels=[...]),
)
```

### `generate_highlight_map_from_results()`

Visualize per-token probe scores as a color-highlighted HTML page.

```python
from easyprobe.visualization.text_highlight import generate_highlight_map_from_results

html = generate_highlight_map_from_results(
    results, orchestrator, "Paris is the capital of France",
    method="best_global_layer",  # or "best_token_layer"
    save_path="highlight.html"
)
```

- **`method="best_global_layer"`** (default): Uses the single best layer and evaluates all positions within it.
- **`method="best_token_layer"`**: For each token position, independently selects the most accurate layer.

## Architecture

```
easyprobe/
├── __init__.py              # Public API & quick_probe()
├── main.py                  # CLI / standalone entry point
├── run_dual_steering.py     # Dual steering demo script
├── orchestrator/            # ProbeOrchestrator — main pipeline
├── extractors/              # Activation extraction backends
│   ├── base.py              #   Abstract base class
│   ├── transformerlens.py   #   TransformerLens backend
│   └── nnsight.py           #   NNSight backend
├── models/
│   ├── data_models.py       # Data classes, enums, type aliases
│   ├── linear_probe.py      # LinearProbe (inference + steering)
│   ├── probe_results.py     # ProbeResults & MultiFeatureProbeResults
│   └── steering.py          # Steering contexts (standard + dual)
├── probing/
│   ├── train.py             # Probe training (sklearn LogisticRegression)
│   └── normalize.py         # Z-score normalization
├── storage/                 # Batch storage (in-memory + checkpointed)
├── util/                    # Helpers, validation, profiling
├── visualization/           # Plotly heatmaps, HTML reports, text highlighting
│   ├── heatmap.py           #   Interactive Plotly heatmaps
│   ├── report.py            #   HTML report generation
│   ├── text_highlight.py    #   Per-token probe score highlighting
│   └── templates/           #   Jinja2 HTML templates
├── data/                    # JSON loader + built-in datasets
│   ├── json_loader.py       #   Generic JSON dataset loader
│   ├── factuality.py        #   Built-in factuality dataset
│   └── datasets/            #   Bundled dataset files
├── examples/                # Example scripts and notebooks
└── notebooks/               # Research notebooks
```

## Preparing Custom Data

### From JSON

```json
[
  {"prompt": "Paris is the capital of France", "label": 1},
  {"prompt": "Berlin is the capital of France", "label": 0}
]
```

```python
from easyprobe.data.json_loader import load_json_dataset

prompts, labels = load_json_dataset("my_dataset.json")
data = SingleFeatureData(prompts=prompts, labels=labels)
```

### Multi-label JSON

```json
{
  "_metadata": {"description": "Factuality + Topic dataset"},
  "samples": [
    {"prompt": "Paris is the capital of France", "factuality": 1, "topic": 0},
    {"prompt": "Berlin is the capital of France", "factuality": 0, "topic": 0}
  ]
}
```

```python
prompts, labels_dict = load_json_dataset("my_dataset.json", multi_label=True)
# labels_dict = {"factuality": [1, 0], "topic": [0, 0]}
```

### From Python lists

```python
data = SingleFeatureData(
    prompts=["statement 1", "statement 2", ...],
    labels=[1, 0, ...]  # Binary: 1 = positive class, 0 = negative class
)
```

## Requirements

- Python ≥ 3.9
- At least one backend: `transformer-lens` or `nnsight` + `transformers`
- Core dependencies: `numpy`, `pandas`, `scikit-learn`, `plotly`, `jinja2`

## License

MIT
