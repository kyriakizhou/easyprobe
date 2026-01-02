# easyprobe

**One-liner linear probes for mechanistic interpretability.**

Performance optimized library to automatically train linear probes across all model layers, model components (attn, mlp, resid), token positions, training stages, models. 

## Installation

```bash
pip install -e .

# With TransformerLens backend
pip install -e ".[transformerlens]"

# With NNSight backend (for OLMo-3 and other models)
pip install -e ".[nnsight]"

# Both backends
pip install -e ".[all]"
```

## Quick Start

```python
from easyprobe import ProbeOrchestrator, SingleFeatureData
from easyprobe.datamodels import PositionOption

orchestrator = ProbeOrchestrator("pythia-410m")

data = SingleFeatureData(
    prompts=[
        "Paris is the capital of France",      # True
        "Tokyo is the capital of Japan",       # True
        "Berlin is the capital of France",     # False
        "London is the capital of Japan",      # False
    ],
    labels=[1, 1, 0, 0]
)

results = orchestrator.probe(
    data=data,
    layers="all",
    position=PositionOption.LAST,
    include_selectivity=True,
)

results.best_layer       # Layer with highest accuracy
results.best_accuracy    # e.g., 0.85

results.plot_heatmap_interactive(output_path="heatmap.html")
results.generate_report(output_path="report.html")
```

## Preparing Your Data

### JSON Format

**Single-label format:**
```json
[
  {"prompt": "Paris is the capital of France", "label": 1},
  {"prompt": "Berlin is the capital of France", "label": 0}
]
```

**Multi-label format** (for probing multiple features on shared prompts):
```json
{
  "_metadata": {
    "label1": "factuality (1=true, 0=false)",
    "label2": "topic (1=math, 0=climate)"
  },
  "samples": [
    {"prompt": "Two plus two equals four", "label1": 1, "label2": 1},
    {"prompt": "Two plus two equals five", "label1": 0, "label2": 1}
  ]
}
```

### Loading Data

```python
from easyprobe.data.json_loader import load_json_dataset

# Single-label
prompts, labels = load_json_dataset("my_dataset.json")

# Multi-label
prompts, labels_dict = load_json_dataset("data.json", multi_label=True)
```

## Use Cases

The library supports 6 main scenarios:

1. **Basic Layer Sweep** - Find which layer encodes your feature
2. **Component Comparison** - Compare residual stream, attention, and MLP outputs
3. **Position Analysis** - Probe all token positions
4. **Multi-Feature (Shared Prompts)** - Probe multiple features with shared activations
5. **Multi-Feature (Separate Prompts)** - Probe different features with independent datasets
6. **Model Comparison** - Compare across models or training stages

See [`main.py`](main.py) for complete examples of all scenarios.

## Output

### Interactive Heatmap

![Heatmap Example](docs/heatmap_example.png)

## Backend Support

- **TransformerLens** (default): GPT-2, Pythia, etc.
- **NNSight**: OLMo-3, Llama, and other HuggingFace models

```python
from easyprobe.datamodels import BackendOption

# TransformerLens
orchestrator = ProbeOrchestrator("pythia-410m", backend=BackendOption.TRANSFORMERLENS)

# NNSight
orchestrator = ProbeOrchestrator("allenai/OLMo-2-1124-7B", backend=BackendOption.NNSIGHT)
```

## Development

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd) && python main.py
```

## License

MIT
