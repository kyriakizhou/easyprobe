# easyprobe

**One-liner linear probes for mechanistic interpretability.**

Stop writing boilerplate. Start finding features.

## The Problem

Every mech interp project starts with ~50 lines of the same code:

```python
# Load model
model = HookedTransformer.from_pretrained("pythia-410m")

# Prepare dataset
prompts = [...]
labels = [...]

# Extract activations (manually handling batching)
all_activations = []
for batch in batched(prompts, 8):
    _, cache = model.run_with_cache(batch)
    acts = cache["resid_post", 10][:, -1, :]
    all_activations.append(acts)
activations = torch.cat(all_activations)

# Train probe (remembering to normalize, regularize, cross-validate)
scaler = StandardScaler()
activations_norm = scaler.fit_transform(activations)
probe = LogisticRegression(penalty="l2", C=1.0)
scores = cross_val_score(probe, activations_norm, labels, cv=5)

# Check selectivity (did we find a real signal?)
random_scores = []
for _ in range(5):
    shuffled = np.random.permutation(labels)
    random_scores.append(cross_val_score(probe, activations_norm, shuffled, cv=5).mean())

# Repeat for every layer, component, position...
```

## The Solution

```python
from easyprobe import ProbeAnalyzer

analyzer = ProbeAnalyzer("pythia-410m")
results = analyzer.probe(
    prompts=["I love this!", "I hate this."],
    labels=[1, 0],
)
results.show()
```

That's it. You get:
- ✅ Automatic activation extraction with batching
- ✅ Z-score normalization (makes layers comparable)
- ✅ L2 regularization (prevents overfitting)
- ✅ 5-fold cross-validation (reliable accuracy)
- ✅ Selectivity check (random baseline comparison)
- ✅ Layer-by-layer accuracy visualization
- ✅ Built on TransformerLens for mechanistic interpretability

## Installation

```bash
# Install with TransformerLens backend
pip install easyprobe[transformerlens]

# Or install all dependencies
pip install easyprobe[all]
```

## Quick Start

### One-Liner

```python
from easyprobe import quick_probe

results = quick_probe(
    model="pythia-410m",
    prompts=["Paris is in France", "Paris is in Germany"],
    labels=[1, 0],
)
results.show()
```

### Basic Usage

```python
from easyprobe import ProbeAnalyzer

# Initialize
analyzer = ProbeAnalyzer("pythia-410m")

# Probe with your data
results = analyzer.probe(
    prompts=sentiment_texts,  # List of strings
    labels=sentiment_labels,  # List of 0/1 (or any integers)
)

# Visualize
results.plot_layer_accuracy()
results.plot_heatmap()
results.plot_selectivity()

# Export
df = results.to_dataframe()
print(results.summary())
```

### Advanced Usage

```python
results = analyzer.probe(
    prompts=prompts,
    labels=labels,
    
    # What to probe
    layers=[0, 5, 10, 15, 20, 25, 31],  # Specific layers
    position="last",                     # Token position
    components=["resid", "attn", "mlp"], # All components
    heads="all",                         # Individual attention heads
    
    # Probe settings
    regularization=1.0,  # L2 strength (higher = simpler)
    cv_folds=10,         # More folds = more reliable
    
    # Validation
    include_selectivity=True,  # Compare to random baseline
    random_trials=10,          # More trials = more stable
)

# Head-level visualization
results.plot_head_heatmap(layer=15)

# Component comparison
results.plot_component_comparison()
```

## Understanding Results

### Accuracy

Cross-validated accuracy on your classification task.

- **~50%**: Chance level (binary). The model doesn't encode this feature.
- **60-70%**: Weak signal. The feature might be present.
- **70-85%**: Good signal. The feature is encoded.
- **85%+**: Strong signal. The feature is clearly encoded.

### Selectivity

How much better than random: `selectivity = accuracy - random_baseline`

- **< 5%**: No real signal (probe might be memorizing)
- **5-10%**: Weak signal
- **> 10%**: Strong signal ✓

### Components

| Component | What it is | When to use |
|-----------|------------|-------------|
| `resid` | Cumulative representation (default) | "Is feature encoded by this layer?" |
| `attn` | Attention block output | "Does attention compute this feature?" |
| `mlp` | MLP block output | "Does the MLP compute this feature?" |

## Classification vs Regression

**Classification** (default): Predict categories

```python
results = analyzer.probe(
    prompts=["positive text", "negative text"],
    labels=[1, 0],  # Integers
    probe_type="classification",  # Default
)
# Metric: accuracy
```

**Regression**: Predict continuous values

```python
results = analyzer.probe(
    prompts=["Paris", "London", "Tokyo"],
    labels=[48.8, 51.5, 35.7],  # Latitudes
    probe_type="regression",
)
# Metric: R² (0 to 1)
```

## API Reference

### ProbeAnalyzer

```python
ProbeAnalyzer(
    model: str,            # Model identifier (e.g., "pythia-410m", "gpt2-small")
    backend: str = "auto", # "transformerlens" or "auto" (both use TransformerLens)
    device: str = "auto",  # "cuda", "mps", "cpu", or "auto"
)
```

### probe()

```python
analyzer.probe(
    # Required
    prompts: list[str],
    labels: list[int],
    
    # What to probe
    layers: str | list[int] = "all",
    position: str | int = "last",
    components: list[str] = ["resid"],
    heads: str | list[int] | None = None,
    
    # Settings
    regularization: float = 1.0,
    cv_folds: int = 5,
    normalize: str = "zscore",
    probe_type: str = "classification",
    include_selectivity: bool = True,
    random_trials: int = 5,
    batch_size: int = 8,
    max_workers: int | None = None,
) -> ProbeResults
```

### ProbeResults

```python
results.show()                    # Summary + main plot
results.summary()                 # Text summary
results.to_dataframe()            # Export to pandas
results.to_numpy()                # Export accuracies as array

results.plot_layer_accuracy()     # Accuracy by layer
results.plot_heatmap()            # Layer × component heatmap
results.plot_head_heatmap()       # Head-level heatmap
results.plot_selectivity()        # Selectivity by layer
results.plot_component_comparison() # Compare components

results.best_layer                # Layer with highest accuracy
results.best_accuracy             # Highest accuracy
results.mean_selectivity          # Average selectivity
```

## Contributing

Contributions welcome! Areas where help is needed:

- [ ] More backends (HuggingFace Transformers direct)
- [ ] Cached activation support
- [ ] Interactive visualizations (Plotly)
- [ ] Probe weight visualization
- [ ] Causal intervention using probe directions

## Development
```python
export MPLBACKEND=Agg && export PYTHONPATH=$PYTHONPATH:$(pwd)/.. && .venv/bin/python main.py

```

## License

MIT
