"""
Data classes for probe tasks and results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np

class ProbeType(Enum):
    """
    Type of probe to train.

    CLASSIFICATION: Predict discrete categories (0, 1, 2, ...)
        - Use for: sentiment (pos/neg), factuality (true/false), language detection
        - Metric: accuracy (% correct predictions)
        - Random baseline: ~50% for binary, ~1/k for k classes
    """

    CLASSIFICATION = "classification"
    # Future: add REGRESSION


class PositionOption(Enum):
    """Which token position to extract activations from."""

    LAST = "last"  # Last token (most common for autoregressive models)
    MEAN = "mean"  # Mean across all tokens
    ALL = "all"  # All tokens


class ComponentOption(Enum):
    """Which component to extract activations from."""

    RESID = "resid"  # Residual stream (cumulative representation)
    ATTN = "attn"  # Attention block output
    MLP = "mlp"  # MLP block output


class DeviceOption(Enum):
    """Which device to run computations on."""

    AUTO = "auto"  # Auto-detect best available device (CUDA > MPS > CPU)
    CPU = "cpu"  # Force CPU execution
    CUDA = "cuda"  # NVIDIA GPU (or CUDA:0, CUDA:1, etc. for multi-GPU)
    MPS = "mps"  # Apple Silicon GPU (Metal Performance Shaders)


class BackendOption(Enum):
    """Which backend library to use for activation extraction."""

    AUTO = "auto"  # Use TransformerLens (default)
    TRANSFORMERLENS = "transformerlens"  # Use TransformerLens (mech interp focused)


# Type aliases for cleaner type hints
LayerSpec = Union[str, list[int]]  # "all" or [0, 5, 10] or range(0, 32)
PositionSpec = Union[PositionOption, int, list[int]]
ComponentSpec = Optional[list[ComponentOption]]


@dataclass
class SingleFeatureData:
    """
    Single feature probe data (one-to-one).

    Use this for the basic case: one set of prompts, one set of labels.

    Example:
        data = SingleFeatureData(
            prompts=["I love this!", "I hate this!"],
            labels=[1, 0]
        )
    """
    prompts: list[str]
    labels: list[int]

    def __post_init__(self):
        """Validate that prompts and labels have the same length."""
        if len(self.prompts) != len(self.labels):
            raise ValueError(
                f"Prompts ({len(self.prompts)}) and labels ({len(self.labels)}) "
                f"must have the same length"
            )

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return len(self.prompts)


@dataclass
class MultiFeatureSharedPromptsData:
    """
    Multiple features with shared prompts (one-to-many).

    Use this when you want to probe multiple features on the same set of prompts.
    This is the most efficient mode because activations are extracted only once.

    Example:
        data = MultiFeatureSharedPromptsData(
            prompts=["I love this!", "I hate this!"],
            features={
                "sentiment": [1, 0],
                "formality": [0, 1],
                "topic": [1, 1],
            }
        )
    """
    prompts: list[str]
    features: dict[str, list[int]]

    def __post_init__(self):
        """Validate that all feature labels have the same length as prompts."""
        if not self.features:
            raise ValueError("features dict cannot be empty")

        for feature_name, labels in self.features.items():
            if len(self.prompts) != len(labels):
                raise ValueError(
                    f"Feature '{feature_name}': prompts ({len(self.prompts)}) and "
                    f"labels ({len(labels)}) must have the same length"
                )

    @property
    def num_samples(self) -> int:
        """Return the number of samples."""
        return len(self.prompts)

    @property
    def num_features(self) -> int:
        """Return the number of features."""
        return len(self.features)

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names."""
        return list(self.features.keys())


@dataclass
class MultiFeatureSeparatePromptsData:
    """
    Multiple features with separate prompts (many-to-many).

    Use this when each feature has its own set of prompts and labels.
    Activations will be extracted separately for each feature.

    Example:
        data = MultiFeatureSeparatePromptsData(
            features={
                "sentiment": (
                    ["I love this!", "I hate this!"],
                    [1, 0]
                ),
                "deception": (
                    ["I told a lie", "I was honest"],
                    [1, 0]
                ),
            }
        )
    """
    features: dict[str, tuple[list[str], list[int]]]

    def __post_init__(self):
        """Validate that each feature has matching prompts and labels."""
        if not self.features:
            raise ValueError("features dict cannot be empty")

        for feature_name, (prompts, labels) in self.features.items():
            if len(prompts) != len(labels):
                raise ValueError(
                    f"Feature '{feature_name}': prompts ({len(prompts)}) and "
                    f"labels ({len(labels)}) must have the same length"
                )

    @property
    def num_features(self) -> int:
        """Return the number of features."""
        return len(self.features)

    @property
    def feature_names(self) -> list[str]:
        """Return the list of feature names."""
        return list(self.features.keys())

    def get_feature_data(self, feature_name: str) -> tuple[list[str], list[int]]:
        """Get the (prompts, labels) tuple for a specific feature."""
        return self.features[feature_name]


# Union type for all probe data types
ProbeData = Union[SingleFeatureData, MultiFeatureSharedPromptsData, MultiFeatureSeparatePromptsData]


@dataclass
class ProbeTask:
    """
    A single probing task to be executed.

    This is passed to worker processes for parallel probe training.
    Contains all information needed to train one probe independently.
    """


    layer: int
    component: ComponentOption  # ComponentOption enum
    position: PositionOption | list[int]
    activations: np.ndarray  # Shape: (n_samples, hidden_dim)
    labels: np.ndarray  # Shape: (n_samples,). Ground truth targets (integers for classification).
    regularization: float
    cv_folds: int
    probe_type: ProbeType
    include_selectivity: bool
    random_trials: int  # Number of random shuffles for selectivity check (calculating baseline accuracy with shuffled labels).


@dataclass
class ProbeResult:
    """
    Result from a single probe.

    Attributes:
        layer: Which layer this probe was trained on
        component: Which component (ComponentOption enum)
        position: Token position used (PositionOption enum or specific index list)
        accuracy: Cross-validated accuracy (or R² for regression)
        accuracy_std: Standard deviation across CV folds
        random_baseline: Accuracy when trained on shuffled labels
        random_baseline_std: Std of random baseline across trials
        selectivity: accuracy - random_baseline (how much better than random)
        probe_type: Whether classification or regression
        n_samples: Number of samples used for training
    """

    layer: int
    component: ComponentOption
    position: PositionOption | list[int]
    accuracy: float  # Or R² for regression
    accuracy_std: float
    random_baseline: Optional[float]
    random_baseline_std: Optional[float]
    selectivity: Optional[float]
    probe_type: ProbeType
    n_samples: int

    @property
    def is_significant(self) -> bool:
        """
        Check if selectivity indicates real signal.

        Returns True if the probe performs more than 10% better than
        random baseline, suggesting the feature is genuinely encoded.
        """
        if self.selectivity is None:
            return False
        return self.selectivity > 0.10

    # Optional timing fields (populated during training)
    pid: Optional[int] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    training_duration_s: Optional[float] = None

    # Probe weights for steering (shape: hidden_dim for binary, or n_classes x hidden_dim)
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        sel_str = f", selectivity={self.selectivity:.1%}" if self.selectivity else ""
        return (
            f"ProbeResult(layer={self.layer}, component='{self.component}'"
            f", accuracy={self.accuracy:.1%}{sel_str})"
        )


# ============================================================================
# Timing Report Data Models
# ============================================================================

@dataclass
class ProbeTrainingRecord:
    """Timing and result data for a single probe."""
    layer: int
    component: ComponentOption
    position: PositionOption | int
    pid: int
    start_time: float  # Unix timestamp
    end_time: float
    duration_s: float
    accuracy: float
    accuracy_std: float
    random_baseline: Optional[float]
    selectivity: Optional[float]
    n_samples: int
    weights: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None


@dataclass
class FeatureTimingReport:
    """Timing report for a single feature."""
    feature_name: str
    n_samples: int
    n_probes: int  # layers × components × positions

    # Timing for this feature
    extraction_s: float  # Time to extract activations (0 if shared)
    normalization_s: float
    training_s: float

    # Per-probe records
    probe_records: list[ProbeTrainingRecord]

    # Best result
    best_layer: int
    best_component: ComponentOption
    best_position: PositionOption | int
    best_accuracy: float
    best_selectivity: Optional[float]
    best_weights: Optional[np.ndarray] = None
    best_bias: Optional[np.ndarray] = None


@dataclass
class ModelTimingReport:
    """Timing report for a single model (may contain multiple features)."""
    model_name: str

    # Timing for this model
    model_loading_s: float
    extraction_s: float  # Time to extract activations

    # Per-feature reports for this model
    feature_reports: list[FeatureTimingReport]


@dataclass
class ProbeTimingReport:
    """Complete timing report for a probe run (supports multiple models and features)."""
    # Metadata
    run_timestamp: str
    total_s: float

    # Per-model reports (supports probing same features across multiple models)
    model_reports: list[ModelTimingReport]

    @property
    def is_multi_model(self) -> bool:
        return len(self.model_reports) > 1

    @property
    def is_multi_feature(self) -> bool:
        return any(len(m.feature_reports) > 1 for m in self.model_reports)

    @property
    def total_probes(self) -> int:
        """Total number of probes trained across all models and features."""
        return sum(
            f.n_probes
            for m in self.model_reports
            for f in m.feature_reports
        )

    @property
    def total_model_loading_s(self) -> float:
        """Total time spent loading models."""
        return sum(m.model_loading_s for m in self.model_reports)

    @property
    def total_extraction_s(self) -> float:
        """Total time spent extracting activations."""
        return sum(m.extraction_s for m in self.model_reports)

    @property
    def total_training_s(self) -> float:
        """Total time spent training probes."""
        return sum(
            f.training_s
            for m in self.model_reports
            for f in m.feature_reports
        )
