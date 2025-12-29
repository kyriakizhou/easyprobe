"""
Activation normalization for fair probe comparisons.

Different layers can have very different activation scales due to:
1. Residual Accumulation: Variance tends to grow (or drift) as vectors are added to the residual stream.
2. LayerNorm: Rescales inputs to components, but the residual stream itself may drift.
3. Outliers: Specific features/neurons may have vastly distinct magnitudes.

Normalization ensures that:
1. Regularization (e.g., L2) penalizes "complexity" equally across layers.
2. Feature importance (probe weights) is comparable across layers.
3. Training is numerically stable.
"""

from enum import Enum
from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


class NormalizationMethod(Enum):
    """
    How to normalize activations before probe training.

    ZSCORE (recommended):
        - Transforms to mean=0, std=1
        - Makes regularization affect all layers equally
        - Probe weights become comparable across layers

    MINMAX:
        - Scales to [0, 1] range
        - Preserves relative differences within each layer
        - Less common for linear probing

    NONE:
        - No normalization
        - Use only if you know what you're doing
        - Different layers will have different effective regularization
    """

    ZSCORE = "zscore"
    MINMAX = "minmax"
    NONE = "none"


class ActivationNormalizer:
    """
    Normalize activations for fair probe comparisons.

    Example:
        normalizer = ActivationNormalizer("zscore")

        # During training: fit and transform
        train_acts_normalized = normalizer.fit_transform(train_acts, key=(5, "resid"))

        # During evaluation: use same transformation
        test_acts_normalized = normalizer.transform(test_acts, key=(5, "resid"))
    """

    def __init__(self, method: NormalizationMethod = NormalizationMethod.ZSCORE):
        """
        Initialize normalizer.

        Args:
            method: Normalization method (NormalizationMethod enum)
        """
        self.method = method
        self.scalers: dict[tuple, StandardScaler] = {}

    def _reshape_if_3d(self, activations: np.ndarray) -> tuple[np.ndarray, Optional[tuple]]:
        """If 3D, flatten to 2D. Returns (reshaped_array, original_shape)."""
        if activations.ndim == 3:
            original_shape = activations.shape
            n_samples, seq_len, hidden_dim = original_shape
            return activations.reshape(n_samples * seq_len, hidden_dim), original_shape
        return activations, None

    def fit_transform(self, activations: np.ndarray, key: tuple) -> np.ndarray:
        """
        Fit normalizer on data and transform.

        Args:
            activations: Array of shape (n_samples, hidden_dim) or (n_samples, seq_len, hidden_dim)
            key: Identifier for this set of activations (e.g., (layer, component))

        Returns:
            Normalized activations with same shape
        """
        if self.method == NormalizationMethod.NONE:
            return activations

        acts_2d, original_shape = self._reshape_if_3d(activations)

        if self.method == NormalizationMethod.ZSCORE:
            scaler = StandardScaler()
            normalized = scaler.fit_transform(acts_2d)
            self.scalers[key] = scaler
            
            if original_shape:
                return normalized.reshape(original_shape)
            return normalized

        elif self.method == NormalizationMethod.MINMAX:
            # Scale across all sample/token dimensions
            min_val = acts_2d.min(axis=0, keepdims=True)
            max_val = acts_2d.max(axis=0, keepdims=True)
            normalized = (acts_2d - min_val) / (max_val - min_val + 1e-8)
            
            if original_shape:
                return normalized.reshape(original_shape)
            return normalized

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

    def transform(self, activations: np.ndarray, key: tuple) -> np.ndarray:
        """
        Transform using previously fit normalizer.

        Args:
            activations: Array of shape (n_samples, hidden_dim) or (n_samples, seq_len, hidden_dim)
            key: Identifier matching a previous fit_transform call

        Returns:
            Normalized activations with same shape
        """
        if self.method == NormalizationMethod.NONE:
            return activations

        acts_2d, original_shape = self._reshape_if_3d(activations)

        if key in self.scalers:
            normalized = self.scalers[key].transform(acts_2d)
            if original_shape:
                return normalized.reshape(original_shape)
            return normalized
        else:
            # If not fit, just fit_transform
            return self.fit_transform(activations, key)
