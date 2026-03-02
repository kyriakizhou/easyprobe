"""
Activation normalization for fair probe comparisons.

Different layers can have very different activation scales due to:
1. Residual Accumulation: Variance tends to grow (or drift) as vectors are added to the residual stream.
2. LayerNorm: Rescales inputs to components, but the residual stream itself may drift.
3. Outliers: Specific features/neurons may have vastly distinct magnitudes.

Z-score normalization (mean=0, std=1) ensures that:
1. Regularization (e.g., L2) penalizes "complexity" equally across layers.
2. Feature importance (probe weights) is comparable across layers.
3. Training is numerically stable.
"""

from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler


class ActivationNormalizer:
    """
    Z-score normalize activations for fair probe comparisons.

    After fitting, the per-key (mean, std) can be retrieved via get_params()
    to store in LinearProbe for inference-time normalization.

    Example:
        normalizer = ActivationNormalizer()

        # During training: fit and transform
        train_acts_normalized = normalizer.fit_transform(train_acts, key=(5, "resid"))

        # Retrieve fitted params for storage in LinearProbe
        mean, std = normalizer.get_params(key=(5, "resid"))

        # During evaluation: use same transformation
        test_acts_normalized = normalizer.transform(test_acts, key=(5, "resid"))
    """

    def __init__(self):
        """Initialize normalizer."""
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
        Fit normalizer on data and transform (z-score: mean=0, std=1).

        Args:
            activations: Array of shape (n_samples, hidden_dim) or (n_samples, seq_len, hidden_dim)
            key: Identifier for this set of activations (e.g., (layer, component))

        Returns:
            Normalized activations with same shape
        """
        acts_2d, original_shape = self._reshape_if_3d(activations)

        scaler = StandardScaler()
        normalized = scaler.fit_transform(acts_2d)
        self.scalers[key] = scaler

        if original_shape:
            return normalized.reshape(original_shape)
        return normalized

    def transform(self, activations: np.ndarray, key: tuple) -> np.ndarray:
        """
        Transform using previously fit normalizer.

        Args:
            activations: Array of shape (n_samples, hidden_dim) or (n_samples, seq_len, hidden_dim)
            key: Identifier matching a previous fit_transform call

        Returns:
            Normalized activations with same shape
        """
        acts_2d, original_shape = self._reshape_if_3d(activations)

        if key in self.scalers:
            normalized = self.scalers[key].transform(acts_2d)
            if original_shape:
                return normalized.reshape(original_shape)
            return normalized
        else:
            # If not fit, just fit_transform
            return self.fit_transform(activations, key)

    def get_params(self, key: tuple) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the fitted (mean, std) for a given key.

        Args:
            key: Identifier matching a previous fit_transform call

        Returns:
            Tuple of (mean, std) numpy arrays, each of shape (hidden_dim,)
        """
        if key not in self.scalers:
            raise KeyError(f"No fitted scaler for key {key}. Call fit_transform first.")
        scaler = self.scalers[key]
        return scaler.mean_.copy(), scaler.scale_.copy()
