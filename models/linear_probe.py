"""
LinearProbe class for inference and steering.
"""

from __future__ import annotations

import os
from typing import Optional, Union
from pathlib import Path
import pickle

from contextlib import contextmanager

import numpy as np
import torch

from easyprobe.models.data_models import ComponentOption, ProbeType, PositionOption, AggregationMethod
from easyprobe.models.steering import SteeringContext, TransformerLensSteeringContext, NNSightSteeringContext, DualNNSightSteeringContext


class LinearProbe:
    """
    A trained linear probe that can be used for inference and steering.
    
    Attributes:
        layer (int): The layer this probe was trained on.
        component (ComponentOption): The component (resid, attn, mlp) this probe targets.
        weights (np.ndarray): The learned weights (shape: [hidden_dim] or [n_classes, hidden_dim]).
        bias (np.ndarray): The learned bias (shape: [1] or [n_classes]).
        probe_type (ProbeType): Classification or Regression.
    """
    
    def __init__(
        self,
        layer: int,
        component: ComponentOption,
        weights: np.ndarray,
        bias: np.ndarray,
        accuracy: float,
        n_samples: int,
        probe_type: ProbeType = ProbeType.CLASSIFICATION,
        position: Optional[Union[PositionOption, list[int]]] = None,
        accuracy_std: Optional[float] = None,
        auc: Optional[float] = None,
        selectivity: Optional[float] = None,
        random_baseline: Optional[float] = None,
        training_time: Optional[float] = None,
        metadata: Optional[dict] = None,
        norm_mean: Optional[np.ndarray] = None,
        norm_std: Optional[np.ndarray] = None,
    ):
        self.layer = layer
        self.component = component
        self.weights = weights
        self.bias = bias
        self.accuracy = accuracy
        self.n_samples = n_samples
        self.probe_type = probe_type
        self.position = position
        self.accuracy_std = accuracy_std
        self.auc = auc
        self.selectivity = selectivity
        self.random_baseline = random_baseline
        self.training_time = training_time
        self.metadata = metadata or {}
        self.norm_mean = norm_mean  # Z-score mean from training, shape: (hidden_dim,)
        self.norm_std = norm_std  # Z-score std from training, shape: (hidden_dim,)
        # is_significant: simple heuristic based on selectivity
        self.is_significant = selectivity is not None and selectivity > 0
        
    @classmethod
    def from_data(cls, data: dict) -> "LinearProbe":
        """Create a LinearProbe from a dictionary."""
        if data.get("weights") is None or data.get("bias") is None:
            raise ValueError("Data does not contain weights/bias.")
            
        return cls(
            layer=data["layer"],
            component=data["component"],
            weights=data["weights"],
            bias=data["bias"],
            accuracy=data["accuracy"],
            n_samples=data["n_samples"],
            probe_type=data["probe_type"],
            position=data.get("position"),
            accuracy_std=data.get("accuracy_std"),
            auc=data.get("auc"),
            selectivity=data.get("selectivity"),
            random_baseline=data.get("random_baseline"),
            training_time=data.get("training_duration_s")
        )
        
    def save(self, path: Union[str, Path]):
        """Save the probe to disk."""
        path = str(path)
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        data = {
            "layer": self.layer,
            "component": self.component,
            "weights": self.weights,
            "bias": self.bias,
            "accuracy": self.accuracy,
            "n_samples": self.n_samples,
            "probe_type": self.probe_type,
            "position": self.position,
            "accuracy_std": self.accuracy_std,
            "auc": self.auc,
            "selectivity": self.selectivity,
            "random_baseline": self.random_baseline,
            "training_time": self.training_time,
            "metadata": self.metadata,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
        }
        
        with open(path, "wb") as f:
            pickle.dump(data, f)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> "LinearProbe":
        """Load a probe from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        # Backward compatibility for old format (if any exist)
        if "metadata" in data and "accuracy" not in data:
            # Old format
            return cls(
                layer=data["layer"],
                component=data["component"],
                weights=data["weights"],
                bias=data["bias"],
                probe_type=data["probe_type"],
                accuracy=data["metadata"].get("accuracy", 0.0),
                n_samples=data["metadata"].get("n_samples", 0),
                selectivity=data["metadata"].get("selectivity"),
                training_time=data["metadata"].get("training_time"),
                metadata=data.get("metadata")
            )

        return cls(
            layer=data["layer"],
            component=data["component"],
            weights=data["weights"],
            bias=data["bias"],
            accuracy=data["accuracy"],
            n_samples=data["n_samples"],
            probe_type=data["probe_type"],
            position=data.get("position"),
            accuracy_std=data.get("accuracy_std"),
            auc=data.get("auc"),
            selectivity=data.get("selectivity"),
            random_baseline=data.get("random_baseline"),
            training_time=data.get("training_time"),
            metadata=data.get("metadata"),
            norm_mean=data.get("norm_mean"),
            norm_std=data.get("norm_std"),
        )

    def _prepare_input(self, activations: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Ensure input is numpy array."""
        if hasattr(activations, "cpu"):
            activations = activations.detach().cpu().numpy()
        return np.array(activations)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization using stored training params.
        
        If norm_mean/norm_std are not set (e.g., old probe), returns x unchanged.
        """
        if self.norm_mean is not None and self.norm_std is not None:
            # Avoid division by zero (same behavior as sklearn StandardScaler)
            std = self.norm_std.copy()
            std[std == 0] = 1.0
            return (x - self.norm_mean) / std
        return x

    def predict_logits(self, activations: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Compute raw logits (dot product + bias).
        
        Automatically applies z-score normalization using training params
        if they are stored in the probe.
        
        Args:
            activations: Input vectors [batch, hidden_dim] (raw or normalized)
            
        Returns:
            Logits [batch] or [batch, n_classes]
        """
        x = self._prepare_input(activations)
        
        # Auto-normalize using stored training params
        x = self._normalize(x)
        
        # Standard: result = x @ w.T + b
        logits = x @ self.weights.T + self.bias
        
        # If binary classification, flatten reasonable outputs
        if self.probe_type == ProbeType.CLASSIFICATION and logits.shape[1] == 1:
            return logits.ravel()
            
        return logits

    def predict_probability(self, activations: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict probabilities (0-1).
        
        For binary classification, applies Sigmoid.
        For multi-class, applies Softmax.
        """
        logits = self.predict_logits(activations)
        
        if self.probe_type == ProbeType.CLASSIFICATION:
            # Binary case (single output logit)
            if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
                # Sigmoid
                return 1.0 / (1.0 + np.exp(-logits))
            else:
                # Softmax (multiclass)
                e_x = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)
        else:
            # Regression: probabilities don't apply, return logits/values
            # Ideally warn user
            return logits

    def predict(self, activations: Union[np.ndarray, torch.Tensor], threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels (0 or 1, etc.).
        
        Args:
            activations: Input vectors
            threshold: Decision threshold for binary classification (default 0.5)
            
        Returns:
            Class labels (integers)
        """
        if self.probe_type == ProbeType.CLASSIFICATION:
            # Binary
            if self.weights.shape[0] == 1:
                probs = self.predict_probability(activations)
                return (probs > threshold).astype(int)
            else:
                # Multiclass: Argmax
                logits = self.predict_logits(activations)
                return np.argmax(logits, axis=1)
        else:
            # Regression: just return values
            return self.predict_logits(activations)

    def predict_on_sequence(
        self, 
        activations: Union[np.ndarray, torch.Tensor], 
        aggregation: AggregationMethod = AggregationMethod.LAST,
        threshold: float = 0.5
    ) -> tuple[float, int]:
        """
        Predict on a sequence of activations (shape: [seq_len, hidden_dim]).
        
        Args:
            activations: Sequence activations [seq_len, hidden_dim]
            aggregation: Method to aggregate scores across tokens.
            threshold: Decision threshold for classification.
            
        Returns:
            Tuple of (aggregated_score, predicted_label)
        """
        acts = self._prepare_input(activations)
        
        # 1. Compute scores for ALL tokens individually
        # For classification, we use probabilities (0-1)
        # linear probe weights: [1, hidden_dim] -> logits: [seq_len, 1]
        scores = self.predict_probability(acts) # Shape: [seq_len] or [seq_len, n_classes]
        
        # Handle shape: ensure [seq_len] for binary
        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores.ravel()
            
        # 2. Aggregate
        if aggregation == AggregationMethod.MAX:
            final_score = np.max(scores)
        elif aggregation == AggregationMethod.MEAN:
            final_score = np.mean(scores)
        elif aggregation == AggregationMethod.FIRST:
            final_score = scores[0]
        elif aggregation == AggregationMethod.LAST:
            final_score = scores[-1]
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
            
        # 3. Threshold
        label = int(final_score > threshold)
        
        return final_score, label

    @property
    def pid(self) -> Optional[int]:
        """Process ID from training (if available)."""
        return self.metadata.get("pid")

    @property
    def start_time(self) -> Optional[float]:
        """Start time of training (if available)."""
        return self.metadata.get("start_time")

    @property
    def end_time(self) -> Optional[float]:
        """End time of training (if available)."""
        return self.metadata.get("end_time")

    def steer(
        self, 
        model, 
        multiplier: float = 1.0, 
        method: str = "standard",
        **kwargs
    ) -> SteeringContext:
        """
        Context manager to steer the model using this probe's direction.
        
        Args:
            model: The model to steer (HookedTransformer or LanguageModel)
            multiplier: Strength of steering (positive to promote, negative to suppress)
            method: Steering method to use ("standard" or "dual")
            **kwargs: Additional parameters for specific steering methods 
                     (e.g., iterations, lambda_reg for dual steering)
        """
        # We need the probe vector.
        # For binary/single-label, shape is [1, hidden_dim]
        if self.weights.shape[0] == 1:
            vector = self.weights[0]
        else:
            raise NotImplementedError("Steering for multiclass probes not yet supported.")
            
        # Convert to torch tensor if needed
        if isinstance(vector, np.ndarray):
            vector = torch.from_numpy(vector).float()
            
        # Detect backend
        is_nnsight = hasattr(model, "trace") and hasattr(model, "generate")

        if is_nnsight:
            if method.lower() == "dual":
                return DualNNSightSteeringContext(
                    model=model, 
                    layer=self.layer, 
                    component=self.component, 
                    vector=vector, 
                    multiplier=multiplier,
                    **kwargs
                )
            else:
                return NNSightSteeringContext(
                    model=model, 
                    layer=self.layer, 
                    component=self.component, 
                    vector=vector, 
                    multiplier=multiplier
                )
        else:
            if method.lower() == "dual":
                raise NotImplementedError("Dual steering is currently only implemented for NNSight backend.")
            return TransformerLensSteeringContext(
                model=model, 
                layer=self.layer, 
                component=self.component, 
                vector=vector, 
                multiplier=multiplier
            )

