"""
Base class for activation extraction backends.
"""

from abc import ABC, abstractmethod
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from easyprobe.models.data_models import ComponentOption, PositionOption, ActivationKey, DeviceOption


from easyprobe.storage import (
    BatchStorage,
    create_batch_storage,
    concatenate_batches,
    SEQ_LENGTHS_KEY
)


BatchResults = dict[ActivationKey, np.ndarray] # { (layer_index, component_option): np.ndarray }


class ActivationExtractor(ABC):
    """
    Abstract base class for activation extraction.

    Implementations handle the specifics of different libraries
    while providing a uniform interface.

    Notes on Model Size & Hardware:
    - TransformerLens relies on loading model weights into GPU/CPU memory.
      Limit is determined by available VRAM/RAM.
    """

    @abstractmethod
    def get_model_config(self) -> dict:
        """
        Return model configuration.

        Returns:
            Dictionary with keys:
            - n_layers: int
            - n_heads: int
            - hidden_dim: int (Width of the residual stream / d_model)
            - head_dim: int
        """
        pass

    @staticmethod
    def _resolve_device(device: DeviceOption) -> str:
        """Resolve DeviceOption enum to actual device string."""
        if device == DeviceOption.AUTO:
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device.value

    def extract_activations(
        self,
        prompts: list[str],
        layers: list[int],
        components: list[ComponentOption],
        position: PositionOption,
        batch_size: int,
        activation_checkpoint_path: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> dict[tuple[int, ComponentOption], np.ndarray]: # layer, component -> activations
        """
        Extract activations for given prompts.

        Args:
            prompts: List of text inputs
            layers: List of layer indices to extract
            components: List of components to extract
            position: Which token position to extract
            batch_size: Batch size for processing
            activation_checkpoint_path: Optional directory to save/load checkpoints.
                           If provided, will save after each batch and resume from existing checkpoints.
            auto_cleanup: If True (default), delete checkpoint directory after successful completion.
                         Set to False to keep checkpoints for later manual cleanup.

        Returns:
            Dictionary mapping (layer, component) to activations.
            Shape of each array: (n_prompts, hidden_dim)
        """
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        # Create storage strategy (in-memory or checkpointed)
        storage = create_batch_storage(activation_checkpoint_path, auto_cleanup)

        # Process batches
        for batch_idx, batch_start in enumerate(range(0, len(prompts), batch_size)):
            if storage.should_skip_batch(batch_idx):
                continue

            batch_prompts = prompts[batch_start : batch_start + batch_size]
            batch_results = self._extract_single_batch(
                batch_prompts, layers, components, position
            )
            storage.store_batch(batch_idx, batch_results)

        # Retrieve all results and concatenate
        all_batches = storage.get_all_batches(total_batches)
        final_results = concatenate_batches(all_batches, position)
        # Cleanup (no-op for in-memory, removes files for checkpointed)
        storage.cleanup()

        return final_results

    @abstractmethod
    def _extract_single_batch(
        self,
        batch_prompts: list[str],
        layers: list[int],
        components: list[ComponentOption],
        position: PositionOption,
    ) -> BatchResults:
        """
        Extract activations for a single batch.
        
        Must be implemented by subclasses.
        """
        pass

    def _select_position(
        self, acts: np.ndarray, position: PositionOption
    ) -> np.ndarray:
        """
        Select token position from activations.

        Args:
            acts: Activations with shape (batch, seq, hidden_dim)
            position: Position option enum

        Returns:
            Activations with shape (batch, hidden_dim) or (batch, seq, hidden_dim) for ALL
        """
        if position == PositionOption.LAST:
            return acts[:, -1, :]
        elif position == PositionOption.MEAN:
            return acts.mean(axis=1)
        elif position == PositionOption.ALL:
            return acts
        elif isinstance(position, int):
            try:
                return acts[:, position, :]
            except IndexError:
                raise ValueError(
                    f"Token index {position} is out of range for sequence length {acts.shape[1]}."
                )
        else:
            raise ValueError(f"Unknown PositionOption or invalid index: {position}")
