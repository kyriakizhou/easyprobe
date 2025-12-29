"""
Base class for activation extraction backends.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from easyprobe.datamodels import ComponentOption, PositionOption


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

    @abstractmethod
    def extract_activations(
        self,
        prompts: list[str],
        layers: list[int], # TODO: consolidate with LayerSepc
        components: list[ComponentOption],
        position: PositionOption,
        batch_size: int,
    ) -> dict[tuple[int, str, Optional[int]], np.ndarray]:
        """
        Extract activations for given prompts.

        Args:
            prompts: List of text inputs
            layers: List of layer indices to extract
            components: List of components to extract
            position: Which token position to extract
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping (layer, component, head) to activations.
            - head is None for component-level activations
            - head is int for head-level activations
            - Shape of each array: (n_prompts, hidden_dim) or (n_prompts, head_dim)
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
