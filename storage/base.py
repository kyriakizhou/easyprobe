from abc import ABC, abstractmethod
from typing import Optional, Union
import os
import shutil
import warnings
import numpy as np

from easyprobe.models.data_models import ActivationKey, ComponentOption, PositionOption

SEQ_LENGTHS_KEY = "__seq_lengths__"

BatchResults = dict[ActivationKey, Union[np.ndarray, list[np.ndarray]]]


class BatchStorage(ABC):
    """Abstract base class for batch storage strategies."""

    @abstractmethod
    def should_skip_batch(self, batch_idx: int) -> bool:
        """Return True if batch should be skipped (e.g. already processed)."""
        pass

    @abstractmethod
    def store_batch(self, batch_idx: int, results: BatchResults) -> None:
        """Store results for a batch."""
        pass

    @abstractmethod
    def get_all_batches(self, total_batches: int) -> list[BatchResults]:
        """Retrieve all batch results in order."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Perform any necessary cleanup."""
        pass


class InMemoryBatchStorage(BatchStorage):
    """Stores all batches in memory."""

    def __init__(self):
        self.results: list[Optional[BatchResults]] = []

    def should_skip_batch(self, batch_idx: int) -> bool:
        return False

    def store_batch(self, batch_idx: int, results: BatchResults) -> None:
        while len(self.results) <= batch_idx:
            self.results.append(None)
        self.results[batch_idx] = results

    def get_all_batches(self, total_batches: int) -> list[BatchResults]:
        return [r for r in self.results if r is not None]

    def cleanup(self) -> None:
        pass


class CheckpointedBatchStorage(BatchStorage):
    """Stores batches to disk as .npz files."""

    def __init__(self, activation_checkpoint_path: str, auto_cleanup: bool = True):
        self.activation_checkpoint_path = activation_checkpoint_path
        self.auto_cleanup = auto_cleanup
        os.makedirs(activation_checkpoint_path, exist_ok=True)

    def _get_batch_path(self, batch_idx: int) -> str:
        return os.path.join(self.activation_checkpoint_path, f"batch_{batch_idx}.npz")

    def should_skip_batch(self, batch_idx: int) -> bool:
        return os.path.exists(self._get_batch_path(batch_idx))

    def store_batch(self, batch_idx: int, results: BatchResults) -> None:
        path = self._get_batch_path(batch_idx)

        # Convert (layer, component) to string key for np.savez
        save_dict = {}
        for (layer, component), arr in results.items():
            key = f"layer_{layer}_component_{component.name}"
            save_dict[key] = arr

        np.savez(path, **save_dict)

    def get_all_batches(self, total_batches: int) -> list[BatchResults]:
        all_results = []
        for i in range(total_batches):
            path = self._get_batch_path(i)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing batch {i} at {path}")

            with np.load(path) as data:
                batch_res: BatchResults = {}
                for key_str, arr in data.items():
                    # Parse "layer_X_component_NAME"
                    parts = key_str.split("_")
                    if len(parts) >= 4 and parts[0] == "layer" and parts[2] == "component":
                        layer = int(parts[1])
                        comp_name = "_".join(parts[3:])
                        try:
                            component = ComponentOption[comp_name]
                            batch_res[(layer, component)] = arr
                        except KeyError:
                            warnings.warn(f"Unknown component {comp_name} in {path}")
                all_results.append(batch_res)
        return all_results

    def cleanup(self) -> None:
        if self.auto_cleanup and os.path.exists(self.activation_checkpoint_path):
            shutil.rmtree(self.activation_checkpoint_path)


def create_batch_storage(
    activation_checkpoint_path: Optional[str] = None,
    auto_cleanup: Optional[bool] = True,
) -> BatchStorage:
    """
    Factory function to create the appropriate batch storage.

    Args:
        activation_checkpoint_path: Directory for checkpoints. If None, use in-memory.
        auto_cleanup: Whether to cleanup checkpoints after success.
    """
    if activation_checkpoint_path:
        return CheckpointedBatchStorage(activation_checkpoint_path, auto_cleanup)
    return InMemoryBatchStorage()


def concatenate_batches(
    batch_results: list[BatchResults],
    position: PositionOption,
) -> dict:
    """
    Concatenate results from multiple batches.

    Args:
        batch_results: List of batch dictionaries
        position: Position option used (affects structure)

    Returns:
        Dictionary mapping ActivationKey to full dataset arrays.
        If PositionOption.ALL, includes SEQ_LENGTHS_KEY.
    """
    if not batch_results:
        return {}

    all_keys = set()
    for batch in batch_results:
        all_keys.update(batch.keys())

    concatenated = {}
    for key in all_keys:
        arrays = [batch[key] for batch in batch_results if key in batch]
        if not arrays:
            continue
            
        # Handle sequence padding if arrays have 3 dimensions (batch, seq, hidden)
        if arrays[0].ndim == 3:
            max_seq_len = max(a.shape[1] for a in arrays)
            padded_arrays = []
            for a in arrays:
                if a.shape[1] < max_seq_len:
                    # Pad (0, 0) for batch dim, (0, diff) for seq dim, (0, 0) for hidden dim
                    pad_width = ((0, 0), (0, max_seq_len - a.shape[1]), (0, 0))
                    a = np.pad(a, pad_width, mode='constant', constant_values=0)
                padded_arrays.append(a)
            arrays = padded_arrays
            
        concatenated[key] = np.concatenate(arrays, axis=0)

    # Handle sequence lengths for PositionOption.ALL
    arrays = [batch[SEQ_LENGTHS_KEY] for batch in batch_results if SEQ_LENGTHS_KEY in batch]
    if arrays:
         concatenated[SEQ_LENGTHS_KEY] = np.concatenate(arrays, axis=0)

    return concatenated
