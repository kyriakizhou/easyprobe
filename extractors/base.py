"""
Base class for activation extraction backends.
"""

from abc import ABC, abstractmethod
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np

from easyprobe.datamodels import ComponentOption, PositionOption


# Type alias for activation keys
ActivationKey = tuple[int, ComponentOption]
BatchResults = dict[ActivationKey, np.ndarray]


class BatchStorage(ABC):
    """
    Abstract base class for batch storage strategies.

    Implementations handle how batch results are stored during extraction
    and how they are retrieved for final concatenation.
    """

    @abstractmethod
    def should_skip_batch(self, batch_idx: int) -> bool:
        """Check if a batch has already been processed."""
        pass

    @abstractmethod
    def store_batch(self, batch_idx: int, results: BatchResults) -> None:
        """Store results from a single batch."""
        pass

    @abstractmethod
    def get_all_batches(self, total_batches: int) -> dict[ActivationKey, list[np.ndarray]]:
        """Retrieve all batch results for concatenation."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Clean up any resources (e.g., checkpoint files)."""
        pass


class InMemoryBatchStorage(BatchStorage):
    """
    Store batch results in RAM.

    Simple and fast, but uses memory proportional to total activations.
    Use this when you don't need crash recovery and have enough RAM.
    """

    def __init__(self):
        self._results: dict[ActivationKey, list[np.ndarray]] = {}

    def should_skip_batch(self, batch_idx: int) -> bool:
        return False  # Never skip - no persistence

    def store_batch(self, batch_idx: int, results: BatchResults) -> None:
        for key, acts in results.items():
            if key not in self._results:
                self._results[key] = []
            self._results[key].append(acts)

    def get_all_batches(self, total_batches: int) -> dict[ActivationKey, list[np.ndarray]]:
        return self._results

    def cleanup(self) -> None:
        pass  # Nothing to clean up


class CheckpointedBatchStorage(BatchStorage):
    """
    Store batch results to disk, loading only at the end.

    Memory efficient for large models - only one batch in RAM at a time.
    Supports resuming from crashes by tracking completed batches.
    """

    def __init__(self, checkpoint_dir: str, auto_cleanup: bool = True):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._auto_cleanup = auto_cleanup
        self._completed_batches: set[int] = set()

        # Load existing progress
        progress = self._load_progress()
        self._completed_batches = set(progress.get("completed_batches", []))
        if self._completed_batches:
            print(f"[EasyProbe] Resuming from checkpoint: {len(self._completed_batches)} batches already completed")

    def should_skip_batch(self, batch_idx: int) -> bool:
        return batch_idx in self._completed_batches

    def store_batch(self, batch_idx: int, results: BatchResults) -> None:
        # Save batch to disk
        batch_file = self._dir / f"batch_{batch_idx:04d}.npz"
        save_dict = {}
        for (layer, component), acts in results.items():
            key = f"{layer}_{component.value}"
            save_dict[key] = acts
        np.savez(batch_file, **save_dict)

        # Update progress
        self._completed_batches.add(batch_idx)
        self._save_progress({"completed_batches": list(self._completed_batches)})

    def get_all_batches(self, total_batches: int) -> dict[ActivationKey, list[np.ndarray]]:
        """Load all batches from disk."""
        print(f"[EasyProbe] Loading {total_batches} batches from checkpoints...")
        results: dict[ActivationKey, list[np.ndarray]] = {}

        for batch_idx in range(total_batches):
            batch_file = self._dir / f"batch_{batch_idx:04d}.npz"
            loaded = np.load(batch_file)

            for key in loaded.files:
                parts = key.split("_")
                layer = int(parts[0])
                component = ComponentOption(parts[1])
                activation_key = (layer, component)

                if activation_key not in results:
                    results[activation_key] = []
                results[activation_key].append(loaded[key])

        return results

    def cleanup(self) -> None:
        if self._auto_cleanup and self._dir.exists():
            shutil.rmtree(self._dir)
            print(f"[EasyProbe] Cleaned up checkpoint directory: {self._dir}")

    def _load_progress(self) -> dict:
        progress_file = self._dir / "progress.json"
        if progress_file.exists():
            with open(progress_file, "r") as f:
                return json.load(f)
        return {"completed_batches": []}

    def _save_progress(self, progress: dict) -> None:
        progress_file = self._dir / "progress.json"
        with open(progress_file, "w") as f:
            json.dump(progress, f)


def create_batch_storage(
    checkpoint_dir: Optional[str] = None,
    auto_cleanup: bool = True,
) -> BatchStorage:
    """
    Factory function to create the appropriate batch storage.

    Args:
        checkpoint_dir: If provided, use checkpointed storage. Otherwise, use in-memory.
        auto_cleanup: If True, delete checkpoint files after successful completion.

    Returns:
        BatchStorage instance.
    """
    if checkpoint_dir:
        return CheckpointedBatchStorage(checkpoint_dir, auto_cleanup)
    return InMemoryBatchStorage()


def concatenate_batches(
    batch_results: dict[ActivationKey, list[np.ndarray]],
    position: PositionOption,
) -> dict[ActivationKey, np.ndarray]:
    """
    Concatenate batch results into final arrays.

    Handles variable sequence lengths for PositionOption.ALL by padding.
    """
    if position == PositionOption.ALL:
        concatenated = {}
        for key, vals in batch_results.items():
            max_seq_len = max(v.shape[1] for v in vals)
            padded_vals = []
            for v in vals:
                if v.shape[1] < max_seq_len:
                    pad_width = ((0, 0), (0, max_seq_len - v.shape[1]), (0, 0))
                    v = np.pad(v, pad_width, mode='constant', constant_values=0)
                padded_vals.append(v)
            concatenated[key] = np.concatenate(padded_vals, axis=0)
        return concatenated
    else:
        return {key: np.concatenate(vals, axis=0) for key, vals in batch_results.items()}


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
        layers: list[int],  # TODO: consolidate with LayerSpec
        components: list[ComponentOption],
        position: PositionOption,
        batch_size: int,
        checkpoint_dir: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> dict[tuple[int, ComponentOption], np.ndarray]:
        """
        Extract activations for given prompts.

        Args:
            prompts: List of text inputs
            layers: List of layer indices to extract
            components: List of components to extract
            position: Which token position to extract
            batch_size: Batch size for processing
            checkpoint_dir: Optional directory to save/load checkpoints.
                           If provided, will save after each batch and resume from existing checkpoints.
            auto_cleanup: If True (default), delete checkpoint directory after successful completion.
                         Set to False to keep checkpoints for later manual cleanup.

        Returns:
            Dictionary mapping (layer, component) to activations.
            Shape of each array: (n_prompts, hidden_dim)
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
