"""
Activation extraction using TransformerLens.

TransformerLens is the most popular library for mechanistic interpretability.
It provides semantic cache keys and supports 50+ models out of the box.

Installation: pip install transformer-lens
"""

import warnings
# Suppress torch_dtype deprecation warning from TransformerLens/HuggingFace
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

from typing import Optional
import numpy as np
import torch

from easyprobe.extractors.base import (
    ActivationExtractor,
    BatchResults,
    create_batch_storage,
    concatenate_batches,
)
from easyprobe.datamodels import ComponentOption, DeviceOption, PositionOption


class TransformerLensExtractor(ActivationExtractor):
    """
    Activation extraction using TransformerLens.

    TransformerLens provides:
    - Semantic cache keys: cache["resid_post", 5] instead of cryptic indices
    - Built-in support for 50+ models
    - Easy hook-based intervention

    Example:
        from easyprobe.datamodels import ComponentOption, PositionOption
        
        extractor = TransformerLensExtractor("pythia-410m")
        activations = extractor.extract_activations(
            prompts=["Hello world"],
            layers=[0, 5, 10],
            components=[ComponentOption.RESID],
            position=PositionOption.LAST,
            batch_size=8,
        )
    """

    # Map component options to TransformerLens cache keys (constant, shared across instances)
    _COMPONENT_KEYS = {
        ComponentOption.RESID: "resid_post",
        ComponentOption.ATTN: "attn_out",
        ComponentOption.MLP: "mlp_out",
    }

    def __init__(self, model_name: str, device: DeviceOption = DeviceOption.AUTO):
        """
        Initialize TransformerLens extractor.

        Args:
            model_name: Model identifier (e.g., "pythia-410m", "gpt2-small")
            device: Device to run on (DeviceOption enum)
        """
        try:
            from transformer_lens import HookedTransformer
        except ImportError:
            raise ImportError(
                "TransformerLens not installed. Install with:\n"
                "  pip install transformer-lens"
            )

        # Convert DeviceOption to string for TransformerLens
        device_str = self._resolve_device(device)
        self.model = HookedTransformer.from_pretrained(model_name, device=device_str)
        self.model_name = model_name
        self.device = device

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

    def get_model_config(self) -> dict:
        """Return model configuration."""
        return {
            "n_layers": self.model.cfg.n_layers,
            "n_heads": self.model.cfg.n_heads,
            "hidden_dim": self.model.cfg.d_model,
            "head_dim": self.model.cfg.d_head,
            "device": self._resolve_device(self.device),
        }

    def _extract_single_batch(
        self,
        batch_prompts: list[str],
        layers: list[int],
        components: list[ComponentOption],
        position: PositionOption,
    ) -> BatchResults:
        """
        Extract activations for a single batch of prompts.

        Args:
            batch_prompts: Prompts for this batch
            layers: Layer indices to extract
            components: Components to extract
            position: Token position selection

        Returns:
            Dictionary mapping (layer, component) to batch activations.
        """
        from transformer_lens import utils

        # Construct filter to ONLY cache what we need (saves massive memory/compute)
        names_filter = []
        for layer in layers:
            for component in components:
                short_name = self._COMPONENT_KEYS[component]
                full_name = utils.get_act_name(short_name, layer)
                names_filter.append(full_name)

        # Run model and cache activations
        with torch.no_grad():
            _, cache = self.model.run_with_cache(batch_prompts, names_filter=names_filter)

        # Extract requested activations
        batch_results: BatchResults = {}
        for layer in layers:
            for component in components:
                cache_key = self._COMPONENT_KEYS[component]

                # Get activations: shape (batch, seq, hidden_dim)
                acts = cache[cache_key, layer].detach().cpu().numpy()

                # Select position
                acts = self._select_position(acts, position)

                batch_results[(layer, component)] = acts

        return batch_results

    def extract_activations(
        self,
        prompts: list[str],
        layers: list[int],
        components: list[ComponentOption],
        position: PositionOption,
        batch_size: int,
        checkpoint_dir: Optional[str] = None,
        auto_cleanup: bool = True,
    ) -> dict[tuple[int, ComponentOption], np.ndarray]:
        """
        Extract activations for given prompts using TransformerLens.

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
        """
        total_batches = (len(prompts) + batch_size - 1) // batch_size

        # Create storage strategy (in-memory or checkpointed)
        storage = create_batch_storage(checkpoint_dir, auto_cleanup)

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
