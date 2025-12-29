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

from easyprobe.extractors.base import ActivationExtractor
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
        }

    def extract_activations(
        self,
        prompts: list[str],
        layers: list[int],
        components: list[ComponentOption],
        position: PositionOption,
        batch_size: int,
    ) -> dict[tuple[int, ComponentOption, PositionOption], np.ndarray]:
        results: dict[tuple[int, str, Optional[int]], list[np.ndarray]] = {}

        try:
            from transformer_lens import utils
        except ImportError:
            pass  # Already checked in __init__ for main package

        # Process in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]

            # Construct filter to ONLY cache what we need (saves massive memory/compute)
            names_filter = []
            for layer in layers:
                for component in components:
                    short_name = self._COMPONENT_KEYS[component]
                    full_name = utils.get_act_name(short_name, layer)
                    names_filter.append(full_name)

            # Run model and cache activatons
            with torch.no_grad():
                _, cache = self.model.run_with_cache(batch_prompts, names_filter=names_filter)

            # Extract requested activations
            for layer in layers:
                for component in components:
                    cache_key = self._COMPONENT_KEYS[component]

                    # Get activations: shape (batch, seq, hidden_dim)
                    acts = cache[cache_key, layer].detach().cpu().numpy()

                    # Select position
                    acts = self._select_position(acts, position)

                    key = (layer, component, None)
                    if key not in results:
                        results[key] = []
                    results[key].append(acts)



        # Concatenate batches
        # For PositionOption.ALL, we need to handle variable sequence lengths
        # by padding to the maximum sequence length across all batches
        if position == PositionOption.ALL:
            concatenated = {}
            for key, vals in results.items():
                # Find max sequence length across all batches
                max_seq_len = max(v.shape[1] for v in vals)
                # Pad each batch to max_seq_len
                padded_vals = []
                for v in vals:
                    if v.shape[1] < max_seq_len:
                        pad_width = ((0, 0), (0, max_seq_len - v.shape[1]), (0, 0))
                        v = np.pad(v, pad_width, mode='constant', constant_values=0)
                    padded_vals.append(v)
                concatenated[key] = np.concatenate(padded_vals, axis=0)
            return concatenated
        else:
            return {key: np.concatenate(vals, axis=0) for key, vals in results.items()}
