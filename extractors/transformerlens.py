"""
Activation extraction using TransformerLens.

TransformerLens is the most popular library for mechanistic interpretability.
It provides semantic cache keys and supports 50+ models out of the box.

Installation: pip install transformer-lens
"""

import warnings
# Suppress torch_dtype deprecation warning from TransformerLens/HuggingFace
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

import torch

from easyprobe.extractors.base import (
    ActivationExtractor,
    BatchResults,
)
from easyprobe.models.data_models import ComponentOption, DeviceOption, PositionOption


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

    def __init__(self, model_name: str, device: DeviceOption = DeviceOption.AUTO, torch_dtype=None):
        """
        Initialize TransformerLens extractor.

        Args:
            model_name: Model identifier (e.g., "pythia-410m", "gpt2-small")
            device: Device to run on (DeviceOption enum)
            torch_dtype: Optional torch dtype (e.g., torch.bfloat16, torch.float16).
                         If None, uses model's default dtype.
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
        load_kwargs = {"device": device_str}
        if torch_dtype is not None:
            load_kwargs["dtype"] = torch_dtype  # TransformerLens uses `dtype`, not `torch_dtype`
        self.model = HookedTransformer.from_pretrained(model_name, **load_kwargs)
        self.model_name = model_name
        self.device = device



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
                acts = cache[cache_key, layer].detach().cpu().float().numpy()

                # Select position
                acts = self._select_position(acts, position)

                batch_results[(layer, component)] = acts

        return batch_results


