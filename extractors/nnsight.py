"""
Activation extraction using NNSight.

NNSight is a library that extends PyTorch for interpretability research.
It works with any HuggingFace model without reimplementation, preserving
exact model behavior.

This extractor is designed for models not supported by TransformerLens,
such as OLMo-3 and its training checkpoints.

Installation: pip install nnsight transformers
"""

from typing import Optional

import numpy as np
import torch

from easyprobe.extractors.base import ActivationExtractor
from easyprobe.datamodels import ComponentOption, DeviceOption, PositionOption


class NNSightExtractor(ActivationExtractor):
    """
    Activation extraction using NNSight.

    NNSight provides:
    - Works with any HuggingFace model (no reimplementation needed)
    - Preserves exact model behavior (no numerical mismatch)
    - Memory efficient (uses fake tensors for validation)
    - Supports latest transformers versions

    Example:
        from easyprobe.datamodels import ComponentOption, PositionOption

        extractor = NNSightExtractor("allenai/OLMo-2-7B-1124")
        activations = extractor.extract_activations(
            prompts=["Hello world"],
            layers=[0, 5, 10],
            components=[ComponentOption.RESID],
            position=PositionOption.LAST,
            batch_size=8,
        )

    Supported architectures:
        - OLMo (allenai/OLMo-*)
        - LLaMA (meta-llama/*)
        - Qwen (Qwen/*)
        - Mistral (mistralai/*)
        - And any other HuggingFace causal LM
    """

    def __init__(
        self,
        model_name: str,
        device: DeviceOption = DeviceOption.AUTO,
        torch_dtype: Optional[torch.dtype] = None,
        revision: Optional[str] = None,
    ):
        """
        Initialize NNSight extractor.

        Args:
            model_name: HuggingFace model identifier (e.g., "allenai/OLMo-2-7B-1124")
            device: Device to run on (DeviceOption enum)
            torch_dtype: Optional dtype (e.g., torch.float16, torch.bfloat16).
                         If None, uses model's default dtype.
            revision: Optional git revision (branch, tag, or commit hash) to load.
                      Useful for loading specific training checkpoints.
                      E.g., "stage1-step896000" for OLMo-3 checkpoints.
        """
        try:
            from nnsight import LanguageModel
        except ImportError:
            raise ImportError(
                "NNSight not installed. Install with:\n"
                "  pip install nnsight transformers"
            )

        self.model_name = model_name
        self.revision = revision
        self.device_option = device
        self.device_str = self._resolve_device(device)

        # Load model with NNSight wrapper
        # NNSight's LanguageModel wraps HuggingFace models
        load_kwargs = {"device_map": self.device_str}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if revision is not None:
            load_kwargs["revision"] = revision

        self.model = LanguageModel(model_name, **load_kwargs)

        # Cache model config
        self._config = self._extract_model_config()

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

    def _extract_model_config(self) -> dict:
        """Extract model configuration from HuggingFace config."""
        hf_config = self.model.config

        # Different models use different config attribute names
        # Try common patterns
        n_layers = getattr(hf_config, "num_hidden_layers", None) or \
                   getattr(hf_config, "n_layers", None) or \
                   getattr(hf_config, "num_layers", None)

        n_heads = getattr(hf_config, "num_attention_heads", None) or \
                  getattr(hf_config, "n_heads", None)

        hidden_dim = getattr(hf_config, "hidden_size", None) or \
                     getattr(hf_config, "d_model", None)

        head_dim = getattr(hf_config, "head_dim", None)
        if head_dim is None and hidden_dim and n_heads:
            head_dim = hidden_dim // n_heads

        return {
            "n_layers": n_layers,
            "n_heads": n_heads,
            "hidden_dim": hidden_dim,
            "head_dim": head_dim,
        }

    def get_model_config(self) -> dict:
        """Return model configuration."""
        return self._config

    def _get_layer_module(self, layer_idx: int):
        """
        Get the layer module for a given layer index.

        Different model architectures have different module paths:
        - OLMo: model.model.layers[i]
        - LLaMA: model.model.layers[i]
        - GPT2: model.transformer.h[i]
        - GPT-NeoX/Pythia: model.gpt_neox.layers[i]
        """
        # Try common patterns for layer access
        # Most modern models (OLMo, LLaMA, Mistral, Qwen) use model.model.layers
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[layer_idx]
        # GPT-NeoX/Pythia models use gpt_neox.layers
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers[layer_idx]
        # GPT2-style models use transformer.h
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[layer_idx]
        # Fallback: try model.layers directly
        elif hasattr(self.model, "layers"):
            return self.model.layers[layer_idx]
        else:
            raise ValueError(
                f"Could not find layer modules for model {self.model_name}. "
                "Please check the model architecture."
            )

    def _get_component_output(self, layer, component: ComponentOption):
        """
        Get the output proxy for a specific component.

        Returns the NNSight proxy that will capture the component's output.
        """
        if component == ComponentOption.RESID:
            # Residual stream = layer output (after LayerNorm in most models)
            return layer.output[0]  # First element is hidden states

        elif component == ComponentOption.ATTN:
            # Attention output
            # Most models: layer.self_attn or layer.attn
            if hasattr(layer, "self_attn"):
                return layer.self_attn.output[0]
            elif hasattr(layer, "attn"):
                return layer.attn.output[0]
            elif hasattr(layer, "attention"):
                return layer.attention.output[0]
            else:
                raise ValueError(
                    f"Could not find attention module in layer. "
                    f"Available: {dir(layer)}"
                )

        elif component == ComponentOption.MLP:
            # MLP output
            if hasattr(layer, "mlp"):
                return layer.mlp.output
            elif hasattr(layer, "feed_forward"):
                return layer.feed_forward.output
            elif hasattr(layer, "ffn"):
                return layer.ffn.output
            else:
                raise ValueError(
                    f"Could not find MLP module in layer. "
                    f"Available: {dir(layer)}"
                )

        else:
            raise ValueError(f"Unknown component: {component}")

    def extract_activations(
        self,
        prompts: list[str],
        layers: list[int],
        components: list[ComponentOption],
        position: PositionOption,
        batch_size: int,
    ) -> dict[tuple[int, ComponentOption, Optional[int]], np.ndarray]:
        """
        Extract activations for given prompts using NNSight.

        Args:
            prompts: List of text inputs
            layers: List of layer indices to extract
            components: List of components to extract
            position: Which token position to extract
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping (layer, component, head) to activations.
        """
        results: dict[tuple[int, ComponentOption, Optional[int]], list[np.ndarray]] = {}

        # Process in batches
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[batch_start : batch_start + batch_size]

            # Dictionary to store saved activations for this batch
            saved_activations = {}

            # Use NNSight tracing context
            with self.model.trace(batch_prompts) as tracer:
                for layer_idx in layers:
                    layer = self._get_layer_module(layer_idx)

                    for component in components:
                        # Get the output proxy for this component
                        output_proxy = self._get_component_output(layer, component)

                        # Save the activation (this creates a reference that will be populated)
                        key = (layer_idx, component)
                        saved_activations[key] = output_proxy.save()

            # After tracing completes, extract the saved activations
            for (layer_idx, component), saved in saved_activations.items():
                # Get the actual tensor value
                # NNSight 0.5.x: .save() returns tensor directly after trace
                # NNSight 0.3.x: .save() returns proxy with .value attribute
                if hasattr(saved, 'value'):
                    acts = saved.value
                else:
                    acts = saved

                # Handle tuple outputs (some modules return (hidden_states, ...))
                if isinstance(acts, tuple):
                    acts = acts[0]

                # Convert to numpy
                acts = acts.detach().cpu().numpy()

                # Select position
                acts = self._select_position(acts, position)

                # Store results
                key = (layer_idx, component, None)
                if key not in results:
                    results[key] = []
                results[key].append(acts)

        # Concatenate batches
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
