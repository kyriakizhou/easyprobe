"""
Activation extraction using NNSight.

NNSight is a library that extends PyTorch for interpretability research.
It works with any HuggingFace model without reimplementation, preserving
exact model behavior.

This extractor is designed for models not supported by TransformerLens,
such as OLMo-3 and its training checkpoints. It is also generally better
for biggers models (70B+) as it supports efficient sharding and offloading
via the underlying HuggingFace/Accelerate integration.

Installation: pip install nnsight transformers
"""

import logging
from typing import Optional

import numpy as np
import torch

from easyprobe.extractors.base import (
    ActivationExtractor,
    BatchResults,
)
from easyprobe.models.data_models import ComponentOption, DeviceOption, PositionOption


logger = logging.getLogger(__name__)


class NNSightExtractor(ActivationExtractor):
    """
    Activation extraction using NNSight.
    - Works with any HuggingFace model
    - Preserves exact model behavior (no numerical mismatch)
    - Memory efficient (uses fake tensors for validation)
    - Supports latest transformers versions
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
        # If device is AUTO and CUDA is available, use "auto" for device_map
        # to enable multi-GPU sharding (which base _resolve_device returns as "cuda")
        device_map = self.device_str
        if device == DeviceOption.AUTO and torch.cuda.is_available():
            device_map = "auto"

        load_kwargs = {"device_map": device_map}
        if torch_dtype is not None:
            load_kwargs["torch_dtype"] = torch_dtype
        if revision is not None:
            load_kwargs["revision"] = revision

        self.model = LanguageModel(model_name, dispatch=False, **load_kwargs)
        
        # Ensure correct tokenizer settings
        if self.model.tokenizer.pad_token is None:
             self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        
        # Enforce left padding for correct PositionOption.LAST extraction
        self.model.tokenizer.padding_side = "left"
        logger.debug(f"Tokenizer padding set to: {self.model.tokenizer.padding_side}")

        # Cache model config
        self._config = self._extract_model_config()



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
            "device": self.device_str,
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

    def _detect_architecture(self) -> str:
        """Detect the model architecture type."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return "llama"  # OLMo, LLaMA, Mistral, Qwen all use this
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return "gpt_neox"  # GPT-NeoX, Pythia
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return "gpt2"
        elif hasattr(self.model, "layers"):
            return "generic"
        else:
            return "unknown"

    def _get_component_output(self, layer, component: ComponentOption, architecture: str):
        """
        Get the output proxy for a specific component.

        Returns the NNSight proxy that will capture the component's output.

        Note: Different architectures have different module structures:
        - OLMo/LLaMA/Mistral: layer.self_attn, layer.mlp
        - GPT-NeoX/Pythia: only supports RESID (residual stream) due to tuple outputs in attention/mlp modules.
        - GPT2: layer.attn, layer.mlp

        Limitations:
        - GPT-NeoX/Pythia: Only RESID component is supported due to NNSight tracing
          constraints. ATTN and MLP require different access patterns that don't
          work with NNSight's proxy system for this architecture.
        """
        if component == ComponentOption.RESID:
            # Residual stream = layer output (after LayerNorm in most models)
            # Note: Some models return a tensor directly, others return a tuple.
            # We return layer.output directly and handle the tuple case in extract_activations
            # after the trace completes (when we can inspect the actual type).
            return layer.output

        elif component == ComponentOption.ATTN:
            # Attention output - architecture specific
            if architecture == "gpt_neox":
                # GPT-NeoX/Pythia: The attention module uses a complex tuple output
                # that doesn't work well with NNSight's tracing. For now, we skip
                # this component for GPT-NeoX models.
                raise ValueError(
                    f"ATTN component extraction is not supported for GPT-NeoX/Pythia models "
                    f"due to NNSight tracing limitations. Use RESID component instead, or "
                    f"use TransformerLens backend for component-level analysis."
                )
            elif hasattr(layer, "self_attn"):
                return layer.self_attn.output
            elif hasattr(layer, "attn"):
                return layer.attn.output
            elif hasattr(layer, "attention"):
                return layer.attention.output
            else:
                raise ValueError(
                    f"Could not find attention module in layer. "
                    f"Available: {dir(layer)}"
                )

        elif component == ComponentOption.MLP:
            # MLP output
            if architecture == "gpt_neox":
                # GPT-NeoX/Pythia: Similar issue with MLP output access
                raise ValueError(
                    f"MLP component extraction is not supported for GPT-NeoX/Pythia models "
                    f"due to NNSight tracing limitations. Use RESID component instead, or "
                    f"use TransformerLens backend for component-level analysis."
                )
            elif hasattr(layer, "mlp"):
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
        architecture = self._detect_architecture()
        saved_activations = {}

        # IMPORTANT: Components must be accessed in forward-pass order to avoid
        # NNSight's OutOfOrderError. The order is: ATTN -> MLP -> RESID (layer output)
        forward_order = [ComponentOption.ATTN, ComponentOption.MLP, ComponentOption.RESID]
        ordered_components = [c for c in forward_order if c in components]

        # Manually tokenize to ensure left padding is respected (NNSight can be flaky with raw strings)
        inputs = self.model.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        
        with self.model.trace(inputs, remote=False):
            for layer_idx in layers:
                layer = self._get_layer_module(layer_idx)

                for component in ordered_components:
                    output_proxy = self._get_component_output(layer, component, architecture)
                    key = (layer_idx, component)
                    saved_activations[key] = output_proxy.save()

        # Convert saved activations to numpy arrays
        batch_results: BatchResults = {}
        for (layer_idx, component), saved in saved_activations.items():
            acts = saved

            # Handle tuple outputs (some modules return (hidden_states, ...))
            if isinstance(acts, tuple):
                acts = acts[0]

            # Convert to numpy
            acts = acts.detach().cpu().float().numpy()

            # Some models return 2D tensors (batch, hidden_dim) instead of 3D
            if acts.ndim == 2:
                acts = acts[:, np.newaxis, :]

            # Select position
            acts = self._select_position(acts, position)

            batch_results[(layer_idx, component)] = acts

        return batch_results
