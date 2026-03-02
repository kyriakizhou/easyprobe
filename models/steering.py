"""
Steering context implementations for different backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Any
import torch

from easyprobe.models.data_models import ComponentOption


class SteeringContext(ABC):
    """
    Abstract base class for steering contexts.
    
    Provides a common interface for steering models with different backends.
    """
    def __init__(
        self, 
        model: Any, 
        layer: int, 
        component: ComponentOption, 
        vector: torch.Tensor, 
        multiplier: float
    ):
        self.model = model
        self.layer = layer
        self.component = component
        self.vector = vector
        self.multiplier = multiplier
        
    @abstractmethod
    def __enter__(self):
        """Apply steering when entering context."""
        pass
        
    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Remove steering when exiting context."""
        pass

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """
        Generate text with steering applied at every generation step.
        
        Args:
            prompt: Input text to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            
        Returns:
            Full generated text (prompt + new tokens).
        """
        pass


class TransformerLensSteeringContext(SteeringContext):
    """
    Steering context for TransformerLens models (using hooks).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hook_handle = None
        
    def __enter__(self):
        self.hook_handle = self._register_hook()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hook_handle:
            self.hook_handle.remove()
            self.hook_handle = None

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """Generate text with steering using TransformerLens hooks."""
        with self:
            tokens = self.model.to_tokens(prompt)
            output = self.model.generate(tokens, max_new_tokens=max_new_tokens)
            return self.model.to_string(output[0])
            
    def _register_hook(self):
        """Register the steering hook."""
        hook_name = self._get_hook_name()
        
        def steering_hook(activations, hook):
            steering_vec = self.vector.to(activations.device)
            activations += self.multiplier * steering_vec
            return activations
            
        return self.model.add_hook(hook_name, steering_hook)
        
    def _get_hook_name(self) -> str:
        """Get the hook name for transformer_lens."""
        if self.component == ComponentOption.RESID:
            return f"blocks.{self.layer}.hook_resid_post"
        elif self.component == ComponentOption.ATTN:
            return f"blocks.{self.layer}.attn.hook_z" 
        elif self.component == ComponentOption.MLP:
            return f"blocks.{self.layer}.mlp.hook_post"
        else:
            raise NotImplementedError(f"Steering not implemented for component {self.component}")


class NNSightSteeringContext(SteeringContext):
    """
    Steering context for NNSight models (using graph modification).
    
    For generation with steering, use the `generate()` method which applies
    steering at every autoregressive step via `generator.all()`.
    
    The context manager (`__enter__`/`__exit__`) can still be used inside
    a manually constructed `model.trace()` or `model.generate()` block.
    """
    def __enter__(self):
        self._steer_nnsight()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def generate(self, prompt: str, max_new_tokens: int = 10) -> str:
        """
        Generate text with steering applied at every generation step.
        
        Uses NNSight's `generator.all()` to apply the steering intervention
        and capture `lm_head` logits at each autoregressive step.
        
        Args:
            prompt: Input text to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            
        Returns:
            Full generated text (prompt + new tokens).
        """
        generated_logits = []
        lm_head = self._get_lm_head()
        
        with self.model.generate(max_new_tokens=max_new_tokens, remote=False) as generator:
            with generator.invoke(prompt) as invoker:
                with generator.all():
                    # Apply steering at this generation step
                    self._steer_nnsight()
                    # Capture logits for decoding
                    logits = lm_head.output.save()
                    generated_logits.append(logits)
        
        token_ids = [torch.argmax(t[0, -1, :]).item() for t in generated_logits]
        new_text = self.model.tokenizer.decode(token_ids)
        return f"{prompt}{new_text}"

    @staticmethod
    def generate_text(model, prompt: str, max_new_tokens: int = 10) -> str:
        """
        Generate text WITHOUT steering using NNSight's generation API.
        
        This is the un-steered counterpart to `generate()`, using the same
        `generator.all()` + `lm_head.output.save()` pattern for output capture.
        
        Args:
            model: NNSight LanguageModel.
            prompt: Input text to continue from.
            max_new_tokens: Maximum number of tokens to generate.
            
        Returns:
            Full generated text (prompt + new tokens).
        """
        generated_logits = []
        lm_head = NNSightSteeringContext._find_lm_head(model)
        
        with model.generate(max_new_tokens=max_new_tokens, remote=False) as generator:
            with generator.invoke(prompt) as invoker:
                with generator.all():
                    logits = lm_head.output.save()
                    generated_logits.append(logits)
        
        token_ids = [torch.argmax(t[0, -1, :]).item() for t in generated_logits]
        new_text = model.tokenizer.decode(token_ids)
        return f"{prompt}{new_text}"

    def _steer_nnsight(self):
        """Apply steering for NNSight models via graph modification."""
        layer_mod = self._get_nnsight_layer()
        target = self._get_nnsight_target(layer_mod)
        # Cast vector to model's dtype (e.g. bfloat16) to avoid dtype mismatch
        model_dtype = next(self.model._model.parameters()).dtype
        vector = self.vector.to(device=self.model.device, dtype=model_dtype)

        # Modify the output in-place in the graph
        if isinstance(target.output, tuple):
            target.output[0][:] = target.output[0] + self.multiplier * vector
        else:
            target.output = target.output + (self.multiplier * vector)

    def _get_lm_head(self):
        """Find the lm_head module for this model instance."""
        return self._find_lm_head(self.model)

    @staticmethod
    def _find_lm_head(model):
        """Find the lm_head module for an NNSight model."""
        if hasattr(model, "lm_head"):
            return model.lm_head
        elif hasattr(model, "embed_out"):
            return model.embed_out  # Pythia/GPT-NeoX
        raise ValueError(f"Could not find lm_head in model {type(model)}")

    def _get_nnsight_layer(self):
        """Find the layer module for NNSight models."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer]  # OLMo, LLaMA
        elif hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers[self.layer]  # Pythia
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer]  # GPT2
        elif hasattr(self.model, "layers"):
            return self.model.layers[self.layer]
        else:
            raise ValueError(f"Could not find layers in NNSight model {type(self.model)}")

    def _get_nnsight_target(self, layer_mod):
        """Get the component module whose output we modify."""
        if self.component == ComponentOption.RESID:
            return layer_mod
        elif self.component == ComponentOption.ATTN:
            if hasattr(layer_mod, "self_attn"): return layer_mod.self_attn
            if hasattr(layer_mod, "attn"): return layer_mod.attn
            if hasattr(layer_mod, "attention"): return layer_mod.attention
        elif self.component == ComponentOption.MLP:
            if hasattr(layer_mod, "mlp"): return layer_mod.mlp
            if hasattr(layer_mod, "feed_forward"): return layer_mod.feed_forward
            if hasattr(layer_mod, "ffn"): return layer_mod.ffn
            
        raise ValueError(f"Could not find component {self.component} in layer module")


class DualNNSightSteeringContext(NNSightSteeringContext):
    """
    Implementation of 'Dual Steering' via Regularized Newton Updates.
    
    As described in "Dual Steering with a Linear Probe", this method attempts to
    modify the target concept while minimizing 'off-target' effects by accounting
    for the information geometry (the covariance/Hessian) of the model's output space.
    """
    def __init__(
        self, 
        model: Any, 
        layer: int, 
        component: ComponentOption, 
        vector: torch.Tensor, 
        multiplier: float,
        iterations: int = 3,
        lambda_reg: float = 0.1,
        top_k_cov: int = 100
    ):
        super().__init__(model, layer, component, vector, multiplier)
        self.iterations = iterations
        self.lambda_reg = lambda_reg
        self.top_k_cov = top_k_cov

    def _steer_nnsight(self):
        """Apply Dual Steering path tracing in the NNSight graph."""
        layer_mod = self._get_nnsight_layer()
        target = self._get_nnsight_target(layer_mod)
        lm_head = self._get_lm_head()
        
        # 1. Get current state and model-specific constants
        model_dtype = next(self.model._model.parameters()).dtype
        W = self.vector.to(device=self.model.device, dtype=model_dtype)
        
        # 2. Iterative Newton Update (Algorithm 1)
        current_x = target.output[0] if isinstance(target.output, tuple) else target.output
        
        # Ensure x is [Batch, HiddenDim] – during generation Seq is usually 1
        # but NNSight proxies often keep the dimensions [Batch, Seq, HiddenDim]
        if current_x.ndim == 3:
            current_x_2d = current_x[:, -1, :] # Take last token position
        else:
            current_x_2d = current_x
            
        U = lm_head.weight # [Vocab, HiddenDim]
        
        for _ in range(self.iterations):
            # Compute partial softmax to get local covariance
            logits = torch.matmul(current_x_2d, U.t())
            probs = torch.softmax(logits, dim=-1) # [Batch, Vocab]
            
            # Select top-k tokens for covariance estimation (on the first batch item)
            # Generation usually has batch=1, but we iterate for safety if needed
            # For simplicity and NNSight stability, we calculate based on batch index 0
            vals, idxs = torch.topk(probs[0], self.top_k_cov, dim=-1)
            top_probs = vals / vals.sum(dim=-1, keepdim=True) # [TopK]
            top_U = U[idxs] # [TopK, HiddenDim]
            
            # center the U matrix
            avg_U = torch.sum(top_U * top_probs.unsqueeze(-1), dim=0, keepdim=True) # [1, HiddenDim]
            centered_U = top_U - avg_U # [TopK, HiddenDim]
            
            # Covariance matrix (Hessian approximation) [HiddenDim, HiddenDim]
            cov = torch.matmul(centered_U.t(), centered_U * top_probs.unsqueeze(-1))
            
            # Regularized Newton: (Cov + lambda*I) v = W
            reg_matrix = cov + self.lambda_reg * torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
            
            # Solve linear system for the optimal 'Dual' direction v
            # Use float32 because solve() is not implemented for bfloat16 on CUDA/CUSolver
            v = torch.linalg.solve(
                reg_matrix.to(torch.float32), 
                W.to(torch.float32)
            ).to(model_dtype)
            
            # Step in primal space (normalize v to maintain stability per step)
            step_size = self.multiplier / self.iterations
            current_x_2d = current_x_2d + (step_size * (v / torch.norm(v)))

        # 3. Update the graph output
        # Re-inject the modified activations back into the correct shape
        if current_x.ndim == 3:
            # Reconstruct batch/seq dim
            final_x = current_x.clone()
            final_x[:, -1, :] = current_x_2d
        else:
            final_x = current_x_2d

        if isinstance(target.output, tuple):
            target.output[0][:] = final_x
        else:
            target.output = final_x
