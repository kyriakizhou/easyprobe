"""
Utilities for visualizing probe activations on text.
"""

from typing import Optional
import numpy as np

def generate_highlight_map(
    tokens: list[str], 
    scores: np.ndarray, 
    threshold: float = 0.5,
    token_accuracies: Optional[list[float]] = None
) -> str:
    """
    Generate an HTML string with tokens highlighted by score using a dense colorscale.
    
    Args:
        tokens: List of tokens (strings)
        scores: Array of scores (0-1 probabilities or logits), same length as tokens
        threshold: Neutral threshold (usually 0.5 for probability)
        token_accuracies: Optional list of validation accuracies per token position
        
    Returns:
        HTML string with styled spans.
    """
    if len(tokens) != len(scores):
        raise ValueError(f"Tokens ({len(tokens)}) and scores ({len(scores)}) length mismatch")
    
    if token_accuracies is not None and len(token_accuracies) != len(tokens):
        raise ValueError(f"Tokens ({len(tokens)}) and accuracies ({len(token_accuracies)}) length mismatch")
        
    # 'dense' colorscale from Plotly (matching the heatmap)
    dense_colorscale = [
        (0.0, (230, 240, 240)), 
        (0.091, (191, 221, 229)), 
        (0.182, (156, 201, 226)), 
        (0.273, (129, 180, 227)), 
        (0.364, (115, 154, 228)), 
        (0.455, (117, 127, 221)), 
        (0.545, (120, 100, 202)), 
        (0.636, (119, 74, 175)), 
        (0.727, (113, 50, 141)), 
        (0.818, (100, 31, 104)), 
        (0.909, (80, 20, 66)), 
        (1.0, (54, 14, 36))
    ]

    def _interpolate_color(val: float) -> str:
        """Interpolate RGB color from the dense colorscale based on value 0.0-1.0."""
        val = max(0.0, min(1.0, val))
        
        # Find the two bounding stops
        for i in range(len(dense_colorscale) - 1):
            stop1, rgb1 = dense_colorscale[i]
            stop2, rgb2 = dense_colorscale[i + 1]
            if stop1 <= val <= stop2:
                # Linear interpolation
                ratio = (val - stop1) / (stop2 - stop1) if stop2 > stop1 else 0
                r = int(rgb1[0] + (rgb2[0] - rgb1[0]) * ratio)
                g = int(rgb1[1] + (rgb2[1] - rgb1[1]) * ratio)
                b = int(rgb1[2] + (rgb2[2] - rgb1[2]) * ratio)
                return f"rgba({r}, {g}, {b}, 0.85)"
                
        # Fallback to extreme
        rgb_last = dense_colorscale[-1][1]
        return f"rgba({rgb_last[0]}, {rgb_last[1]}, {rgb_last[2]}, 0.85)"

    html_parts = []
    
    # CSS for the container and explicit tooltip styling just in case
    style = """<style>
    .highlight-token { position: relative; display: inline-block; cursor: help; }
    .highlight-token .tooltiptext { visibility: hidden; width: max-content; background-color: #333; color: #fff; text-align: center; border-radius: 4px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; transform: translateX(-50%); opacity: 0; transition: opacity 0.3s; font-family: sans-serif; font-size: 12px; }
    .highlight-token:hover .tooltiptext { visibility: visible; opacity: 1; }
    </style>"""
    html_parts.append(style)
    html_parts.append('<div style="font-family: monospace; line-height: 2.0; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background: #f9f9f9; color: #1a1a1a;">')
    
    for i, (token, score) in enumerate(zip(tokens, scores)):
        norm_score = float(score)
        color = _interpolate_color(norm_score)
        
        # Determine text color to ensure readability against dark backgrounds
        is_dark = (norm_score > 0.6)
        text_color = "white" if is_dark else "black"
            
        # Clean up GPT-2 / BPE tokenizer weirdness (Ġ represents a space)
        # Some tokenizers render the prefix space explicitly as an extended character
        display_token = token.replace("Ġ", " ").replace("Ċ", "<br>").replace("<|endoftext|>", " [EOS] ")
        # Also clean up the Ä string if it leaked through from raw bytes encoding
        display_token = display_token.replace("Ä ", " ").replace("Ä", " ")
        
        # Format HTML display
        display_token = display_token.replace("\n", "<br>")
        
        # Tooltip content
        tooltip = f"Factuality Score: {norm_score:.3f}"
        if token_accuracies is not None:
            acc = token_accuracies[i] * 100
            tooltip += f" | Probe Accuracy: {acc:.1f}%"
        
        span = f'<span class="highlight-token" style="background-color: {color}; color: {text_color}; padding: 2px 1px; border-radius: 2px; margin: 0 1px;">{display_token}<span class="tooltiptext">{tooltip}</span></span>'
        html_parts.append(span)
        
    html_parts.append('</div>')
    
    return "".join(html_parts)

def generate_highlight_map_from_results(
    results: 'Any',
    orchestrator: 'Any',
    text: str,
    method: str = "best_global_layer",
    save_path: Optional[str] = None
) -> str:
    """
    Extracts activations for a specific text, applies position-specific probes from ProbeResults, 
    and generates an HTML highlight map.
    
    Args:
        results: The trained ProbeResults object.
        orchestrator: The ProbeOrchestrator used to extract activations.
        text: The text string to evaluate.
        method: Searing method, either:
                - "best_global_layer": Uses the single best layer across the whole model and iterates through positions inside that layer.
                - "best_token_layer": For each token position, finds the absolutely most accurate layer and uses that probe.
    """
    import numpy as np
    from easyprobe.models.data_models import ComponentOption, PositionOption
    
    # Tokenize the input text
    # Handle potentially different backend properties (e.g. nnsight vs transformerlens)
    if hasattr(orchestrator.extractor.model, "tokenizer"):
        input_ids = orchestrator.extractor.model.tokenizer(text).input_ids
        tokens = orchestrator.extractor.model.tokenizer.convert_ids_to_tokens(input_ids)
    else:
        tokens = orchestrator.extractor.model.to_str_tokens(text)

    scores = []
    token_accs = []
    
    # Safely get resid component string or enum value
    def _is_resid(probe_comp):
        return (probe_comp.value if hasattr(probe_comp, 'value') else str(probe_comp)) == "resid"
    
    if method == "best_global_layer":
        # TODO: Option 1 (Cumulative Analysis) - Use the single globally best layer. 
        # Here we stay inside one single layer (e.g., Layer 2) and see how factuality evolves across the sentence physically.
        best_layer = results.best_layer
        
        acts_dict = orchestrator.extractor.extract_activations(
            prompts=[text],
            layers=[best_layer],
            components=[ComponentOption.RESID],
            position=PositionOption.ALL,
            batch_size=1
        )
        sample_acts = acts_dict[(best_layer, ComponentOption.RESID)]
        if sample_acts.ndim == 3:
            sample_acts = sample_acts[0] # [seq_len, hidden_dim]
            
        pos_probes = [p for p in results.trained_probes if p.layer == best_layer and _is_resid(p.component)]
        pos_probes = sorted(pos_probes, key=lambda p: p.position[0])
        
        max_len = min(len(tokens), len(pos_probes), len(sample_acts))
        for i in range(max_len):
            probe = pos_probes[i]
            token_act = sample_acts[i:i+1]
            score = probe.predict_probability(token_act)[0]
            if isinstance(score, np.ndarray) and score.ndim > 0:
                score = float(score[0])
            scores.append(float(score))
            token_accs.append(probe.accuracy)
            
    elif method == "best_token_layer":
        # TODO: Option 2 (Spatial Analysis) - Find the absolute best layer for EACH token position completely independently. 
        # Token 0 might pull from Layer 11, while Token 4 pulls from Layer 2. Shows the smartest prediction at each step.
        all_layers = results.layers
        acts_dict = orchestrator.extractor.extract_activations(
            prompts=[text],
            layers=all_layers,
            components=[ComponentOption.RESID],
            position=PositionOption.ALL,
            batch_size=1
        )
        
        resid_probes = [p for p in results.trained_probes if _is_resid(p.component)]
        max_pos = max([p.position[0] for p in resid_probes])
        
        max_len = min(len(tokens), max_pos + 1)
        
        for i in range(max_len):
            # Get all probes at this exact token index
            probes_at_pos = [p for p in resid_probes if p.position[0] == i]
            if not probes_at_pos:
                scores.append(0.5)
                token_accs.append(0.0)
                continue
                
            # Pick the absolute best layer for this specific token
            best_probe_for_pos = max(probes_at_pos, key=lambda p: p.accuracy)
            
            # Get activation for this specific layer at this position
            layer_acts = acts_dict[(best_probe_for_pos.layer, ComponentOption.RESID)]
            if layer_acts.ndim == 3:
                layer_acts = layer_acts[0]
            
            if i >= len(layer_acts):
                scores.append(0.5)
                token_accs.append(0.0)
                continue
                
            token_act = layer_acts[i:i+1]
            score = best_probe_for_pos.predict_probability(token_act)[0]
            if isinstance(score, np.ndarray) and score.ndim > 0:
                score = float(score[0])
            
            scores.append(float(score))
            token_accs.append(best_probe_for_pos.accuracy)
    else:
        raise ValueError(f"Unknown method {method}")

    html_out = generate_highlight_map(
        tokens=tokens[:len(scores)],
        scores=np.array(scores),
        threshold=0.5,
        token_accuracies=token_accs
    )
    
    if save_path:
        title_text = "Position-Specific Probe Highlight"
        if method == "best_global_layer":
            title_text += f" (Layer {best_layer})"
        elif method == "best_token_layer":
            title_text += " (Best Layer Per Token)"
            
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(f"<h3>{title_text}:</h3>{html_out}")
            
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"✓ Saved text highlight map to {save_path}")

    return html_out
