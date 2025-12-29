"""
Input validation utilities.

Provides helpful error messages and warnings for common issues.
"""


def validate_layer_spec(layers, n_layers: int) -> list[int]:
    """
    Parse and validate layer specification.

    Args:
        layers: "all", list of ints, or range
        n_layers: Total number of layers in model

    Returns:
        List of layer indices

    Raises:
        ValueError: If specification is invalid
    """
    if layers == "all":
        return list(range(n_layers))
    elif isinstance(layers, range):
        layer_list = list(layers)
    elif isinstance(layers, list):
        layer_list = layers
    else:
        raise ValueError(
            f"Invalid layer specification: {layers}. "
            f"Expected 'all', a list of ints, or a range."
        )

    # Validate all layers are in range
    for layer in layer_list:
        if layer < 0 or layer >= n_layers:
            raise ValueError(
                f"Layer {layer} is out of range. "
                f"Model has {n_layers} layers (0 to {n_layers - 1})."
            )

    return layer_list



