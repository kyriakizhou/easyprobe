"""
Helper functions for probe analysis.
"""

from typing import Union

from easyprobe.models.data_models import ComponentOption, PositionOption


def parse_position_spec(
    position: Union[PositionOption, int, list[int]], seq_len: int
) -> list[list[int]]:
    """
    Parse position specification into task specs.

    Args:
        position: Position specification
        seq_len: Sequence length from activations

    Returns:
        List of index lists for creating probe tasks.
        - For LAST: [[-1]]
        - For MEAN: [[0, 1, 2, ..., seq_len-1]]
        - For ALL: [[0], [1], [2], ..., [seq_len-1]]
        - For [1, 3, 5]: [[1], [3], [5]]
        - For int: [[int]]
    """
    if position == PositionOption.LAST:
        return [[-1]]
    elif position == PositionOption.MEAN:
        return [list(range(seq_len))]
    elif position == PositionOption.ALL:
        return [[i] for i in range(seq_len)]
    elif isinstance(position, (list, range)):
        return [[int(i)] for i in position]
    else:
        # Single integer
        return [[position]]


def normalize_component_spec(components: list) -> list[ComponentOption]:
    """
    Normalize component specification to list of ComponentOption enums.

    Args:
        components: List of components (strings or ComponentOption enums)

    Returns:
        List of ComponentOption enums

    Raises:
        ValueError: If unknown component specification
    """
    normalized_components: list[ComponentOption] = []
    for c in components:
        if isinstance(c, ComponentOption):
            normalized_components.append(c)
        else:
            try:
                normalized_components.append(ComponentOption(c))
            except ValueError as exc:
                raise ValueError(
                    f"Unknown component specification '{c}'. "
                    f"Expected one of {[opt.value for opt in ComponentOption]}."
                ) from exc
    return normalized_components


def cleanup_activation_checkpoints(paths: list[str]) -> list[str]:
    """
    Clean up temporary activation checkpoint directories.

    Args:
        paths: List of directory paths to remove.

    Returns:
        List of paths that were successfully removed.
    """
    import os
    import shutil

    cleaned_dirs = []
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
            cleaned_dirs.append(path)
    return cleaned_dirs
