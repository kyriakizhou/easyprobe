"""Visualization and reporting utilities for easyprobe."""

from easyprobe.visualization.heatmap import (
    plot_heatmap_interactive,
    plot_layer_position_heatmap,
    plot_multi_feature_heatmap,
    plot_multi_model_heatmap,
)
from easyprobe.visualization.report import (
    generate_report,
    merge_timing_reports,
    generate_multi_model_report,
    get_model_comparison_string,
)

__all__ = [
    # Heatmap functions
    "plot_heatmap_interactive",
    "plot_layer_position_heatmap",
    "plot_multi_feature_heatmap",
    "plot_multi_model_heatmap",
    # Report functions
    "generate_report",
    "merge_timing_reports",
    "generate_multi_model_report",
    "get_model_comparison_string",
]
