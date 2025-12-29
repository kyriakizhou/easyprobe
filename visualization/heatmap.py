"""
Heatmap generation for probe results visualization.

This module provides functions for creating interactive Plotly heatmaps
to visualize probe accuracy across layers, components, and positions.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from easyprobe.probe_results import ProbeResults


def _sort_positions(positions):
    """Sort positions: integers numerically first, then strings like 'last'."""
    return sorted(positions, key=lambda x: (not isinstance(x, int), x if isinstance(x, int) else str(x)))


def plot_heatmap_interactive(
    df: pd.DataFrame,
    model_name: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Create an interactive heatmap (Plotly) with hover tooltips.

    Structure:
    - X-axis: Position × Component (grouped by position, components within)
    - Y-axis: Layers (0 at bottom, higher layers going up)

    Hover shows: layer, position, component, accuracy, selectivity.
    Saves to a self-contained HTML file if output_path is provided.

    Args:
        df: DataFrame with columns: layer, component, position, accuracy, selectivity
        model_name: Name of the model for the title
        title: Optional custom title
        output_path: Optional path to save HTML file
        show: Whether to display the plot

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly not installed. Install with `pip install plotly` "
            "to use plot_heatmap_interactive()."
        ) from exc

    if df.empty:
        raise ValueError("No results to plot.")

    layers = sorted(df["layer"].unique())
    positions = _sort_positions(df["position"].unique())
    components = sorted(df["component"].unique())

    # Build X-axis labels: "pos-component" format, grouped by position
    x_labels = []
    for pos in positions:
        for comp in components:
            x_labels.append(f"{pos}-{comp}")

    # Build matrix: rows = layers, cols = position×component
    accuracy_matrix = []
    customdata = []
    for layer in layers:
        acc_row = []
        custom_row = []
        for pos in positions:
            for comp in components:
                row = df[
                    (df["layer"] == layer) &
                    (df["position"] == pos) &
                    (df["component"] == comp)
                ]
                if not row.empty:
                    acc_row.append(float(row["accuracy"].iloc[0]))
                    sel_val = row["selectivity"].iloc[0]
                    custom_row.append([pos, comp, layer, float(sel_val) if sel_val is not None else None])
                else:
                    acc_row.append(None)
                    custom_row.append([pos, comp, layer, None])
        accuracy_matrix.append(acc_row)
        customdata.append(custom_row)

    fig = go.Figure(
        data=go.Heatmap(
            z=accuracy_matrix,
            x=x_labels,
            y=layers,
            colorscale=[[0, "#1f24b8"], [1, "#f7be02"]],
            zmin=0.5,
            zmax=1.0,
            colorbar=dict(title="Accuracy"),
            customdata=customdata,
            hovertemplate=(
                "Layer: %{customdata[2]}<br>"
                "Position: %{customdata[0]}<br>"
                "Component: %{customdata[1]}<br>"
                "Accuracy: %{z:.1%}<br>"
                "Selectivity: %{customdata[3]:.1%}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or f"Probe Accuracy Heatmap - {model_name}",
        xaxis_title="Position - Component",
        yaxis_title="Layer",
        width=max(600, len(x_labels) * 80),
        hoverlabel=dict(
            bgcolor="white",
            font_color="black",
            bordercolor="white",
        ),
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()
    return fig


def plot_position_heatmap(
    df: pd.DataFrame,
    model_name: str,
    component: str = "resid",
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Create a heatmap of Layer x Token Position for a specific component.

    Useful when position="all" or a list of positions was used.

    Args:
        df: DataFrame with columns: layer, component, position, accuracy
        model_name: Name of the model for the title
        component: Which component to filter for (default: "resid")
        title: Optional custom title
        output_path: Optional path to save HTML file
        show: Whether to display the plot

    Returns:
        Plotly Figure object
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "Plotly not installed. Install with `pip install plotly`."
        ) from exc

    df = df[df["component"] == component]

    # Filter out "mean" if it exists, as it doesn't fit in a per-index heatmap well
    df = df[df["position"] != "mean"]

    unique_pos = _sort_positions(df["position"].unique())

    if len(unique_pos) <= 1:
        print("Warning: plot_position_heatmap called but only one token position found.")

    layers = sorted(df["layer"].unique())

    # Build matrix
    z_matrix = []
    for layer in layers:
        row = []
        for pos in unique_pos:
            val = df[(df["layer"] == layer) & (df["position"] == pos)]
            if not val.empty:
                row.append(float(val["accuracy"].iloc[0]))
            else:
                row.append(None)
        z_matrix.append(row)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_matrix,
            x=[str(p) for p in unique_pos],
            y=layers,
            colorscale=[[0, "#1f24b8"], [1, "#f2b305"]],
            zmin=0.5,
            zmax=1.0,
            colorbar=dict(title="Accuracy"),
            hovertemplate=(
                "Layer: %{y}<br>"
                "Position: %{x}<br>"
                "Accuracy: %{z:.1%}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title or f"Layer x Position Accuracy - {component} ({model_name})",
        xaxis_title="Token Position",
        yaxis_title="Layer",
        hoverlabel=dict(
            bgcolor="white",
            font_color="black",
            bordercolor="white",
        ),
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()
    return fig


def plot_multi_feature_heatmap(
    feature_dataframes: dict[str, pd.DataFrame],
    model_name: str,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Generate a single interactive HTML with all features as subplots.

    Each feature gets its own heatmap with:
    - X-axis: Position × Component (grouped by position, components within)
    - Y-axis: Layers (0 at bottom, higher layers going up)

    Args:
        feature_dataframes: Dict mapping feature names to their DataFrames
        model_name: Name of the model for the title
        title: Overall title for the figure
        output_path: Path for output HTML file
        show: Whether to display the plot

    Returns:
        Plotly Figure with subplots for each feature
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly not installed. Install with `pip install plotly` "
            "to use plot_heatmap_interactive()."
        ) from exc

    num_features = len(feature_dataframes)
    feature_names = list(feature_dataframes.keys())

    # Create subplots - one column per feature (side by side)
    fig = make_subplots(
        rows=1,
        cols=num_features,
        subplot_titles=feature_names,
        horizontal_spacing=0.12,
    )

    for idx, (feature_name, df) in enumerate(feature_dataframes.items(), start=1):
        if df.empty:
            continue

        layers = sorted(df["layer"].unique())
        positions = _sort_positions(df["position"].unique())
        components = sorted(df["component"].unique())

        # Build X-axis labels: "pos-component" format
        x_labels = []
        for pos in positions:
            for comp in components:
                x_labels.append(f"{pos}-{comp}")

        # Build matrix: rows = layers, cols = position×component
        accuracy_matrix = []
        customdata = []
        for layer in layers:
            acc_row = []
            custom_row = []
            for pos in positions:
                for comp in components:
                    row = df[
                        (df["layer"] == layer) &
                        (df["position"] == pos) &
                        (df["component"] == comp)
                    ]
                    if not row.empty:
                        acc_row.append(float(row["accuracy"].iloc[0]))
                        sel_val = row["selectivity"].iloc[0]
                        custom_row.append([pos, comp, layer, float(sel_val) if sel_val is not None else None])
                    else:
                        acc_row.append(None)
                        custom_row.append([pos, comp, layer, None])
            accuracy_matrix.append(acc_row)
            customdata.append(custom_row)

        # Show colorbar only on the last (rightmost) subplot
        is_last = (idx == num_features)
        heatmap = go.Heatmap(
            z=accuracy_matrix,
            x=x_labels,
            y=layers,
            colorscale=[[0, "#1f24b8"], [1, "#f2b305"]],
            zmin=0.5,
            zmax=1.0,
            colorbar=dict(title="Accuracy") if is_last else None,
            showscale=is_last,
            customdata=customdata,
            hovertemplate=(
                f"<b>{feature_name}</b><br>"
                "Layer: %{customdata[2]}<br>"
                "Accuracy: %{z:.1%}<br>"
                "Selectivity: %{customdata[3]:.1%}<extra></extra>"
            ),
        )

        fig.add_trace(heatmap, row=1, col=idx)

        # Update axes for this subplot
        fig.update_xaxes(title_text="Position - Component", row=1, col=idx)
        fig.update_yaxes(title_text="Layer", row=1, col=idx)

    # Calculate width based on number of features (each chart ~400px wide)
    width = max(600, 400 * num_features + 100)  # +100 for colorbar

    fig.update_layout(
        title=title or f"Multi-Feature Probe Results - {model_name}",
        width=width,
        height=500,
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()

    return fig


def plot_multi_model_heatmap(
    results_dict: dict[str, "ProbeResults"],
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    show: bool = True,
):
    """
    Create a combined heatmap showing multiple models side by side.

    Args:
        results_dict: Dictionary mapping model names to ProbeResults
        title: Optional title for the plot
        output_path: Optional path to save the HTML file
        show: Whether to display the plot

    Returns:
        Plotly figure object
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "Plotly not installed. Install with `pip install plotly`."
        ) from exc

    model_names = list(results_dict.keys())
    n_models = len(model_names)

    # Create subplots - one column per model
    fig = make_subplots(
        rows=1,
        cols=n_models,
        subplot_titles=model_names,
        horizontal_spacing=0.1,
    )

    for idx, (model_name, results) in enumerate(results_dict.items(), start=1):
        df = results.to_dataframe()
        layers = sorted(df["layer"].unique())

        # Get unique position-component combinations
        df["pos_comp"] = df.apply(
            lambda row: f"{row['position']}-{row['component']}", axis=1
        )
        x_labels = sorted(df["pos_comp"].unique())

        # Build accuracy matrix
        accuracy_matrix = []
        for layer in layers:
            row = []
            for x_label in x_labels:
                val = df[(df["layer"] == layer) & (df["pos_comp"] == x_label)]
                if not val.empty:
                    row.append(float(val["accuracy"].iloc[0]))
                else:
                    row.append(None)
            accuracy_matrix.append(row)

        fig.add_trace(
            go.Heatmap(
                z=accuracy_matrix,
                x=x_labels,
                y=layers,
                colorscale=[[0, "#1f24b8"], [1, "#f7be02"]],
                zmin=0.5,
                zmax=1.0,
                colorbar=dict(
                    title="Accuracy",
                    x=1.0 if idx == n_models else None,
                ),
                showscale=(idx == n_models),  # Only show colorbar for last
                hovertemplate=(
                    f"Model: {model_name}<br>"
                    "Layer: %{y}<br>"
                    "Accuracy: %{z:.1%}<extra></extra>"
                ),
            ),
            row=1,
            col=idx,
        )

        # Update axes for this subplot
        fig.update_xaxes(title_text="Position-Component", row=1, col=idx)
        if idx == 1:
            fig.update_yaxes(title_text="Layer", row=1, col=idx)

    fig.update_layout(
        title=title or "Model Comparison - Probe Accuracy Heatmaps",
        width=400 * n_models,
        height=600,
        hoverlabel=dict(
            bgcolor="white",
            font_color="black",
            bordercolor="white",
        ),
    )

    if output_path:
        fig.write_html(output_path, include_plotlyjs="cdn", full_html=True)
    if show:
        fig.show()
    return fig
