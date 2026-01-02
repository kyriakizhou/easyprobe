"""
ProbeResults: Collection of probe results with analysis and visualization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from easyprobe.datamodels import ProbeResult, PositionOption, ProbeTimingReport
from easyprobe.visualization import heatmap, report


@dataclass
class ProbeResults:
    """
    Collection of probe results with analysis and visualization methods.

    This is the main object returned by ProbeAnalyzer.probe(). It contains
    all individual probe results and provides methods for:
    - Visualization (heatmaps, line plots)
    - Data export (DataFrame, numpy)
    - Summary statistics

    Example:
        results = analyzer.probe(prompts, labels)

        # Quick overview
        results.show()

        # Interactive heatmap visualization
        results.plot_heatmap_interactive()

        # Export for custom analysis
        df = results.to_dataframe()
    """

    results: list[ProbeResult]
    prompts: list[str]
    labels: np.ndarray
    model_name: str
    timing_report: Optional[ProbeTimingReport] = None
    _df: Optional[pd.DataFrame] = field(default=None, repr=False)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def best_layer(self) -> int:
        """Layer with highest accuracy (for resid component)."""
        resid_results = [
            r for r in self.results if r.component == "resid"
        ]
        if not resid_results:
            resid_results = self.results
        return max(resid_results, key=lambda r: r.accuracy).layer

    @property
    def best_result(self) -> ProbeResult:
        """ProbeResult with highest accuracy (for resid component)."""
        resid_results = [r for r in self.results if r.component == "resid"]
        if not resid_results:
            resid_results = self.results
        return max(resid_results, key=lambda r: r.accuracy)

    @property
    def best_accuracy(self) -> float:
        """Highest accuracy achieved across all probes."""
        return max(r.accuracy for r in self.results)

    @property
    def mean_selectivity(self) -> Optional[float]:
        """Average selectivity across all probes."""
        selectivities = [
            r.selectivity for r in self.results if r.selectivity is not None
        ]
        return float(np.mean(selectivities)) if selectivities else None

    @property
    def layers(self) -> list[int]:
        """List of unique layers probed."""
        return sorted(set(r.layer for r in self.results))

    @property
    def components(self) -> list[str]:
        """List of unique components probed."""
        return list(set(r.component for r in self.results))

    # -------------------------------------------------------------------------
    # Data Export
    # -------------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Returns a DataFrame with columns:
        - layer, component, head, position
        - accuracy, accuracy_std
        - random_baseline, selectivity
        - n_samples, is_significant
        """
        if self._df is not None:
            return self._df

        records = []
        for r in self.results:
            # Convert enums to strings for JSON serialization (Plotly compatibility)
            if hasattr(r.component, 'value'):
                component_str = r.component.value
            else:
                component_str = str(r.component)

            # Format position for display
            if isinstance(r.position, PositionOption):
                position_val = r.position.value
            elif isinstance(r.position, list):
                if len(r.position) == 1:
                    pos_idx = r.position[0]
                    if pos_idx == -1:
                        position_val = "last"
                    else:
                        position_val = pos_idx
                else:
                    position_val = str(r.position)
            else:
                position_val = str(r.position)

            records.append(
                {
                    "layer": r.layer,
                    "component": component_str,
                    "position": position_val,
                    "position_raw": r.position,
                    "accuracy": r.accuracy,
                    "accuracy_std": r.accuracy_std,
                    "random_baseline": r.random_baseline,
                    "selectivity": r.selectivity,
                    "n_samples": r.n_samples,
                    "is_significant": r.is_significant,
                }
            )

        self._df = pd.DataFrame(records)
        return self._df

    def to_numpy(self) -> np.ndarray:
        """
        Return accuracy values as numpy array, shaped by layer.

        Only includes residual stream probes, sorted by layer.
        """
        df = self.to_dataframe()
        resid_df = df[df["component"] == "resid"]
        return resid_df.sort_values("layer")["accuracy"].values

    def filter(
        self,
        component: Optional[str] = None,
        layer: Optional[int] = None,
    ) -> list[ProbeResult]:
        """Filter results by component or layer."""
        filtered = self.results

        if component is not None:
            comp_val = component.value if hasattr(component, 'value') else component
            filtered = [
                r for r in filtered
                if (r.component.value if hasattr(r.component, 'value') else r.component) == comp_val
            ]
        if layer is not None:
            filtered = [r for r in filtered if r.layer == layer]

        return filtered

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_heatmap_interactive(
        self,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Interactive heatmap (Plotly) with hover tooltips.

        Structure:
        - X-axis: Position × Component (grouped by position, components within)
        - Y-axis: Layers (0 at bottom, higher layers going up)

        Hover shows: layer, position, component, accuracy, selectivity.
        Saves to a self-contained HTML file if output_path is provided.
        """
        return heatmap.plot_heatmap_interactive(
            df=self.to_dataframe(),
            model_name=self.model_name,
            title=title,
            output_path=output_path,
            show=show,
        )

    def plot_layer_position_heatmap(
        self,
        component: str = "resid",
        title: Optional[str] = None,
        output_path: Optional[str] = None,
        show: bool = True,
    ):
        """
        Heatmap of Layer x Token Position for a specific component.

        Useful when position="all" or a list of positions was used.
        """
        return heatmap.plot_layer_position_heatmap(
            df=self.to_dataframe(),
            model_name=self.model_name,
            component=component,
            title=title,
            output_path=output_path,
            show=show,
        )

    # -------------------------------------------------------------------------
    # Summary and Reporting
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a text summary of results."""
        lines = [
            "Probe Results Summary",
            "=" * 50,
            f"Model: {self.model_name}",
            f"Samples: {len(self.prompts)}",
            f"Labels: {np.sum(self.labels == 1)} positive, {np.sum(self.labels == 0)} negative",
            "",
            f"Best layer: {self.best_layer}",
            f"Best accuracy: {self.best_accuracy:.1%}",
        ]

        if self.mean_selectivity is not None:
            lines.append(f"Mean selectivity: {self.mean_selectivity:.1%}")

        # Find significant layers
        significant = [r for r in self.results if r.is_significant]
        if significant:
            lines.append("")
            lines.append("Layers with significant signal (selectivity > 10%):")
            for r in sorted(
                significant, key=lambda x: x.selectivity or 0, reverse=True
            )[:5]:
                lines.append(
                    f"  Layer {r.layer} ({r.component}): "
                    f"{r.accuracy:.1%} accuracy, {r.selectivity:.1%} selectivity"
                )

        return "\n".join(lines)

    def show(self) -> None:
        """Display summary and main visualization."""
        print(self.summary())
        print()
        self.plot_heatmap_interactive(show=True)

    def generate_report(
        self,
        output_path: str = "probe_report.html",
        show: bool = False,
    ) -> None:
        """
        Generate an HTML training report with timing and accuracy details.

        Args:
            output_path: Path for the output HTML file
            show: Whether to open the report in a browser
        """
        if self.timing_report is None:
            raise ValueError(
                "No timing report available. The timing report is generated "
                "automatically when using ProbeOrchestrator.probe()."
            )

        report.generate_report(self.timing_report, output_path, show)

    def __repr__(self) -> str:
        return (
            f"ProbeResults(model='{self.model_name}', "
            f"n_results={len(self.results)}, "
            f"best_accuracy={self.best_accuracy:.1%})"
        )

    def __len__(self) -> int:
        return len(self.results)

    def __getitem__(self, idx: int) -> ProbeResult:
        return self.results[idx]


@dataclass
class MultiFeatureProbeResults:
    """
    Collection of probe results for multiple features.

    This is returned when probing multiple features simultaneously.
    Each feature has its own ProbeResults object.

    Example:
        # Probe multiple features
        data = MultiFeatureSharedPromptsData(
            prompts=prompts,
            features={"sentiment": labels1, "topic": labels2}
        )
        results = analyzer.probe(data)

        # Access individual feature results
        results["sentiment"].plot_heatmap_interactive()

        # Interactive heatmap with all features
        results.plot_heatmap_interactive()

        # Get results for all features
        for feature_name, feature_results in results.items():
            print(f"{feature_name}: best layer = {feature_results.best_layer}")
    """

    feature_results: dict[str, ProbeResults]
    model_name: str
    timing_report: Optional[ProbeTimingReport] = None

    # -------------------------------------------------------------------------
    # Dictionary-like access
    # -------------------------------------------------------------------------

    def __getitem__(self, feature_name: str) -> ProbeResults:
        """Get results for a specific feature."""
        return self.feature_results[feature_name]

    def __contains__(self, feature_name: str) -> bool:
        """Check if a feature exists."""
        return feature_name in self.feature_results

    def keys(self):
        """Get feature names."""
        return self.feature_results.keys()

    def values(self):
        """Get all ProbeResults objects."""
        return self.feature_results.values()

    def items(self):
        """Get (feature_name, ProbeResults) pairs."""
        return self.feature_results.items()

    def __len__(self) -> int:
        """Return the number of features."""
        return len(self.feature_results)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """List of feature names."""
        return list(self.feature_results.keys())

    @property
    def num_features(self) -> int:
        """Number of features."""
        return len(self.feature_results)

    # -------------------------------------------------------------------------
    # Data Export
    # -------------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert all feature results to a single DataFrame.

        Adds a 'feature' column to distinguish between features.
        """
        dfs = []
        for feature_name, results in self.feature_results.items():
            df = results.to_dataframe()
            df.insert(0, "feature", feature_name)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------

    def plot_heatmap_interactive(
        self,
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
            title: Overall title for the figure
            output_path: Path for output HTML file
            show: Whether to display the plot

        Returns:
            Plotly Figure with subplots for each feature
        """
        feature_dataframes = {
            name: results.to_dataframe()
            for name, results in self.feature_results.items()
        }
        return heatmap.plot_multi_feature_heatmap(
            feature_dataframes=feature_dataframes,
            model_name=self.model_name,
            title=title,
            output_path=output_path,
            show=show,
        )

    # -------------------------------------------------------------------------
    # Summary and Reporting
    # -------------------------------------------------------------------------

    def summary(self) -> str:
        """Generate a text summary of all feature results."""
        lines = [
            "Multi-Feature Probe Results Summary",
            "=" * 60,
            f"Model: {self.model_name}",
            f"Number of features: {self.num_features}",
            "",
        ]

        for feature_name, results in self.feature_results.items():
            lines.append(f"Feature: {feature_name}")
            lines.append(f"  Best layer: {results.best_layer}")
            lines.append(f"  Best accuracy: {results.best_accuracy:.1%}")
            if results.mean_selectivity is not None:
                lines.append(f"  Mean selectivity: {results.mean_selectivity:.1%}")
            lines.append("")

        return "\n".join(lines)

    def show(self) -> None:
        """Display summary and main visualizations."""
        print(self.summary())
        print()
        self.plot_heatmap_interactive(show=True)

    def generate_report(
        self,
        output_path: str = "probe_report.html",
        show: bool = False,
    ) -> None:
        """
        Generate an HTML training report with timing and accuracy details.

        Args:
            output_path: Path for the output HTML file
            show: Whether to open the report in a browser
        """
        if self.timing_report is None:
            raise ValueError(
                "No timing report available. The timing report is generated "
                "automatically when using ProbeOrchestrator.probe()."
            )

        report.generate_report(self.timing_report, output_path, show)

    def __repr__(self) -> str:
        feature_list = ", ".join(self.feature_names[:3])
        if len(self.feature_names) > 3:
            feature_list += f", ... ({len(self.feature_names)} total)"
        return (
            f"MultiFeatureProbeResults(model='{self.model_name}', "
            f"features=[{feature_list}])"
        )
