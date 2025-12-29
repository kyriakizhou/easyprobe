"""
HTML report generation for probe results.

This module provides functions for generating HTML training reports
with timing details, accuracy metrics, and probe weights.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from easyprobe.datamodels import ProbeTimingReport
    from easyprobe.probe_results import ProbeResults


def generate_report(
    timing_report: "ProbeTimingReport",
    output_path: str = "probe_report.html",
    show: bool = False,
) -> None:
    """
    Generate an HTML training report with timing and accuracy details.

    Args:
        timing_report: ProbeTimingReport containing all timing and result data
        output_path: Path for the output HTML file
        show: Whether to open the report in a browser
    """
    try:
        from jinja2 import Environment, PackageLoader, select_autoescape
    except ImportError:
        raise ImportError(
            "Jinja2 not installed. Install with:\n  pip install jinja2"
        )

    # Set up Jinja2 environment
    env = Environment(
        loader=PackageLoader("easyprobe", "templates"),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")

    # Collect all feature names
    all_features = []
    for model_report in timing_report.model_reports:
        for feature_report in model_report.feature_reports:
            if feature_report.feature_name not in all_features:
                all_features.append(feature_report.feature_name)

    # Find max accuracy for highlighting
    max_accuracy = max(
        feature_report.best_accuracy
        for model_report in timing_report.model_reports
        for feature_report in model_report.feature_reports
    )

    # Render template
    html_content = template.render(
        report=timing_report,
        all_features=all_features,
        max_accuracy=max_accuracy,
    )

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    if show:
        import webbrowser
        webbrowser.open(f"file://{output_path}")


def merge_timing_reports(
    results_dict: dict[str, "ProbeResults"],
) -> "ProbeTimingReport":
    """
    Merge timing reports from multiple ProbeResults into a single report.

    Args:
        results_dict: Dictionary mapping model names to ProbeResults

    Returns:
        Combined ProbeTimingReport with all models
    """
    from easyprobe.datamodels import ProbeTimingReport

    model_reports = []
    total_s = 0.0

    for model_name, results in results_dict.items():
        if results.timing_report is None:
            raise ValueError(f"No timing report available for model '{model_name}'")

        # Get the model report from this result
        for model_report in results.timing_report.model_reports:
            model_reports.append(model_report)

        total_s += results.timing_report.total_s

    return ProbeTimingReport(
        run_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_s=total_s,
        model_reports=model_reports,
    )


def generate_multi_model_report(
    results_dict: dict[str, "ProbeResults"],
    output_path: str = "multi_model_report.html",
    show: bool = False,
) -> None:
    """
    Generate a combined HTML report for multiple models.

    Args:
        results_dict: Dictionary mapping model names to ProbeResults
        output_path: Path for the output HTML file
        show: Whether to open the report in a browser
    """
    # Merge timing reports
    timing_report = merge_timing_reports(results_dict)

    # Use the common report generator
    generate_report(timing_report, output_path, show)
