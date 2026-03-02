"""
HTML report generation for probe results.

This module provides functions for generating HTML training reports
with timing details, accuracy metrics, and probe weights.
"""

from __future__ import annotations

from datetime import datetime


def generate_report(
    timing_report: "ProbeTimingReport",
    path: str = "probe_report.html",
    show: bool = False,
) -> None:
    """
    Generate an HTML training report with timing and accuracy details.

    Args:
        timing_report: ProbeTimingReport containing all timing and result data
        path: Path for the output HTML file
        show: Whether to open the report in a browser
    """
    from jinja2 import Environment, FileSystemLoader, select_autoescape
    import os

    # Set up Jinja2 environment
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    env = Environment(
        loader=FileSystemLoader(template_dir),
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
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_content)

    if show:
        import webbrowser
        webbrowser.open(f"file://{path}")


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
    from easyprobe.models.data_models import ProbeTimingReport

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


def get_model_comparison_string(results_dict: dict[str, "ProbeResults"]) -> str:
    """
    Generate a formatted string comparing results across multiple models.
    """
    summary_logs = ["-" * 60 + "\nTRAINING STAGE COMPARISON RESULTS\n" + "-" * 60]
    for stage_name, results in results_dict.items():
        n_layers = len(set(r.layer for r in results.trained_probes))
        stage_summary = f"{stage_name}:\n  Layers: {n_layers}\n  Best layer: {results.best_layer}\n  Best accuracy: {results.best_result.accuracy:.1%}"
        if results.best_result.selectivity is not None:
            stage_summary += f"\n  Selectivity: {results.best_result.selectivity:.1%}"
        summary_logs.append(stage_summary)
    return "\n\n".join(summary_logs)


def generate_multi_model_report(
    results_dict: dict[str, "ProbeResults"],
    path: str = "multi_model_report.html",
    show: bool = False,
) -> None:
    """
    Generate a combined HTML report for multiple models.

    Args:
        results_dict: Dictionary mapping model names to ProbeResults
        path: Path for the output HTML file
        show: Whether to open the report in a browser
    """
    # Merge timing reports
    timing_report = merge_timing_reports(results_dict)

    # Use the common report generator
    generate_report(timing_report, path, show)
