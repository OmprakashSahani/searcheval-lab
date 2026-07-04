"""Regression detection for SearchEval Lab."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricComparison:
    """Comparison result for one benchmark metric."""

    metric_name: str
    baseline_value: float
    current_value: float
    delta: float
    threshold: float
    passed: bool


@dataclass(frozen=True)
class RegressionCheckResult:
    """Result of comparing two benchmark metric sets."""

    passed: bool
    comparisons: list[MetricComparison]
    missing_metrics: list[str]


DEFAULT_REGRESSION_THRESHOLDS: dict[str, float] = {
    "precision_at_10": -0.05,
    "recall_at_10": -0.03,
    "mrr_at_10": -0.03,
    "ndcg_at_10": -0.02,
    "latency_avg_ms": 5.0,
    "latency_max_ms": 10.0,
}


def metric_passed(delta: float, threshold: float) -> bool:
    """Return whether a metric delta passes its regression threshold.

    Negative thresholds are used for quality metrics where drops are bad.

    Example:
    - threshold = -0.02
    - delta = -0.01
    - pass, because the metric dropped by less than the allowed threshold

    Positive thresholds are used for latency metrics where increases are bad.

    Example:
    - threshold = 10.0
    - delta = 4.0
    - pass, because latency increased by less than the allowed threshold
    """
    if threshold < 0:
        return delta >= threshold

    return delta <= threshold


def compare_metric(
    metric_name: str,
    baseline_value: float,
    current_value: float,
    threshold: float,
) -> MetricComparison:
    """Compare one metric between baseline and current runs."""
    delta = current_value - baseline_value

    return MetricComparison(
        metric_name=metric_name,
        baseline_value=baseline_value,
        current_value=current_value,
        delta=delta,
        threshold=threshold,
        passed=metric_passed(delta=delta, threshold=threshold),
    )


def check_regressions(
    baseline_metrics: dict[str, float],
    current_metrics: dict[str, float],
    thresholds: dict[str, float] | None = None,
) -> RegressionCheckResult:
    """Compare baseline and current metrics against regression thresholds."""
    if thresholds is None:
        thresholds = DEFAULT_REGRESSION_THRESHOLDS

    comparisons: list[MetricComparison] = []
    missing_metrics: list[str] = []

    for metric_name, threshold in thresholds.items():
        if metric_name not in baseline_metrics or metric_name not in current_metrics:
            missing_metrics.append(metric_name)
            continue

        comparisons.append(
            compare_metric(
                metric_name=metric_name,
                baseline_value=baseline_metrics[metric_name],
                current_value=current_metrics[metric_name],
                threshold=threshold,
            )
        )

    passed = all(comparison.passed for comparison in comparisons) and not missing_metrics

    return RegressionCheckResult(
        passed=passed,
        comparisons=comparisons,
        missing_metrics=missing_metrics,
    )


def summarize_regression_result(result: RegressionCheckResult) -> dict[str, object]:
    """Create a JSON-friendly summary of a regression check."""
    return {
        "passed": result.passed,
        "comparisons": [
            {
                "metric_name": comparison.metric_name,
                "baseline_value": comparison.baseline_value,
                "current_value": comparison.current_value,
                "delta": comparison.delta,
                "threshold": comparison.threshold,
                "passed": comparison.passed,
            }
            for comparison in result.comparisons
        ],
        "missing_metrics": result.missing_metrics,
    }
