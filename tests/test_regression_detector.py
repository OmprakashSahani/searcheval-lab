"""Tests for regression detection."""

from searcheval.regression.detector import (
    MetricComparison,
    RegressionCheckResult,
    check_regressions,
    compare_metric,
    metric_passed,
    summarize_regression_result,
)


def test_quality_metric_passes_when_drop_is_within_threshold() -> None:
    assert metric_passed(delta=-0.01, threshold=-0.02) is True


def test_quality_metric_fails_when_drop_exceeds_threshold() -> None:
    assert metric_passed(delta=-0.05, threshold=-0.02) is False


def test_latency_metric_passes_when_increase_is_within_threshold() -> None:
    assert metric_passed(delta=4.0, threshold=5.0) is True


def test_latency_metric_fails_when_increase_exceeds_threshold() -> None:
    assert metric_passed(delta=8.0, threshold=5.0) is False


def test_compare_metric_returns_metric_comparison() -> None:
    comparison = compare_metric(
        metric_name="ndcg_at_10",
        baseline_value=0.80,
        current_value=0.79,
        threshold=-0.02,
    )

    assert isinstance(comparison, MetricComparison)
    assert comparison.metric_name == "ndcg_at_10"
    assert comparison.baseline_value == 0.80
    assert comparison.current_value == 0.79
    assert round(comparison.delta, 4) == -0.01
    assert comparison.threshold == -0.02
    assert comparison.passed is True


def test_check_regressions_passes_when_all_metrics_are_within_thresholds() -> None:
    baseline_metrics = {
        "recall_at_10": 0.90,
        "ndcg_at_10": 0.80,
        "mrr_at_10": 0.85,
        "latency_avg_ms": 10.0,
    }

    current_metrics = {
        "recall_at_10": 0.89,
        "ndcg_at_10": 0.79,
        "mrr_at_10": 0.84,
        "latency_avg_ms": 13.0,
    }

    thresholds = {
        "recall_at_10": -0.03,
        "ndcg_at_10": -0.02,
        "mrr_at_10": -0.03,
        "latency_avg_ms": 5.0,
    }

    result = check_regressions(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        thresholds=thresholds,
    )

    assert isinstance(result, RegressionCheckResult)
    assert result.passed is True
    assert len(result.comparisons) == 4
    assert result.missing_metrics == []


def test_check_regressions_fails_when_quality_metric_drops_too_much() -> None:
    baseline_metrics = {
        "ndcg_at_10": 0.80,
    }

    current_metrics = {
        "ndcg_at_10": 0.74,
    }

    thresholds = {
        "ndcg_at_10": -0.02,
    }

    result = check_regressions(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        thresholds=thresholds,
    )

    assert result.passed is False
    assert result.comparisons[0].metric_name == "ndcg_at_10"
    assert round(result.comparisons[0].delta, 4) == -0.06
    assert result.comparisons[0].passed is False


def test_check_regressions_fails_when_latency_increases_too_much() -> None:
    baseline_metrics = {
        "latency_avg_ms": 10.0,
    }

    current_metrics = {
        "latency_avg_ms": 18.0,
    }

    thresholds = {
        "latency_avg_ms": 5.0,
    }

    result = check_regressions(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        thresholds=thresholds,
    )

    assert result.passed is False
    assert result.comparisons[0].metric_name == "latency_avg_ms"
    assert result.comparisons[0].delta == 8.0
    assert result.comparisons[0].passed is False


def test_check_regressions_tracks_missing_metrics() -> None:
    baseline_metrics = {
        "ndcg_at_10": 0.80,
    }

    current_metrics = {}

    thresholds = {
        "ndcg_at_10": -0.02,
    }

    result = check_regressions(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        thresholds=thresholds,
    )

    assert result.passed is False
    assert result.comparisons == []
    assert result.missing_metrics == ["ndcg_at_10"]


def test_summarize_regression_result_returns_json_friendly_payload() -> None:
    result = check_regressions(
        baseline_metrics={"ndcg_at_10": 0.80},
        current_metrics={"ndcg_at_10": 0.79},
        thresholds={"ndcg_at_10": -0.02},
    )

    summary = summarize_regression_result(result)

    assert summary["passed"] is True
    assert summary["missing_metrics"] == []
    assert summary["comparisons"] == [
        {
            "metric_name": "ndcg_at_10",
            "baseline_value": 0.80,
            "current_value": 0.79,
            "delta": -0.010000000000000009,
            "threshold": -0.02,
            "passed": True,
        }
    ]
