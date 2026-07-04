"""Benchmark run comparison utilities for SearchEval Lab."""

from __future__ import annotations

import json
from pathlib import Path

from searcheval.regression.detector import (
    RegressionCheckResult,
    check_regressions,
    summarize_regression_result,
)


def load_json(path: Path) -> dict[str, object]:
    """Load a JSON file from disk."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_run_summary_metrics(run_dir: Path) -> dict[str, float]:
    """Load comparable metrics from a saved benchmark run directory.

    The CLI stores aggregate quality metrics and latency summaries inside:

    run_dir / summary.json

    Expected structure:

    {
      "run_id": "...",
      "engine": "tfidf",
      "k": 10,
      "summary": {
        "precision_at_10": 0.18,
        "recall_at_10": 0.95,
        "mrr_at_10": 0.95,
        "ndcg_at_10": 0.91,
        "latency_avg_ms": 0.60,
        "latency_min_ms": 0.45,
        "latency_max_ms": 0.80
      }
    }
    """
    summary_path = run_dir / "summary.json"
    payload = load_json(summary_path)

    raw_summary = payload.get("summary")

    if not isinstance(raw_summary, dict):
        raise ValueError(f"Invalid summary payload in {summary_path}")

    metrics: dict[str, float] = {}

    for key, value in raw_summary.items():
        if isinstance(value, int | float):
            metrics[key] = float(value)

    return metrics


def compare_run_dirs(
    baseline_run_dir: Path,
    current_run_dir: Path,
    thresholds: dict[str, float] | None = None,
) -> RegressionCheckResult:
    """Compare two saved benchmark run directories."""
    baseline_metrics = load_run_summary_metrics(baseline_run_dir)
    current_metrics = load_run_summary_metrics(current_run_dir)

    return check_regressions(
        baseline_metrics=baseline_metrics,
        current_metrics=current_metrics,
        thresholds=thresholds,
    )


def save_comparison_result(
    result: RegressionCheckResult,
    output_path: Path,
) -> Path:
    """Save a regression comparison result as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(
            summarize_regression_result(result),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return output_path
