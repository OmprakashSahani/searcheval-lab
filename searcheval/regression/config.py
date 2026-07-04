"""Regression threshold configuration loading for SearchEval Lab."""

from __future__ import annotations

import json
from pathlib import Path


def load_regression_thresholds(config_path: Path) -> dict[str, float]:
    """Load regression thresholds from a JSON config file.

    Expected format:

    {
      "description": "Default regression thresholds...",
      "thresholds": {
        "recall_at_10": -0.03,
        "ndcg_at_10": -0.02,
        "latency_avg_ms": 5.0
      }
    }
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Regression config file not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid regression config format in {config_path}")

    raw_thresholds = payload.get("thresholds")

    if not isinstance(raw_thresholds, dict):
        raise ValueError(f"Missing or invalid 'thresholds' section in {config_path}")

    thresholds: dict[str, float] = {}

    for metric_name, value in raw_thresholds.items():
        if not isinstance(metric_name, str):
            raise ValueError(f"Invalid metric name in {config_path}: {metric_name}")

        if not isinstance(value, int | float):
            raise ValueError(
                f"Invalid threshold value for metric '{metric_name}' in {config_path}: {value}"
            )

        thresholds[metric_name] = float(value)

    if not thresholds:
        raise ValueError(f"Regression config contains no thresholds: {config_path}")

    return thresholds
