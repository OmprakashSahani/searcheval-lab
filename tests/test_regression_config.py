"""Tests for regression threshold config loading."""

import json

import pytest

from searcheval.regression.config import load_regression_thresholds


def test_load_regression_thresholds_reads_valid_config(tmp_path) -> None:
    config_path = tmp_path / "regression.json"

    payload = {
        "description": "Test regression thresholds.",
        "thresholds": {
            "recall_at_10": -0.03,
            "ndcg_at_10": -0.02,
            "latency_avg_ms": 5.0,
        },
    }

    config_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    thresholds = load_regression_thresholds(config_path)

    assert thresholds == {
        "recall_at_10": -0.03,
        "ndcg_at_10": -0.02,
        "latency_avg_ms": 5.0,
    }


def test_load_regression_thresholds_raises_for_missing_file(tmp_path) -> None:
    config_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_regression_thresholds(config_path)


def test_load_regression_thresholds_rejects_non_object_payload(tmp_path) -> None:
    config_path = tmp_path / "regression.json"

    config_path.write_text(
        json.dumps(["invalid"]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_regression_thresholds(config_path)


def test_load_regression_thresholds_rejects_missing_thresholds_section(tmp_path) -> None:
    config_path = tmp_path / "regression.json"

    payload = {
        "description": "Missing thresholds section.",
    }

    config_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_regression_thresholds(config_path)


def test_load_regression_thresholds_rejects_non_numeric_threshold(tmp_path) -> None:
    config_path = tmp_path / "regression.json"

    payload = {
        "thresholds": {
            "ndcg_at_10": "bad-value",
        },
    }

    config_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_regression_thresholds(config_path)


def test_load_regression_thresholds_rejects_empty_thresholds(tmp_path) -> None:
    config_path = tmp_path / "regression.json"

    payload = {
        "thresholds": {},
    }

    config_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_regression_thresholds(config_path)
