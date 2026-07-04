"""Tests for benchmark comparison utilities."""

import json

import pytest

from searcheval.benchmarks.compare import (
    compare_run_dirs,
    load_json,
    load_run_summary_metrics,
    save_comparison_result,
)


def write_summary(run_dir, summary) -> None:
    """Write a benchmark summary.json file for tests."""
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_dir.name,
        "engine": "tfidf",
        "k": 10,
        "summary": summary,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_load_json_reads_file(tmp_path) -> None:
    path = tmp_path / "sample.json"
    path.write_text('{"name": "searcheval"}', encoding="utf-8")

    payload = load_json(path)

    assert payload == {"name": "searcheval"}


def test_load_json_raises_for_missing_file(tmp_path) -> None:
    missing_path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        load_json(missing_path)


def test_load_run_summary_metrics_returns_numeric_metrics_only(tmp_path) -> None:
    run_dir = tmp_path / "run_baseline"

    write_summary(
        run_dir=run_dir,
        summary={
            "engine": "tfidf",
            "k": 10,
            "precision_at_10": 0.18,
            "recall_at_10": 0.95,
            "mrr_at_10": 0.95,
            "ndcg_at_10": 0.91,
            "latency_avg_ms": 0.60,
            "latency_max_ms": 0.80,
        },
    )

    metrics = load_run_summary_metrics(run_dir)

    assert metrics == {
        "k": 10.0,
        "precision_at_10": 0.18,
        "recall_at_10": 0.95,
        "mrr_at_10": 0.95,
        "ndcg_at_10": 0.91,
        "latency_avg_ms": 0.60,
        "latency_max_ms": 0.80,
    }


def test_load_run_summary_metrics_rejects_invalid_summary_payload(tmp_path) -> None:
    run_dir = tmp_path / "run_invalid"
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": "run_invalid",
        "engine": "tfidf",
        "k": 10,
        "summary": "invalid",
    }

    (run_dir / "summary.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_run_summary_metrics(run_dir)


def test_compare_run_dirs_passes_when_metrics_are_within_thresholds(tmp_path) -> None:
    baseline_run_dir = tmp_path / "run_baseline"
    current_run_dir = tmp_path / "run_current"

    write_summary(
        run_dir=baseline_run_dir,
        summary={
            "recall_at_10": 0.95,
            "ndcg_at_10": 0.91,
            "mrr_at_10": 0.95,
            "latency_avg_ms": 1.0,
        },
    )

    write_summary(
        run_dir=current_run_dir,
        summary={
            "recall_at_10": 0.94,
            "ndcg_at_10": 0.90,
            "mrr_at_10": 0.94,
            "latency_avg_ms": 2.0,
        },
    )

    thresholds = {
        "recall_at_10": -0.03,
        "ndcg_at_10": -0.02,
        "mrr_at_10": -0.03,
        "latency_avg_ms": 5.0,
    }

    result = compare_run_dirs(
        baseline_run_dir=baseline_run_dir,
        current_run_dir=current_run_dir,
        thresholds=thresholds,
    )

    assert result.passed is True
    assert len(result.comparisons) == 4
    assert result.missing_metrics == []


def test_compare_run_dirs_fails_when_metric_regresses(tmp_path) -> None:
    baseline_run_dir = tmp_path / "run_baseline"
    current_run_dir = tmp_path / "run_current"

    write_summary(
        run_dir=baseline_run_dir,
        summary={
            "ndcg_at_10": 0.91,
        },
    )

    write_summary(
        run_dir=current_run_dir,
        summary={
            "ndcg_at_10": 0.85,
        },
    )

    thresholds = {
        "ndcg_at_10": -0.02,
    }

    result = compare_run_dirs(
        baseline_run_dir=baseline_run_dir,
        current_run_dir=current_run_dir,
        thresholds=thresholds,
    )

    assert result.passed is False
    assert result.comparisons[0].metric_name == "ndcg_at_10"
    assert round(result.comparisons[0].delta, 4) == -0.06
    assert result.comparisons[0].passed is False


def test_save_comparison_result_creates_json_file(tmp_path) -> None:
    baseline_run_dir = tmp_path / "run_baseline"
    current_run_dir = tmp_path / "run_current"

    write_summary(
        run_dir=baseline_run_dir,
        summary={
            "ndcg_at_10": 0.91,
        },
    )

    write_summary(
        run_dir=current_run_dir,
        summary={
            "ndcg_at_10": 0.90,
        },
    )

    result = compare_run_dirs(
        baseline_run_dir=baseline_run_dir,
        current_run_dir=current_run_dir,
        thresholds={"ndcg_at_10": -0.02},
    )

    output_path = tmp_path / "comparison.json"

    saved_path = save_comparison_result(
        result=result,
        output_path=output_path,
    )

    assert saved_path == output_path
    assert output_path.exists()

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["passed"] is True
    assert payload["missing_metrics"] == []
    assert payload["comparisons"][0]["metric_name"] == "ndcg_at_10"
