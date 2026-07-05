"""Tests for the SearchEval Lab FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from searcheval.api import api_app

client = TestClient(api_app)


def write_summary(run_dir: Path, summary: dict[str, float]) -> None:
    """Write a saved benchmark summary.json file for API compare tests."""
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


def test_api_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert "version" in payload


def test_api_engines_endpoint() -> None:
    response = client.get("/engines")

    assert response.status_code == 200

    payload = response.json()

    assert payload["engines"] == ["bm25", "tfidf"]


def test_api_validate_dataset_endpoint() -> None:
    response = client.post(
        "/datasets/validate",
        json={"dataset_path": "data/search_eval_small"},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["is_valid"] is True
    assert payload["errors"] == []
    assert payload["warnings"] == []


def test_api_validate_dataset_endpoint_rejects_missing_dataset() -> None:
    response = client.post(
        "/datasets/validate",
        json={"dataset_path": "data/does_not_exist"},
    )

    assert response.status_code == 404


def test_api_run_benchmark_without_saving_artifacts() -> None:
    response = client.post(
        "/benchmarks/run",
        json={
            "dataset_path": "data/search_eval_small",
            "engine": "tfidf",
            "k": 10,
            "save_artifacts": False,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["engine"] == "tfidf"
    assert payload["k"] == 10
    assert payload["run_dir"] is None
    assert payload["report_path"] is None
    assert "precision_at_10" in payload["summary"]
    assert "recall_at_10" in payload["summary"]
    assert "mrr_at_10" in payload["summary"]
    assert "ndcg_at_10" in payload["summary"]


def test_api_run_benchmark_with_artifacts(tmp_path: Path) -> None:
    response = client.post(
        "/benchmarks/run",
        json={
            "dataset_path": "data/search_eval_small",
            "engine": "bm25",
            "k": 10,
            "save_artifacts": True,
            "runs_dir": str(tmp_path),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["engine"] == "bm25"
    assert payload["run_dir"] is not None
    assert payload["report_path"] is not None

    run_dirs = list(tmp_path.iterdir())

    assert len(run_dirs) == 1
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "report.md").exists()


def test_api_run_benchmark_rejects_unknown_engine() -> None:
    response = client.post(
        "/benchmarks/run",
        json={
            "dataset_path": "data/search_eval_small",
            "engine": "unknown",
            "k": 10,
            "save_artifacts": False,
        },
    )

    assert response.status_code == 400
    assert "Unsupported search engine" in response.json()["detail"]


def test_api_compare_benchmark_runs(tmp_path: Path) -> None:
    baseline_run = tmp_path / "run_baseline_tfidf"
    current_run = tmp_path / "run_current_bm25"
    output_path = tmp_path / "comparison.json"

    write_summary(
        run_dir=baseline_run,
        summary={
            "precision_at_10": 0.18,
            "recall_at_10": 0.95,
            "mrr_at_10": 0.95,
            "ndcg_at_10": 0.91,
            "latency_avg_ms": 1.0,
            "latency_max_ms": 2.0,
        },
    )

    write_summary(
        run_dir=current_run,
        summary={
            "precision_at_10": 0.19,
            "recall_at_10": 1.0,
            "mrr_at_10": 0.95,
            "ndcg_at_10": 0.92,
            "latency_avg_ms": 1.2,
            "latency_max_ms": 2.5,
        },
    )

    response = client.post(
        "/benchmarks/compare",
        json={
            "baseline_run": str(baseline_run),
            "current_run": str(current_run),
            "output_path": str(output_path),
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["passed"] is True
    assert payload["comparison"]["passed"] is True
    assert payload["output_path"] == str(output_path)
    assert output_path.exists()


def test_api_compare_rejects_missing_baseline_run(tmp_path: Path) -> None:
    current_run = tmp_path / "run_current_bm25"

    write_summary(
        run_dir=current_run,
        summary={
            "precision_at_10": 0.19,
            "recall_at_10": 1.0,
            "mrr_at_10": 0.95,
            "ndcg_at_10": 0.92,
        },
    )

    response = client.post(
        "/benchmarks/compare",
        json={
            "baseline_run": str(tmp_path / "missing_baseline"),
            "current_run": str(current_run),
        },
    )

    assert response.status_code == 404
    assert "Baseline run directory not found" in response.json()["detail"]
