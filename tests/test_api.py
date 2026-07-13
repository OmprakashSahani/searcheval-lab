"""Tests for the SearchEval Lab FastAPI backend."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from searcheval.api import api_app

client = TestClient(api_app)


def write_summary(run_dir: Path, summary: dict[str, float]) -> None:
    """Write a saved benchmark summary.json file for API tests."""
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


def sample_summary() -> dict[str, float]:
    """Return a sample benchmark summary."""
    return {
        "precision_at_10": 0.19,
        "recall_at_10": 1.0,
        "mrr_at_10": 0.95,
        "ndcg_at_10": 0.92,
        "latency_avg_ms": 1.2,
        "latency_max_ms": 2.5,
    }


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

    assert payload["engines"] == ["bm25", "hybrid", "tfidf"]


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


def test_api_run_hybrid_benchmark_without_saving_artifacts() -> None:
    response = client.post(
        "/benchmarks/run",
        json={
            "dataset_path": "data/search_eval_small",
            "engine": "hybrid",
            "k": 10,
            "save_artifacts": False,
        },
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["engine"] == "hybrid"
    assert payload["k"] == 10
    assert payload["run_dir"] is None
    assert payload["report_path"] is None
    assert "precision_at_10" in payload["summary"]
    assert "recall_at_10" in payload["summary"]
    assert "mrr_at_10" in payload["summary"]
    assert "ndcg_at_10" in payload["summary"]


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


def test_api_list_saved_benchmark_runs(tmp_path: Path) -> None:
    first_run = tmp_path / "run_001_tfidf"
    second_run = tmp_path / "run_002_bm25"

    write_summary(first_run, sample_summary())
    write_summary(second_run, sample_summary())

    response = client.get(
        "/benchmarks/runs",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 200

    payload = response.json()

    assert len(payload) == 2
    assert payload[0]["run_id"] == "run_001_tfidf"
    assert payload[1]["run_id"] == "run_002_bm25"
    assert payload[0]["summary"]["precision_at_10"] == 0.19


def test_api_list_saved_benchmark_runs_returns_empty_for_missing_dir(
    tmp_path: Path,
) -> None:
    response = client.get(
        "/benchmarks/runs",
        params={"runs_dir": str(tmp_path / "missing_runs")},
    )

    assert response.status_code == 200
    assert response.json() == []


def test_api_get_saved_benchmark_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 200

    payload = response.json()

    assert payload["run_id"] == "run_001_tfidf"
    assert payload["engine"] == "tfidf"
    assert payload["k"] == 10
    assert payload["run_dir"] == str(run_dir)
    assert payload["summary"]["ndcg_at_10"] == 0.92


def test_api_get_saved_benchmark_run_rejects_missing_run(tmp_path: Path) -> None:
    response = client.get(
        "/benchmarks/runs/missing_run",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 404
    assert "Benchmark run not found" in response.json()["detail"]


def test_api_get_saved_benchmark_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    report_text = "# Benchmark Report\n\nThis is a saved benchmark report.\n"
    report_path = run_dir / "report.md"
    report_path.write_text(report_text, encoding="utf-8")

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}/report",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    assert response.text == report_text


def test_api_get_saved_benchmark_report_rejects_missing_report(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}/report",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 404
    assert "Benchmark report not found" in response.json()["detail"]


def test_api_get_saved_benchmark_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    metrics_payload = {
        "run_id": run_dir.name,
        "engine": "tfidf",
        "k": 10,
        "metrics": sample_summary(),
    }

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}/metrics",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    assert response.json()["metrics"]["ndcg_at_10"] == 0.92


def test_api_get_saved_benchmark_per_query_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    per_query_payload = {
        "run_id": run_dir.name,
        "engine": "tfidf",
        "k": 10,
        "queries": [
            {
                "query_id": "q_001",
                "precision_at_10": 0.2,
                "recall_at_10": 1.0,
                "mrr_at_10": 1.0,
                "ndcg_at_10": 0.95,
            }
        ],
    }

    (run_dir / "per_query_metrics.json").write_text(
        json.dumps(per_query_payload, indent=2),
        encoding="utf-8",
    )

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}/per-query",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    assert response.json()["queries"][0]["query_id"] == "q_001"


def test_api_get_saved_benchmark_latencies(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    latency_payload = {
        "run_id": run_dir.name,
        "engine": "tfidf",
        "k": 10,
        "latencies": [
            {
                "query_id": "q_001",
                "latency_ms": 1.25,
            }
        ],
    }

    (run_dir / "latencies.json").write_text(
        json.dumps(latency_payload, indent=2),
        encoding="utf-8",
    )

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}/latencies",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 200
    assert response.json()["latencies"][0]["latency_ms"] == 1.25


def test_api_get_saved_benchmark_artifact_rejects_missing_artifact(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_001_tfidf"

    write_summary(run_dir, sample_summary())

    response = client.get(
        f"/benchmarks/runs/{run_dir.name}/metrics",
        params={"runs_dir": str(tmp_path)},
    )

    assert response.status_code == 404
    assert "Benchmark artifact not found" in response.json()["detail"]


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
        summary=sample_summary(),
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
        summary=sample_summary(),
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
