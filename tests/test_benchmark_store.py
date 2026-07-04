"""Tests for benchmark artifact storage."""

import json

from searcheval.benchmarks.runner import BenchmarkRun, QueryLatency
from searcheval.benchmarks.store import (
    build_latencies_payload,
    build_metrics_payload,
    build_per_query_metrics_payload,
    build_summary_payload,
    create_run_id,
    save_benchmark_run,
)
from searcheval.eval.evaluator import EvaluationReport, QueryEvaluation


def sample_benchmark_run() -> BenchmarkRun:
    """Create a sample benchmark run for store tests."""
    evaluation = EvaluationReport(
        k=10,
        aggregate_metrics={
            "precision_at_10": 0.5,
            "recall_at_10": 1.0,
            "mrr_at_10": 1.0,
            "ndcg_at_10": 0.95,
        },
        per_query=[
            QueryEvaluation(
                query_id="q_001",
                metrics={
                    "precision_at_10": 0.5,
                    "recall_at_10": 1.0,
                    "mrr_at_10": 1.0,
                    "ndcg_at_10": 0.95,
                },
                retrieved_doc_ids=["doc_001", "doc_002"],
                relevant_doc_ids=["doc_001"],
            )
        ],
    )

    return BenchmarkRun(
        engine_name="tfidf",
        k=10,
        evaluation=evaluation,
        query_latencies=[
            QueryLatency(query_id="q_001", latency_ms=1.25),
        ],
    )


def test_create_run_id_contains_engine_name() -> None:
    run_id = create_run_id("tfidf")

    assert run_id.startswith("run_")
    assert run_id.endswith("_tfidf")


def test_build_summary_payload() -> None:
    run = sample_benchmark_run()

    payload = build_summary_payload(run)

    assert payload["run_id"] is None
    assert payload["engine"] == "tfidf"
    assert payload["k"] == 10
    assert payload["summary"]["precision_at_10"] == 0.5
    assert payload["summary"]["latency_avg_ms"] == 1.25


def test_build_metrics_payload() -> None:
    run = sample_benchmark_run()

    payload = build_metrics_payload(run)

    assert payload == {
        "engine": "tfidf",
        "k": 10,
        "aggregate_metrics": {
            "precision_at_10": 0.5,
            "recall_at_10": 1.0,
            "mrr_at_10": 1.0,
            "ndcg_at_10": 0.95,
        },
    }


def test_build_per_query_metrics_payload() -> None:
    run = sample_benchmark_run()

    payload = build_per_query_metrics_payload(run)

    assert payload == [
        {
            "query_id": "q_001",
            "metrics": {
                "precision_at_10": 0.5,
                "recall_at_10": 1.0,
                "mrr_at_10": 1.0,
                "ndcg_at_10": 0.95,
            },
            "retrieved_doc_ids": ["doc_001", "doc_002"],
            "relevant_doc_ids": ["doc_001"],
        }
    ]


def test_build_latencies_payload() -> None:
    run = sample_benchmark_run()

    payload = build_latencies_payload(run)

    assert payload == [
        {
            "query_id": "q_001",
            "latency_ms": 1.25,
        }
    ]


def test_save_benchmark_run_creates_artifacts(tmp_path) -> None:
    run = sample_benchmark_run()

    run_dir = save_benchmark_run(
        run=run,
        runs_dir=tmp_path,
        run_id="run_test_tfidf",
    )

    assert run_dir.exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "per_query_metrics.json").exists()
    assert (run_dir / "latencies.json").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["run_id"] == "run_test_tfidf"
    assert summary["engine"] == "tfidf"
    assert summary["summary"]["precision_at_10"] == 0.5
