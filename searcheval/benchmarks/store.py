"""Benchmark artifact storage for SearchEval Lab."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from searcheval.benchmarks.runner import BenchmarkRun, benchmark_summary


def create_run_id(engine_name: str) -> str:
    """Create a timestamped benchmark run ID."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    safe_engine_name = engine_name.lower().replace(" ", "_")
    return f"run_{timestamp}_{safe_engine_name}"


def write_json(path: Path, data: object) -> None:
    """Write JSON data to disk with readable formatting."""
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_summary_payload(run: BenchmarkRun) -> dict[str, object]:
    """Build the benchmark summary payload."""
    return {
        "run_id": None,
        "engine": run.engine_name,
        "k": run.k,
        "summary": benchmark_summary(run),
    }


def build_metrics_payload(run: BenchmarkRun) -> dict[str, object]:
    """Build aggregate metrics payload."""
    return {
        "engine": run.engine_name,
        "k": run.k,
        "aggregate_metrics": run.evaluation.aggregate_metrics,
    }


def build_per_query_metrics_payload(run: BenchmarkRun) -> list[dict[str, object]]:
    """Build per-query metrics payload."""
    return [
        {
            "query_id": query_evaluation.query_id,
            "metrics": query_evaluation.metrics,
            "retrieved_doc_ids": query_evaluation.retrieved_doc_ids,
            "relevant_doc_ids": query_evaluation.relevant_doc_ids,
        }
        for query_evaluation in run.evaluation.per_query
    ]


def build_latencies_payload(run: BenchmarkRun) -> list[dict[str, object]]:
    """Build per-query latency payload."""
    return [
        {
            "query_id": latency.query_id,
            "latency_ms": latency.latency_ms,
        }
        for latency in run.query_latencies
    ]


def save_benchmark_run(
    run: BenchmarkRun,
    runs_dir: Path = Path("runs"),
    run_id: str | None = None,
) -> Path:
    """Save benchmark run artifacts to disk.

    Output structure:

    runs/
      run_YYYYMMDD_HHMMSS_engine/
        summary.json
        metrics.json
        per_query_metrics.json
        latencies.json
    """
    if run_id is None:
        run_id = create_run_id(run.engine_name)

    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_payload = build_summary_payload(run)
    summary_payload["run_id"] = run_id

    write_json(run_dir / "summary.json", summary_payload)
    write_json(run_dir / "metrics.json", build_metrics_payload(run))
    write_json(run_dir / "per_query_metrics.json", build_per_query_metrics_payload(run))
    write_json(run_dir / "latencies.json", build_latencies_payload(run))

    return run_dir
