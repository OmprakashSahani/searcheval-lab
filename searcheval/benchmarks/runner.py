"""Benchmark runner for SearchEval Lab."""

from __future__ import annotations

import time
from dataclasses import dataclass

from searcheval.datasets.loader import SearchDataset
from searcheval.datasets.schema import QueryResults
from searcheval.eval.evaluator import EvaluationReport, evaluate_run
from searcheval.search.base import SearchEngine


@dataclass(frozen=True)
class QueryLatency:
    """Latency measurement for one query."""

    query_id: str
    latency_ms: float


@dataclass(frozen=True)
class BenchmarkRun:
    """Result of a benchmark run."""

    engine_name: str
    k: int
    evaluation: EvaluationReport
    query_latencies: list[QueryLatency]


def run_benchmark(
    dataset: SearchDataset,
    search_engine: SearchEngine,
    engine_name: str,
    k: int,
) -> BenchmarkRun:
    """Run a search benchmark over all dataset queries."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    query_results_list: list[QueryResults] = []
    query_latencies: list[QueryLatency] = []

    for query in dataset.queries:
        start_time = time.perf_counter()

        results = search_engine.search(
            query=query.query,
            top_k=k,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        query_results_list.append(
            QueryResults(
                query_id=query.query_id,
                results=results,
            )
        )

        query_latencies.append(
            QueryLatency(
                query_id=query.query_id,
                latency_ms=latency_ms,
            )
        )

    evaluation = evaluate_run(
        query_results_list=query_results_list,
        qrels=dataset.qrels,
        k=k,
    )

    return BenchmarkRun(
        engine_name=engine_name,
        k=k,
        evaluation=evaluation,
        query_latencies=query_latencies,
    )


def latency_summary(query_latencies: list[QueryLatency]) -> dict[str, float]:
    """Compute simple latency summary statistics."""
    if not query_latencies:
        return {
            "latency_avg_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    latencies = [item.latency_ms for item in query_latencies]

    return {
        "latency_avg_ms": sum(latencies) / len(latencies),
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
    }


def benchmark_summary(run: BenchmarkRun) -> dict[str, float | str | int]:
    """Create a compact benchmark summary."""
    summary: dict[str, float | str | int] = {
        "engine": run.engine_name,
        "k": run.k,
    }

    summary.update(run.evaluation.aggregate_metrics)
    summary.update(latency_summary(run.query_latencies))

    return summary
