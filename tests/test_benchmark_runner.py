"""Tests for the benchmark runner."""

import pytest

from searcheval.benchmarks.runner import (
    benchmark_summary,
    latency_summary,
    run_benchmark,
)
from searcheval.datasets.loader import SearchDataset
from searcheval.datasets.schema import Document, Query, Qrel, SearchResult
from searcheval.search.base import SearchEngine


class FakeSearchEngine(SearchEngine):
    """Simple deterministic search engine for benchmark tests."""

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return fixed search results for testing."""
        return [
            SearchResult(doc_id="doc_001", score=1.0, rank=1),
            SearchResult(doc_id="doc_002", score=0.8, rank=2),
        ][:top_k]


def sample_dataset() -> SearchDataset:
    """Create a small benchmark dataset for tests."""
    return SearchDataset(
        documents=[
            Document(
                doc_id="doc_001",
                title="Transformer Architecture",
                text="Transformers use self-attention mechanisms.",
            ),
            Document(
                doc_id="doc_002",
                title="Distributed Training",
                text="Distributed training introduces communication overhead.",
            ),
        ],
        queries=[
            Query(
                query_id="q_001",
                query="how do transformers use attention",
            ),
            Query(
                query_id="q_002",
                query="distributed training communication overhead",
            ),
        ],
        qrels=[
            Qrel(query_id="q_001", doc_id="doc_001", relevance=3),
            Qrel(query_id="q_002", doc_id="doc_002", relevance=3),
        ],
    )


def test_run_benchmark_returns_benchmark_run() -> None:
    dataset = sample_dataset()
    search_engine = FakeSearchEngine(dataset.documents)

    run = run_benchmark(
        dataset=dataset,
        search_engine=search_engine,
        engine_name="fake",
        k=2,
    )

    assert run.engine_name == "fake"
    assert run.k == 2
    assert len(run.evaluation.per_query) == 2
    assert len(run.query_latencies) == 2
    assert "precision_at_2" in run.evaluation.aggregate_metrics
    assert "recall_at_2" in run.evaluation.aggregate_metrics
    assert "mrr_at_2" in run.evaluation.aggregate_metrics
    assert "ndcg_at_2" in run.evaluation.aggregate_metrics


def test_run_benchmark_rejects_invalid_k() -> None:
    dataset = sample_dataset()
    search_engine = FakeSearchEngine(dataset.documents)

    with pytest.raises(ValueError):
        run_benchmark(
            dataset=dataset,
            search_engine=search_engine,
            engine_name="fake",
            k=0,
        )


def test_latency_summary_returns_zero_for_empty_input() -> None:
    summary = latency_summary([])

    assert summary == {
        "latency_avg_ms": 0.0,
        "latency_min_ms": 0.0,
        "latency_max_ms": 0.0,
    }


def test_benchmark_summary_contains_metrics_and_latency() -> None:
    dataset = sample_dataset()
    search_engine = FakeSearchEngine(dataset.documents)

    run = run_benchmark(
        dataset=dataset,
        search_engine=search_engine,
        engine_name="fake",
        k=2,
    )

    summary = benchmark_summary(run)

    assert summary["engine"] == "fake"
    assert summary["k"] == 2
    assert "precision_at_2" in summary
    assert "recall_at_2" in summary
    assert "mrr_at_2" in summary
    assert "ndcg_at_2" in summary
    assert "latency_avg_ms" in summary
    assert "latency_min_ms" in summary
    assert "latency_max_ms" in summary
