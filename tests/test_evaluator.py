"""Tests for the evaluation engine."""

import pytest

from searcheval.datasets.schema import Qrel, QueryResults, SearchResult
from searcheval.eval.evaluator import (
    aggregate_query_metrics,
    build_relevance_map,
    evaluate_query_results,
    evaluate_run,
)


def test_build_relevance_map() -> None:
    qrels = [
        Qrel(query_id="q_001", doc_id="doc_001", relevance=3),
        Qrel(query_id="q_001", doc_id="doc_002", relevance=1),
        Qrel(query_id="q_002", doc_id="doc_003", relevance=2),
    ]

    relevance_map = build_relevance_map(qrels)

    assert relevance_map == {
        "q_001": {
            "doc_001": 3,
            "doc_002": 1,
        },
        "q_002": {
            "doc_003": 2,
        },
    }


def test_evaluate_query_results() -> None:
    query_results = QueryResults(
        query_id="q_001",
        results=[
            SearchResult(doc_id="doc_001", score=0.95, rank=1),
            SearchResult(doc_id="doc_003", score=0.70, rank=2),
            SearchResult(doc_id="doc_002", score=0.60, rank=3),
        ],
    )

    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_002": 1,
    }

    evaluation = evaluate_query_results(
        query_results=query_results,
        relevance_by_doc_id=relevance_by_doc_id,
        k=3,
    )

    assert evaluation.query_id == "q_001"
    assert evaluation.retrieved_doc_ids == ["doc_001", "doc_003", "doc_002"]
    assert evaluation.relevant_doc_ids == ["doc_001", "doc_002"]
    assert evaluation.metrics["precision_at_3"] == 2 / 3
    assert evaluation.metrics["recall_at_3"] == 1.0
    assert evaluation.metrics["mrr_at_3"] == 1.0
    assert evaluation.metrics["ndcg_at_3"] > 0.0


def test_aggregate_query_metrics() -> None:
    query_results_list = [
        QueryResults(
            query_id="q_001",
            results=[
                SearchResult(doc_id="doc_001", score=0.90, rank=1),
                SearchResult(doc_id="doc_002", score=0.80, rank=2),
            ],
        ),
        QueryResults(
            query_id="q_002",
            results=[
                SearchResult(doc_id="doc_004", score=0.90, rank=1),
                SearchResult(doc_id="doc_003", score=0.80, rank=2),
            ],
        ),
    ]

    qrels = [
        Qrel(query_id="q_001", doc_id="doc_001", relevance=3),
        Qrel(query_id="q_002", doc_id="doc_003", relevance=3),
    ]

    report = evaluate_run(
        query_results_list=query_results_list,
        qrels=qrels,
        k=2,
    )

    aggregate_metrics = aggregate_query_metrics(report.per_query)

    assert aggregate_metrics["precision_at_2"] == 0.5
    assert aggregate_metrics["recall_at_2"] == 1.0
    assert aggregate_metrics["mrr_at_2"] == 0.75


def test_evaluate_run_returns_report() -> None:
    query_results_list = [
        QueryResults(
            query_id="q_001",
            results=[
                SearchResult(doc_id="doc_001", score=0.95, rank=1),
                SearchResult(doc_id="doc_002", score=0.80, rank=2),
            ],
        )
    ]

    qrels = [
        Qrel(query_id="q_001", doc_id="doc_001", relevance=3),
    ]

    report = evaluate_run(
        query_results_list=query_results_list,
        qrels=qrels,
        k=2,
    )

    assert report.k == 2
    assert len(report.per_query) == 1
    assert report.aggregate_metrics["precision_at_2"] == 0.5
    assert report.aggregate_metrics["recall_at_2"] == 1.0
    assert report.aggregate_metrics["mrr_at_2"] == 1.0


def test_evaluate_run_rejects_invalid_k() -> None:
    with pytest.raises(ValueError):
        evaluate_run(
            query_results_list=[],
            qrels=[],
            k=0,
        )
