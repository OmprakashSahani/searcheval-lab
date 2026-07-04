"""Tests for query failure analysis."""

import pytest

from searcheval.eval.evaluator import EvaluationReport, QueryEvaluation
from searcheval.eval.failure_analysis import (
    WeakQuery,
    find_queries_below_threshold,
    find_weak_queries,
    get_metric_score,
    summarize_weak_queries,
)


def sample_evaluation_report() -> EvaluationReport:
    """Create a sample evaluation report for failure analysis tests."""
    return EvaluationReport(
        k=10,
        aggregate_metrics={
            "precision_at_10": 0.4,
            "recall_at_10": 0.8,
            "mrr_at_10": 0.7,
            "ndcg_at_10": 0.6,
        },
        per_query=[
            QueryEvaluation(
                query_id="q_001",
                metrics={
                    "precision_at_10": 0.5,
                    "recall_at_10": 1.0,
                    "mrr_at_10": 1.0,
                    "ndcg_at_10": 0.9,
                },
                retrieved_doc_ids=["doc_001", "doc_002"],
                relevant_doc_ids=["doc_001"],
            ),
            QueryEvaluation(
                query_id="q_002",
                metrics={
                    "precision_at_10": 0.1,
                    "recall_at_10": 0.5,
                    "mrr_at_10": 0.25,
                    "ndcg_at_10": 0.2,
                },
                retrieved_doc_ids=["doc_003", "doc_004"],
                relevant_doc_ids=["doc_002"],
            ),
            QueryEvaluation(
                query_id="q_003",
                metrics={
                    "precision_at_10": 0.3,
                    "recall_at_10": 0.7,
                    "mrr_at_10": 0.5,
                    "ndcg_at_10": 0.4,
                },
                retrieved_doc_ids=["doc_005", "doc_006"],
                relevant_doc_ids=["doc_006"],
            ),
        ],
    )


def test_get_metric_score_returns_metric_value() -> None:
    report = sample_evaluation_report()
    query_evaluation = report.per_query[0]

    score = get_metric_score(
        query_evaluation=query_evaluation,
        metric_name="ndcg_at_10",
    )

    assert score == 0.9


def test_get_metric_score_raises_for_missing_metric() -> None:
    report = sample_evaluation_report()
    query_evaluation = report.per_query[0]

    with pytest.raises(ValueError):
        get_metric_score(
            query_evaluation=query_evaluation,
            metric_name="missing_metric",
        )


def test_find_weak_queries_returns_lowest_scoring_queries() -> None:
    report = sample_evaluation_report()

    weak_queries = find_weak_queries(
        report=report,
        metric_name="ndcg_at_10",
        limit=2,
    )

    assert len(weak_queries) == 2
    assert all(isinstance(query, WeakQuery) for query in weak_queries)
    assert weak_queries[0].query_id == "q_002"
    assert weak_queries[0].score == 0.2
    assert weak_queries[1].query_id == "q_003"
    assert weak_queries[1].score == 0.4


def test_find_weak_queries_rejects_invalid_limit() -> None:
    report = sample_evaluation_report()

    with pytest.raises(ValueError):
        find_weak_queries(
            report=report,
            metric_name="ndcg_at_10",
            limit=0,
        )


def test_find_queries_below_threshold() -> None:
    report = sample_evaluation_report()

    weak_queries = find_queries_below_threshold(
        report=report,
        metric_name="ndcg_at_10",
        threshold=0.5,
    )

    assert len(weak_queries) == 2
    assert [query.query_id for query in weak_queries] == ["q_002", "q_003"]


def test_summarize_weak_queries_returns_json_friendly_payload() -> None:
    report = sample_evaluation_report()

    weak_queries = find_weak_queries(
        report=report,
        metric_name="ndcg_at_10",
        limit=1,
    )

    summary = summarize_weak_queries(weak_queries)

    assert summary == [
        {
            "query_id": "q_002",
            "metric_name": "ndcg_at_10",
            "score": 0.2,
            "retrieved_doc_ids": ["doc_003", "doc_004"],
            "relevant_doc_ids": ["doc_002"],
        }
    ]
