"""Query failure analysis for SearchEval Lab."""

from __future__ import annotations

from dataclasses import dataclass

from searcheval.eval.evaluator import EvaluationReport, QueryEvaluation


@dataclass(frozen=True)
class WeakQuery:
    """A query with weak retrieval quality."""

    query_id: str
    score: float
    metric_name: str
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]


def get_metric_score(
    query_evaluation: QueryEvaluation,
    metric_name: str,
) -> float:
    """Return a metric score for a query evaluation."""
    if metric_name not in query_evaluation.metrics:
        raise ValueError(
            f"Metric '{metric_name}' not found for query '{query_evaluation.query_id}'."
        )

    return query_evaluation.metrics[metric_name]


def find_weak_queries(
    report: EvaluationReport,
    metric_name: str,
    limit: int = 5,
) -> list[WeakQuery]:
    """Find queries with the lowest score for a selected metric."""
    if limit <= 0:
        raise ValueError("limit must be greater than 0.")

    scored_queries = []

    for query_evaluation in report.per_query:
        score = get_metric_score(
            query_evaluation=query_evaluation,
            metric_name=metric_name,
        )

        scored_queries.append(
            WeakQuery(
                query_id=query_evaluation.query_id,
                score=score,
                metric_name=metric_name,
                retrieved_doc_ids=query_evaluation.retrieved_doc_ids,
                relevant_doc_ids=query_evaluation.relevant_doc_ids,
            )
        )

    return sorted(
        scored_queries,
        key=lambda weak_query: (weak_query.score, weak_query.query_id),
    )[:limit]


def find_queries_below_threshold(
    report: EvaluationReport,
    metric_name: str,
    threshold: float,
) -> list[WeakQuery]:
    """Find queries whose metric score is below a threshold."""
    weak_queries = find_weak_queries(
        report=report,
        metric_name=metric_name,
        limit=len(report.per_query),
    )

    return [
        weak_query
        for weak_query in weak_queries
        if weak_query.score < threshold
    ]


def summarize_weak_queries(weak_queries: list[WeakQuery]) -> list[dict[str, object]]:
    """Create a JSON-friendly weak query summary."""
    return [
        {
            "query_id": weak_query.query_id,
            "metric_name": weak_query.metric_name,
            "score": weak_query.score,
            "retrieved_doc_ids": weak_query.retrieved_doc_ids,
            "relevant_doc_ids": weak_query.relevant_doc_ids,
        }
        for weak_query in weak_queries
    ]
