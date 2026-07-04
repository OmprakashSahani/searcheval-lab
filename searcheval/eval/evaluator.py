"""Evaluation engine for SearchEval Lab."""

from __future__ import annotations

from dataclasses import dataclass

from searcheval.datasets.schema import Qrel, QueryResults
from searcheval.eval.metrics import ranking_metrics_at_k


@dataclass(frozen=True)
class QueryEvaluation:
    """Evaluation metrics for one benchmark query."""

    query_id: str
    metrics: dict[str, float]
    retrieved_doc_ids: list[str]
    relevant_doc_ids: list[str]


@dataclass(frozen=True)
class EvaluationReport:
    """Evaluation report for a benchmark run."""

    k: int
    aggregate_metrics: dict[str, float]
    per_query: list[QueryEvaluation]


def build_relevance_map(qrels: list[Qrel]) -> dict[str, dict[str, int]]:
    """Build a query_id -> doc_id -> relevance lookup table."""
    relevance_by_query_id: dict[str, dict[str, int]] = {}

    for qrel in qrels:
        relevance_by_query_id.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance

    return relevance_by_query_id


def evaluate_query_results(
    query_results: QueryResults,
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> QueryEvaluation:
    """Evaluate retrieved results for a single query."""
    retrieved_doc_ids = [result.doc_id for result in query_results.results]
    relevant_doc_ids = [
        doc_id
        for doc_id, relevance in relevance_by_doc_id.items()
        if relevance > 0
    ]

    metrics = ranking_metrics_at_k(
        retrieved_doc_ids=retrieved_doc_ids,
        relevance_by_doc_id=relevance_by_doc_id,
        k=k,
    )

    return QueryEvaluation(
        query_id=query_results.query_id,
        metrics=metrics,
        retrieved_doc_ids=retrieved_doc_ids[:k],
        relevant_doc_ids=sorted(relevant_doc_ids),
    )


def aggregate_query_metrics(
    per_query: list[QueryEvaluation],
) -> dict[str, float]:
    """Average per-query metrics into aggregate benchmark metrics."""
    if not per_query:
        return {}

    metric_names = per_query[0].metrics.keys()
    aggregate_metrics: dict[str, float] = {}

    for metric_name in metric_names:
        aggregate_metrics[metric_name] = sum(
            query_eval.metrics[metric_name]
            for query_eval in per_query
        ) / len(per_query)

    return aggregate_metrics


def evaluate_run(
    query_results_list: list[QueryResults],
    qrels: list[Qrel],
    k: int,
) -> EvaluationReport:
    """Evaluate a full benchmark run."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    relevance_by_query_id = build_relevance_map(qrels)

    per_query: list[QueryEvaluation] = []

    for query_results in query_results_list:
        relevance_by_doc_id = relevance_by_query_id.get(query_results.query_id, {})

        query_evaluation = evaluate_query_results(
            query_results=query_results,
            relevance_by_doc_id=relevance_by_doc_id,
            k=k,
        )

        per_query.append(query_evaluation)

    aggregate_metrics = aggregate_query_metrics(per_query)

    return EvaluationReport(
        k=k,
        aggregate_metrics=aggregate_metrics,
        per_query=per_query,
    )
