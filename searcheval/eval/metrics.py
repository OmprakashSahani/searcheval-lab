"""Ranking metrics for SearchEval Lab."""

from __future__ import annotations

import math


def validate_k(k: int) -> None:
    """Validate that k is a positive integer."""
    if k <= 0:
        raise ValueError("k must be greater than 0.")


def precision_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """Compute Precision@K.

    Precision@K measures how many of the top-K retrieved documents are relevant.

    A document is treated as relevant if its relevance score is greater than 0.
    """
    validate_k(k)

    top_k_doc_ids = retrieved_doc_ids[:k]

    if not top_k_doc_ids:
        return 0.0

    relevant_retrieved = sum(
        1 for doc_id in top_k_doc_ids if relevance_by_doc_id.get(doc_id, 0) > 0
    )

    return relevant_retrieved / k


def recall_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """Compute Recall@K.

    Recall@K measures how many relevant documents were retrieved in the top-K results.
    """
    validate_k(k)

    relevant_doc_ids = {
        doc_id for doc_id, relevance in relevance_by_doc_id.items() if relevance > 0
    }

    if not relevant_doc_ids:
        return 0.0

    top_k_doc_ids = set(retrieved_doc_ids[:k])
    relevant_retrieved = relevant_doc_ids.intersection(top_k_doc_ids)

    return len(relevant_retrieved) / len(relevant_doc_ids)


def reciprocal_rank_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """Compute Reciprocal Rank@K.

    Reciprocal Rank@K rewards the first relevant result appearing early.
    """
    validate_k(k)

    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if relevance_by_doc_id.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain@K.

    DCG rewards highly relevant documents appearing near the top of the ranking.
    """
    validate_k(k)

    dcg = 0.0

    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        relevance = relevance_by_doc_id.get(doc_id, 0)
        gain = (2**relevance) - 1
        discount = math.log2(rank + 1)
        dcg += gain / discount

    return dcg


def ndcg_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> float:
    """Compute Normalized Discounted Cumulative Gain@K.

    NDCG compares the actual ranking against the ideal ranking.
    """
    validate_k(k)

    actual_dcg = dcg_at_k(retrieved_doc_ids, relevance_by_doc_id, k)

    ideal_relevances = sorted(
        relevance_by_doc_id.values(),
        reverse=True,
    )

    ideal_dcg = 0.0

    for rank, relevance in enumerate(ideal_relevances[:k], start=1):
        gain = (2**relevance) - 1
        discount = math.log2(rank + 1)
        ideal_dcg += gain / discount

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def ranking_metrics_at_k(
    retrieved_doc_ids: list[str],
    relevance_by_doc_id: dict[str, int],
    k: int,
) -> dict[str, float]:
    """Compute all core ranking metrics at K."""
    return {
        f"precision_at_{k}": precision_at_k(retrieved_doc_ids, relevance_by_doc_id, k),
        f"recall_at_{k}": recall_at_k(retrieved_doc_ids, relevance_by_doc_id, k),
        f"mrr_at_{k}": reciprocal_rank_at_k(retrieved_doc_ids, relevance_by_doc_id, k),
        f"ndcg_at_{k}": ndcg_at_k(retrieved_doc_ids, relevance_by_doc_id, k),
    }
