"""Tests for ranking metrics."""

from math import isclose

import pytest

from searcheval.eval.metrics import (
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    ranking_metrics_at_k,
    recall_at_k,
    reciprocal_rank_at_k,
)


def test_precision_at_k() -> None:
    retrieved_doc_ids = ["doc_001", "doc_002", "doc_003"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_003": 1,
    }

    assert precision_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3) == 2 / 3


def test_recall_at_k() -> None:
    retrieved_doc_ids = ["doc_001", "doc_002", "doc_003"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_003": 1,
        "doc_004": 2,
    }

    assert recall_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3) == 2 / 3


def test_reciprocal_rank_at_k() -> None:
    retrieved_doc_ids = ["doc_009", "doc_003", "doc_001"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_003": 1,
    }

    assert reciprocal_rank_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3) == 1 / 2


def test_reciprocal_rank_at_k_returns_zero_when_no_relevant_result() -> None:
    retrieved_doc_ids = ["doc_009", "doc_010", "doc_011"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_003": 1,
    }

    assert reciprocal_rank_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3) == 0.0


def test_dcg_at_k() -> None:
    retrieved_doc_ids = ["doc_001", "doc_002", "doc_003"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_002": 2,
        "doc_003": 1,
    }

    score = dcg_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3)

    assert score > 0


def test_ndcg_at_k_is_one_for_ideal_ranking() -> None:
    retrieved_doc_ids = ["doc_001", "doc_002", "doc_003"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_002": 2,
        "doc_003": 1,
    }

    assert isclose(ndcg_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3), 1.0)


def test_ndcg_at_k_is_less_than_one_for_non_ideal_ranking() -> None:
    retrieved_doc_ids = ["doc_003", "doc_002", "doc_001"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_002": 2,
        "doc_003": 1,
    }

    assert ndcg_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3) < 1.0


def test_ranking_metrics_at_k_returns_all_metrics() -> None:
    retrieved_doc_ids = ["doc_001", "doc_002", "doc_003"]
    relevance_by_doc_id = {
        "doc_001": 3,
        "doc_003": 1,
    }

    metrics = ranking_metrics_at_k(retrieved_doc_ids, relevance_by_doc_id, k=3)

    assert set(metrics.keys()) == {
        "precision_at_3",
        "recall_at_3",
        "mrr_at_3",
        "ndcg_at_3",
    }


def test_metrics_raise_error_for_invalid_k() -> None:
    retrieved_doc_ids = ["doc_001"]
    relevance_by_doc_id = {"doc_001": 3}

    with pytest.raises(ValueError):
        precision_at_k(retrieved_doc_ids, relevance_by_doc_id, k=0)
