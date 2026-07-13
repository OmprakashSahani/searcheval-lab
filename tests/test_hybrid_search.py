"""Tests for the hybrid search engine."""

from __future__ import annotations

import pytest

from searcheval.datasets.schema import Document, SearchResult
from searcheval.search.hybrid import (
    HybridSearchEngine,
    HybridWeights,
    normalize_scores,
    validate_weights,
)


def sample_documents() -> list[Document]:
    """Create sample documents for hybrid search tests."""
    return [
        Document(
            doc_id="doc_001",
            title="Transformer Attention",
            text="Transformers use attention mechanisms to model token relationships.",
        ),
        Document(
            doc_id="doc_002",
            title="Distributed Training",
            text=(
                "Distributed training introduces communication overhead and synchronization costs."
            ),
        ),
        Document(
            doc_id="doc_003",
            title="Search Evaluation Metrics",
            text=(
                "Search quality can be evaluated using Precision at K, Recall at K, MRR, and NDCG."
            ),
        ),
    ]


def test_validate_weights_accepts_positive_weights() -> None:
    validate_weights(HybridWeights(bm25=0.7, tfidf=0.3))


def test_validate_weights_rejects_negative_bm25_weight() -> None:
    with pytest.raises(ValueError, match="BM25 weight must be non-negative"):
        validate_weights(HybridWeights(bm25=-0.1, tfidf=0.5))


def test_validate_weights_rejects_negative_tfidf_weight() -> None:
    with pytest.raises(ValueError, match="TF-IDF weight must be non-negative"):
        validate_weights(HybridWeights(bm25=0.5, tfidf=-0.1))


def test_validate_weights_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError, match="At least one hybrid search weight"):
        validate_weights(HybridWeights(bm25=0.0, tfidf=0.0))


def test_normalize_scores_returns_empty_dict_for_empty_results() -> None:
    assert normalize_scores([]) == {}


def test_normalize_scores_returns_one_for_equal_scores() -> None:
    results = [
        SearchResult(doc_id="doc_001", score=2.0, rank=1),
        SearchResult(doc_id="doc_002", score=2.0, rank=2),
    ]

    assert normalize_scores(results) == {
        "doc_001": 1.0,
        "doc_002": 1.0,
    }


def test_normalize_scores_scales_scores_between_zero_and_one() -> None:
    results = [
        SearchResult(doc_id="doc_001", score=2.0, rank=1),
        SearchResult(doc_id="doc_002", score=4.0, rank=2),
        SearchResult(doc_id="doc_003", score=6.0, rank=3),
    ]

    normalized = normalize_scores(results)

    assert normalized["doc_001"] == 0.0
    assert normalized["doc_002"] == 0.5
    assert normalized["doc_003"] == 1.0


def test_hybrid_search_requires_documents() -> None:
    with pytest.raises(ValueError, match="requires at least one document"):
        HybridSearchEngine([])


def test_hybrid_search_returns_ranked_results() -> None:
    engine = HybridSearchEngine(sample_documents())
    results = engine.search(
        query="distributed training communication overhead",
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].doc_id == "doc_002"
    assert [result.rank for result in results] == [1, 2]
    assert results[0].score >= results[1].score


def test_hybrid_search_supports_custom_weights() -> None:
    engine = HybridSearchEngine(
        documents=sample_documents(),
        weights=HybridWeights(bm25=0.8, tfidf=0.2),
    )
    results = engine.search(
        query="search evaluation metrics",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].doc_id == "doc_003"


def test_hybrid_search_rejects_invalid_top_k() -> None:
    engine = HybridSearchEngine(sample_documents())
    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        engine.search(query="distributed training", top_k=0)
