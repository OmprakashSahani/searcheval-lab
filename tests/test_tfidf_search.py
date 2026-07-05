"""Tests for the TF-IDF search engine."""

import pytest

from searcheval.datasets.schema import Document
from searcheval.search.tfidf import TfidfSearchEngine


def sample_documents() -> list[Document]:
    """Create a small document collection for search tests."""
    return [
        Document(
            doc_id="doc_001",
            title="Transformer Architecture",
            text="Transformers use self-attention mechanisms for sequence modeling.",
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


def test_tfidf_search_returns_top_k_results() -> None:
    documents = sample_documents()
    engine = TfidfSearchEngine(documents)

    results = engine.search("distributed training communication overhead", top_k=2)

    assert len(results) == 2
    assert results[0].doc_id == "doc_002"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_tfidf_search_finds_search_metrics_document() -> None:
    documents = sample_documents()
    engine = TfidfSearchEngine(documents)

    results = engine.search("metrics for evaluating search quality", top_k=1)

    assert len(results) == 1
    assert results[0].doc_id == "doc_003"


def test_tfidf_search_returns_empty_list_for_empty_query() -> None:
    documents = sample_documents()
    engine = TfidfSearchEngine(documents)

    results = engine.search("", top_k=3)

    assert results == []


def test_tfidf_search_rejects_invalid_top_k() -> None:
    documents = sample_documents()
    engine = TfidfSearchEngine(documents)

    with pytest.raises(ValueError):
        engine.search("distributed training", top_k=0)


def test_tfidf_search_requires_documents() -> None:
    with pytest.raises(ValueError):
        TfidfSearchEngine([])
