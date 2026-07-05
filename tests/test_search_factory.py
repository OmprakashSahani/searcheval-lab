"""Tests for the search engine factory."""

import pytest

from searcheval.datasets.schema import Document
from searcheval.search.bm25 import BM25SearchEngine
from searcheval.search.factory import build_search_engine, supported_search_engines
from searcheval.search.tfidf import TfidfSearchEngine


def sample_documents() -> list[Document]:
    """Create a small document collection for factory tests."""
    return [
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
    ]


def test_supported_search_engines_returns_sorted_engine_names() -> None:
    assert supported_search_engines() == ["bm25", "tfidf"]


def test_build_search_engine_creates_tfidf_engine() -> None:
    engine = build_search_engine(
        engine_name="tfidf",
        documents=sample_documents(),
    )

    assert isinstance(engine, TfidfSearchEngine)


def test_build_search_engine_creates_bm25_engine() -> None:
    engine = build_search_engine(
        engine_name="bm25",
        documents=sample_documents(),
    )

    assert isinstance(engine, BM25SearchEngine)


def test_build_search_engine_normalizes_engine_name() -> None:
    engine = build_search_engine(
        engine_name=" BM25 ",
        documents=sample_documents(),
    )

    assert isinstance(engine, BM25SearchEngine)


def test_build_search_engine_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="Unsupported search engine"):
        build_search_engine(
            engine_name="unknown",
            documents=sample_documents(),
        )
