"""Search engine factory for SearchEval Lab."""

from __future__ import annotations

from searcheval.datasets.schema import Document
from searcheval.search.base import SearchEngine
from searcheval.search.bm25 import BM25SearchEngine
from searcheval.search.hybrid import HybridSearchEngine
from searcheval.search.tfidf import TfidfSearchEngine

SUPPORTED_SEARCH_ENGINES = {
    "bm25",
    "hybrid",
    "tfidf",
}


def supported_search_engines() -> list[str]:
    """Return supported search engine names."""
    return sorted(SUPPORTED_SEARCH_ENGINES)


def build_search_engine(
    engine_name: str,
    documents: list[Document],
) -> SearchEngine:
    """Build a search engine by name."""
    normalized_engine_name = engine_name.lower().strip()

    if normalized_engine_name == "bm25":
        return BM25SearchEngine(documents)

    if normalized_engine_name == "hybrid":
        return HybridSearchEngine(documents)

    if normalized_engine_name == "tfidf":
        return TfidfSearchEngine(documents)

    supported = ", ".join(supported_search_engines())

    raise ValueError(f"Unsupported search engine '{engine_name}'. Supported engines: {supported}")
