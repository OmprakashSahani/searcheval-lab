"""Base search interface for SearchEval Lab."""

from __future__ import annotations

from abc import ABC, abstractmethod

from searcheval.datasets.schema import Document, SearchResult


class SearchEngine(ABC):
    """Base interface for all search engines.

    Every retrieval method in SearchEval Lab should implement this interface.

    This keeps evaluation independent from the search method. The evaluator should
    not care whether results come from TF-IDF, BM25, vector search, hybrid search,
    or an external retrieval API.
    """

    def __init__(self, documents: list[Document]) -> None:
        """Initialize the search engine with benchmark documents."""
        self.documents = documents

    @abstractmethod
    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top-k search results for a query."""
        raise NotImplementedError


def validate_top_k(top_k: int) -> None:
    """Validate that top_k is a positive integer."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0.")
