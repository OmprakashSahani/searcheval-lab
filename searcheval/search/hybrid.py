"""Hybrid search engine for SearchEval Lab.

The hybrid engine combines BM25 and TF-IDF scores into a single ranking.
This provides a stronger retrieval baseline than using either lexical method alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from searcheval.datasets.schema import Document, SearchResult
from searcheval.search.base import SearchEngine, validate_top_k
from searcheval.search.bm25 import BM25SearchEngine
from searcheval.search.tfidf import TfidfSearchEngine


@dataclass(frozen=True)
class HybridWeights:
    """Weights used to combine BM25 and TF-IDF scores."""

    bm25: float = 0.5
    tfidf: float = 0.5


def validate_weights(weights: HybridWeights) -> None:
    """Validate hybrid search weights."""
    if weights.bm25 < 0:
        raise ValueError("BM25 weight must be non-negative.")

    if weights.tfidf < 0:
        raise ValueError("TF-IDF weight must be non-negative.")

    if weights.bm25 == 0 and weights.tfidf == 0:
        raise ValueError("At least one hybrid search weight must be positive.")


def normalize_scores(results: list[SearchResult]) -> dict[str, float]:
    """Normalize search scores to the range [0, 1]."""
    if not results:
        return {}

    scores = [result.score for result in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return {result.doc_id: 1.0 for result in results}

    return {
        result.doc_id: (result.score - min_score) / (max_score - min_score) for result in results
    }


class HybridSearchEngine(SearchEngine):
    """Hybrid BM25 + TF-IDF search engine."""

    def __init__(
        self,
        documents: list[Document],
        weights: HybridWeights | None = None,
    ) -> None:
        """Create a hybrid search engine."""
        if not documents:
            raise ValueError("HybridSearchEngine requires at least one document.")

        self.documents = documents
        self.weights = weights or HybridWeights()

        validate_weights(self.weights)

        self.bm25_engine = BM25SearchEngine(documents)
        self.tfidf_engine = TfidfSearchEngine(documents)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Search documents using combined BM25 and TF-IDF scores."""
        validate_top_k(top_k)

        candidate_k = len(self.documents)

        bm25_results = self.bm25_engine.search(
            query=query,
            top_k=candidate_k,
        )
        tfidf_results = self.tfidf_engine.search(
            query=query,
            top_k=candidate_k,
        )

        bm25_scores = normalize_scores(bm25_results)
        tfidf_scores = normalize_scores(tfidf_results)

        combined_scores: dict[str, float] = {}

        for document in self.documents:
            bm25_score = bm25_scores.get(document.doc_id, 0.0)
            tfidf_score = tfidf_scores.get(document.doc_id, 0.0)

            combined_scores[document.doc_id] = (
                self.weights.bm25 * bm25_score + self.weights.tfidf * tfidf_score
            )

        ranked_results = sorted(
            combined_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            SearchResult(
                doc_id=doc_id,
                score=float(score),
                rank=rank,
            )
            for rank, (doc_id, score) in enumerate(
                ranked_results[:top_k],
                start=1,
            )
        ]
