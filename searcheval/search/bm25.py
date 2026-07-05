"""BM25 search engine for SearchEval Lab."""

from __future__ import annotations

import math
import re
from collections import Counter

from searcheval.datasets.schema import Document, SearchResult
from searcheval.search.base import SearchEngine, validate_top_k

TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25SearchEngine(SearchEngine):
    """Lexical search engine using BM25 scoring.

    BM25 is a classic information retrieval ranking function. It improves on
    simple keyword matching by considering term frequency, inverse document
    frequency, and document length normalization.
    """

    def __init__(
        self,
        documents: list[Document],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Build a BM25 index over benchmark documents."""
        super().__init__(documents)

        if not documents:
            raise ValueError("BM25SearchEngine requires at least one document.")

        if k1 <= 0:
            raise ValueError("k1 must be greater than 0.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = k1
        self.b = b

        self.doc_ids = [document.doc_id for document in documents]
        self.corpus = [f"{document.title} {document.text}" for document in documents]

        self.tokenized_documents = [tokenize(text) for text in self.corpus]

        self.document_lengths = [len(tokens) for tokens in self.tokenized_documents]

        self.avg_document_length = sum(self.document_lengths) / len(self.document_lengths)

        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_documents]

        self.document_frequencies = self._build_document_frequencies()
        self.total_documents = len(documents)

    def _build_document_frequencies(self) -> dict[str, int]:
        """Build term -> number of documents containing the term."""
        document_frequencies: dict[str, int] = {}

        for tokens in self.tokenized_documents:
            unique_terms = set(tokens)

            for term in unique_terms:
                document_frequencies[term] = document_frequencies.get(term, 0) + 1

        return document_frequencies

    def inverse_document_frequency(self, term: str) -> float:
        """Compute BM25 inverse document frequency for a term."""
        document_frequency = self.document_frequencies.get(term, 0)

        return math.log(
            1 + (self.total_documents - document_frequency + 0.5) / (document_frequency + 0.5)
        )

    def score_document(self, query_terms: list[str], document_index: int) -> float:
        """Compute BM25 score for one document."""
        score = 0.0

        term_frequency = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]

        for term in query_terms:
            frequency = term_frequency.get(term, 0)

            if frequency == 0:
                continue

            idf = self.inverse_document_frequency(term)

            numerator = frequency * (self.k1 + 1)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * (document_length / self.avg_document_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top-k BM25 search results for a query."""
        validate_top_k(top_k)

        query_terms = tokenize(query)

        if not query_terms:
            return []

        scores = [
            self.score_document(query_terms=query_terms, document_index=index)
            for index in range(len(self.documents))
        ]

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], self.doc_ids[index]),
        )

        results: list[SearchResult] = []

        for rank, index in enumerate(ranked_indices[:top_k], start=1):
            results.append(
                SearchResult(
                    doc_id=self.doc_ids[index],
                    score=float(scores[index]),
                    rank=rank,
                )
            )

        return results
