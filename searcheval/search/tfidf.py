"""TF-IDF search engine for SearchEval Lab."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from searcheval.datasets.schema import Document, SearchResult
from searcheval.search.base import SearchEngine, validate_top_k


class TfidfSearchEngine(SearchEngine):
    """Lexical search engine using TF-IDF scoring.

    This is the first baseline retrieval method for SearchEval Lab.

    TF-IDF is useful as a simple and explainable lexical baseline because it
    ranks documents based on word overlap between the query and document text.
    """

    def __init__(self, documents: list[Document]) -> None:
        """Build a TF-IDF index over benchmark documents."""
        super().__init__(documents)

        if not documents:
            raise ValueError("TfidfSearchEngine requires at least one document.")

        self.doc_ids = [document.doc_id for document in documents]
        self.corpus = [f"{document.title} {document.text}" for document in documents]

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
        )

        self.document_matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int) -> list[SearchResult]:
        """Return top-k TF-IDF search results for a query."""
        validate_top_k(top_k)

        if not query.strip():
            return []

        query_vector = self.vectorizer.transform([query])
        scores = (self.document_matrix @ query_vector.T).toarray().ravel()

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
