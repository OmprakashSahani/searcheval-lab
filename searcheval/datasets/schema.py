"""Dataset schemas for SearchEval Lab."""

from pydantic import BaseModel, Field


class Document(BaseModel):
    """A searchable document in the benchmark dataset."""

    doc_id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class Query(BaseModel):
    """A benchmark query used for retrieval evaluation."""

    query_id: str = Field(..., min_length=1)
    query: str = Field(..., min_length=1)


class Qrel(BaseModel):
    """A relevance label connecting a query to a document."""

    query_id: str = Field(..., min_length=1)
    doc_id: str = Field(..., min_length=1)
    relevance: int = Field(..., ge=0, le=3)


class SearchResult(BaseModel):
    """A single retrieved search result."""

    doc_id: str = Field(..., min_length=1)
    score: float
    rank: int = Field(..., ge=1)


class QueryResults(BaseModel):
    """Top-k search results for a benchmark query."""

    query_id: str = Field(..., min_length=1)
    results: list[SearchResult]
