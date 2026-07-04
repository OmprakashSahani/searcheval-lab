"""Dataset loading utilities for SearchEval Lab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from searcheval.datasets.schema import Document, Query, Qrel

ModelT = TypeVar("ModelT", bound=BaseModel)


@dataclass(frozen=True)
class SearchDataset:
    """A loaded search evaluation dataset."""

    documents: list[Document]
    queries: list[Query]
    qrels: list[Qrel]


def load_jsonl(path: Path, model: type[ModelT]) -> list[ModelT]:
    """Load a JSONL file and validate each row with a Pydantic model."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    items: list[ModelT] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                raw_item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_number}: {exc}") from exc

            try:
                item = model.model_validate(raw_item)
            except ValidationError as exc:
                raise ValueError(
                    f"Schema validation failed in {path} at line {line_number}: {exc}"
                ) from exc

            items.append(item)

    return items


def load_documents(dataset_path: Path) -> list[Document]:
    """Load benchmark documents."""
    return load_jsonl(dataset_path / "documents.jsonl", Document)


def load_queries(dataset_path: Path) -> list[Query]:
    """Load benchmark queries."""
    return load_jsonl(dataset_path / "queries.jsonl", Query)


def load_qrels(dataset_path: Path) -> list[Qrel]:
    """Load benchmark relevance labels."""
    return load_jsonl(dataset_path / "qrels.jsonl", Qrel)


def load_dataset(dataset_path: Path) -> SearchDataset:
    """Load a full search evaluation dataset."""
    return SearchDataset(
        documents=load_documents(dataset_path),
        queries=load_queries(dataset_path),
        qrels=load_qrels(dataset_path),
    )
