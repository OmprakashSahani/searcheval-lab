"""Dataset validation utilities for SearchEval Lab."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from searcheval.datasets.loader import SearchDataset, load_dataset


@dataclass(frozen=True)
class ValidationResult:
    """Result returned after validating a dataset."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


def find_duplicates(values: list[str]) -> set[str]:
    """Return duplicate values from a list."""
    seen: set[str] = set()
    duplicates: set[str] = set()

    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)

    return duplicates


def validate_loaded_dataset(dataset: SearchDataset) -> ValidationResult:
    """Validate an already loaded search evaluation dataset."""
    errors: list[str] = []
    warnings: list[str] = []

    document_ids = [document.doc_id for document in dataset.documents]
    query_ids = [query.query_id for query in dataset.queries]

    document_id_set = set(document_ids)
    query_id_set = set(query_ids)

    duplicate_document_ids = find_duplicates(document_ids)
    duplicate_query_ids = find_duplicates(query_ids)

    if duplicate_document_ids:
        errors.append(
            "Duplicate document IDs found: "
            + ", ".join(sorted(duplicate_document_ids))
        )

    if duplicate_query_ids:
        errors.append(
            "Duplicate query IDs found: "
            + ", ".join(sorted(duplicate_query_ids))
        )

    if not dataset.documents:
        errors.append("Dataset contains no documents.")

    if not dataset.queries:
        errors.append("Dataset contains no queries.")

    if not dataset.qrels:
        errors.append("Dataset contains no relevance labels.")

    qrels_by_query: dict[str, list[str]] = {}

    for qrel in dataset.qrels:
        if qrel.query_id not in query_id_set:
            errors.append(
                f"Qrel references missing query_id '{qrel.query_id}'."
            )

        if qrel.doc_id not in document_id_set:
            errors.append(
                f"Qrel references missing doc_id '{qrel.doc_id}'."
            )

        qrels_by_query.setdefault(qrel.query_id, []).append(qrel.doc_id)

    for query_id in query_id_set:
        if query_id not in qrels_by_query:
            warnings.append(
                f"Query '{query_id}' has no relevance labels."
            )

    for query_id, doc_ids in qrels_by_query.items():
        duplicate_qrel_docs = find_duplicates(doc_ids)
        if duplicate_qrel_docs:
            errors.append(
                f"Query '{query_id}' has duplicate qrel documents: "
                + ", ".join(sorted(duplicate_qrel_docs))
            )

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def validate_dataset(dataset_path: Path) -> ValidationResult:
    """Load and validate a search evaluation dataset."""
    dataset = load_dataset(dataset_path)
    return validate_loaded_dataset(dataset)
