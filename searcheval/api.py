"""FastAPI backend for SearchEval Lab."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from searcheval import __version__
from searcheval.benchmarks.runner import benchmark_summary, run_benchmark
from searcheval.benchmarks.store import save_benchmark_run
from searcheval.datasets.loader import load_dataset
from searcheval.datasets.validator import validate_dataset
from searcheval.reports.markdown import save_markdown_report
from searcheval.search.factory import build_search_engine, supported_search_engines

api_app = FastAPI(
    title="SearchEval Lab API",
    description="API for search evaluation, benchmarking, and regression analysis.",
    version=__version__,
)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str


class EnginesResponse(BaseModel):
    """Supported search engines response."""

    engines: list[str]


class DatasetValidationRequest(BaseModel):
    """Request body for dataset validation."""

    dataset_path: str = Field(..., min_length=1)


class DatasetValidationResponse(BaseModel):
    """Dataset validation API response."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class BenchmarkRequest(BaseModel):
    """Request body for running a benchmark."""

    dataset_path: str = Field(..., min_length=1)
    engine: str = Field("tfidf", min_length=1)
    k: int = Field(10, gt=0)
    save_artifacts: bool = True
    runs_dir: str = Field("runs/api", min_length=1)


class BenchmarkResponse(BaseModel):
    """Benchmark API response."""

    engine: str
    k: int
    summary: dict[str, Any]
    run_dir: str | None = None
    report_path: str | None = None


@api_app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(
        status="ok",
        version=__version__,
    )


@api_app.get("/engines", response_model=EnginesResponse)
def engines() -> EnginesResponse:
    """Return supported search engines."""
    return EnginesResponse(
        engines=supported_search_engines(),
    )


@api_app.post("/datasets/validate", response_model=DatasetValidationResponse)
def validate_dataset_endpoint(
    request: DatasetValidationRequest,
) -> DatasetValidationResponse:
    """Validate a search evaluation dataset."""
    try:
        result = validate_dataset(Path(request.dataset_path))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DatasetValidationResponse(
        is_valid=result.is_valid,
        errors=result.errors,
        warnings=result.warnings,
    )


@api_app.post("/benchmarks/run", response_model=BenchmarkResponse)
def run_benchmark_endpoint(
    request: BenchmarkRequest,
) -> BenchmarkResponse:
    """Run a search benchmark through the API."""
    dataset_path = Path(request.dataset_path)

    try:
        validation_result = validate_dataset(dataset_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not validation_result.is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Dataset validation failed.",
                "errors": validation_result.errors,
                "warnings": validation_result.warnings,
            },
        )

    try:
        dataset = load_dataset(dataset_path)
        search_engine = build_search_engine(
            engine_name=request.engine,
            documents=dataset.documents,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    normalized_engine_name = request.engine.lower().strip()

    benchmark_run = run_benchmark(
        dataset=dataset,
        search_engine=search_engine,
        engine_name=normalized_engine_name,
        k=request.k,
    )

    summary = benchmark_summary(benchmark_run)

    run_dir: Path | None = None
    report_path: Path | None = None

    if request.save_artifacts:
        run_dir = save_benchmark_run(
            run=benchmark_run,
            runs_dir=Path(request.runs_dir),
        )

        report_path = save_markdown_report(
            run=benchmark_run,
            output_path=run_dir / "report.md",
            run_id=run_dir.name,
        )

    return BenchmarkResponse(
        engine=normalized_engine_name,
        k=request.k,
        summary=summary,
        run_dir=str(run_dir) if run_dir else None,
        report_path=str(report_path) if report_path else None,
    )
