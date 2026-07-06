"""FastAPI backend for SearchEval Lab."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from searcheval import __version__
from searcheval.benchmarks.compare import compare_run_dirs, save_comparison_result
from searcheval.benchmarks.runner import benchmark_summary, run_benchmark
from searcheval.benchmarks.store import save_benchmark_run
from searcheval.datasets.loader import load_dataset
from searcheval.datasets.validator import validate_dataset
from searcheval.regression.config import load_regression_thresholds
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


class SavedRunResponse(BaseModel):
    """Saved benchmark run summary response."""

    run_id: str
    engine: str
    k: int
    run_dir: str
    summary: dict[str, Any]


class CompareRequest(BaseModel):
    """Request body for comparing two saved benchmark runs."""

    baseline_run: str = Field(..., min_length=1)
    current_run: str = Field(..., min_length=1)
    config_path: str | None = None
    output_path: str | None = None


class CompareResponse(BaseModel):
    """Benchmark comparison API response."""

    passed: bool
    comparison: dict[str, Any]
    output_path: str | None = None


def load_saved_run_summary(run_dir: Path) -> SavedRunResponse:
    """Load a saved benchmark run summary."""
    summary_path = run_dir / "summary.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found: {summary_path}")

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid summary JSON file: {summary_path}") from exc

    summary = payload.get("summary", {})

    if not isinstance(summary, dict):
        raise ValueError(f"Invalid summary payload in: {summary_path}")

    return SavedRunResponse(
        run_id=str(payload.get("run_id", run_dir.name)),
        engine=str(payload.get("engine", "unknown")),
        k=int(payload.get("k", 0)),
        run_dir=str(run_dir),
        summary=summary,
    )


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


@api_app.get("/benchmarks/runs", response_model=list[SavedRunResponse])
def list_benchmark_runs_endpoint(
    runs_dir: str = "runs/api",
) -> list[SavedRunResponse]:
    """List saved benchmark runs."""
    root = Path(runs_dir)

    if not root.exists():
        return []

    if not root.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Runs path is not a directory: {root}",
        )

    saved_runs: list[SavedRunResponse] = []

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        summary_path = run_dir / "summary.json"

        if not summary_path.exists():
            continue

        try:
            saved_runs.append(load_saved_run_summary(run_dir))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return saved_runs


@api_app.get("/benchmarks/runs/{run_id}", response_model=SavedRunResponse)
def get_benchmark_run_endpoint(
    run_id: str,
    runs_dir: str = "runs/api",
) -> SavedRunResponse:
    """Read one saved benchmark run summary."""
    run_dir = Path(runs_dir) / run_id

    if not run_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark run not found: {run_id}",
        )

    if not run_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark run path is not a directory: {run_dir}",
        )

    try:
        return load_saved_run_summary(run_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@api_app.get(
    "/benchmarks/runs/{run_id}/report",
    response_class=PlainTextResponse,
)
def get_benchmark_report_endpoint(
    run_id: str,
    runs_dir: str = "runs/api",
) -> str:
    """Read one saved benchmark Markdown report."""
    run_dir = Path(runs_dir) / run_id

    if not run_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark run not found: {run_id}",
        )

    if not run_dir.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Benchmark run path is not a directory: {run_dir}",
        )

    report_path = run_dir / "report.md"

    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Benchmark report not found: {report_path}",
        )

    return report_path.read_text(encoding="utf-8")


@api_app.post("/benchmarks/compare", response_model=CompareResponse)
def compare_benchmark_runs_endpoint(
    request: CompareRequest,
) -> CompareResponse:
    """Compare two saved benchmark runs through the API."""
    baseline_run = Path(request.baseline_run)
    current_run = Path(request.current_run)

    if not baseline_run.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Baseline run directory not found: {baseline_run}",
        )

    if not current_run.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Current run directory not found: {current_run}",
        )

    thresholds: dict[str, float] | None = None

    if request.config_path is not None:
        try:
            thresholds = load_regression_thresholds(Path(request.config_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        comparison_result = compare_run_dirs(
            baseline_run_dir=baseline_run,
            current_run_dir=current_run,
            thresholds=thresholds,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output_path: Path | None = None

    if request.output_path is not None:
        output_path = save_comparison_result(
            result=comparison_result,
            output_path=Path(request.output_path),
        )

    return CompareResponse(
        passed=comparison_result.passed,
        comparison=asdict(comparison_result),
        output_path=str(output_path) if output_path else None,
    )
