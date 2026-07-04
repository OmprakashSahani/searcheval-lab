"""Command-line interface for SearchEval Lab."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from searcheval.benchmarks.compare import compare_run_dirs, save_comparison_result
from searcheval.benchmarks.runner import benchmark_summary, run_benchmark
from searcheval.benchmarks.store import save_benchmark_run
from searcheval.datasets.loader import load_dataset
from searcheval.datasets.validator import validate_dataset
from searcheval.regression.config import load_regression_thresholds
from searcheval.reports.markdown import save_markdown_report
from searcheval.search.tfidf import TfidfSearchEngine

app = typer.Typer(
    name="searcheval",
    help="Evaluation and benchmarking infrastructure for search and retrieval systems.",
)

console = Console()


@app.command()
def version() -> None:
    """Show the installed SearchEval Lab version."""
    from searcheval import __version__

    console.print(f"SearchEval Lab version: {__version__}")


@app.command()
def validate(dataset_path: Path) -> None:
    """Validate a search evaluation dataset.

    Expected dataset files:
    - documents.jsonl
    - queries.jsonl
    - qrels.jsonl
    """
    console.print(f"[bold blue]Validating dataset:[/bold blue] {dataset_path}")

    try:
        result = validate_dataset(dataset_path)
    except FileNotFoundError as exc:
        console.print("[bold red]Dataset validation failed.[/bold red]")
        console.print(f"- {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print("[bold red]Dataset validation failed.[/bold red]")
        console.print(f"- {exc}")
        raise typer.Exit(code=1) from exc

    if result.errors:
        console.print("[bold red]Dataset validation failed.[/bold red]")
        for error in result.errors:
            console.print(f"- {error}")

    if result.warnings:
        console.print("[bold yellow]Warnings:[/bold yellow]")
        for warning in result.warnings:
            console.print(f"- {warning}")

    if not result.is_valid:
        raise typer.Exit(code=1)

    console.print("[bold green]Dataset validation passed.[/bold green]")


@app.command()
def run(
    dataset_path: Path,
    engine: str = typer.Option("tfidf", help="Search engine to evaluate."),
    k: int = typer.Option(10, help="Number of top results to retrieve."),
    runs_dir: Path = typer.Option(
        Path("runs"),
        "--runs-dir",
        help="Directory where benchmark run artifacts will be saved.",
    ),
) -> None:
    """Run a benchmark for a search method."""
    console.print("[bold blue]Starting benchmark run[/bold blue]")
    console.print(f"Dataset: {dataset_path}")
    console.print(f"Engine: {engine}")
    console.print(f"Top-K: {k}")

    if engine != "tfidf":
        console.print(f"[bold red]Unsupported engine:[/bold red] {engine}")
        console.print("Currently supported engines: tfidf")
        raise typer.Exit(code=1)

    try:
        validation_result = validate_dataset(dataset_path)
    except FileNotFoundError as exc:
        console.print("[bold red]Benchmark failed.[/bold red]")
        console.print(f"- {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print("[bold red]Benchmark failed.[/bold red]")
        console.print(f"- {exc}")
        raise typer.Exit(code=1) from exc

    if not validation_result.is_valid:
        console.print("[bold red]Benchmark failed because dataset validation failed.[/bold red]")
        for error in validation_result.errors:
            console.print(f"- {error}")
        raise typer.Exit(code=1)

    dataset = load_dataset(dataset_path)
    search_engine = TfidfSearchEngine(dataset.documents)

    benchmark_run = run_benchmark(
        dataset=dataset,
        search_engine=search_engine,
        engine_name=engine,
        k=k,
    )

    summary = benchmark_summary(benchmark_run)

    run_dir = save_benchmark_run(
        run=benchmark_run,
        runs_dir=runs_dir,
    )

    report_path = save_markdown_report(
        run=benchmark_run,
        output_path=run_dir / "report.md",
        run_id=run_dir.name,
    )

    table = Table(title="Benchmark Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    for metric_name, value in summary.items():
        if isinstance(value, float):
            table.add_row(metric_name, f"{value:.4f}")
        else:
            table.add_row(metric_name, str(value))

    console.print(table)
    console.print(f"[bold green]Benchmark artifacts saved to:[/bold green] {run_dir}")
    console.print(f"[bold green]Markdown report saved to:[/bold green] {report_path}")
    console.print("[bold green]Benchmark run completed.[/bold green]")


@app.command()
def compare(
    baseline_run: Path,
    current_run: Path,
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Optional regression threshold config JSON file.",
    ),
    output: Path | None = typer.Option(
        None,
        "--output",
        help="Optional path where the comparison JSON result will be saved.",
    ),
) -> None:
    """Compare two saved benchmark runs for regressions."""
    console.print("[bold blue]Comparing benchmark runs[/bold blue]")
    console.print(f"Baseline run: {baseline_run}")
    console.print(f"Current run: {current_run}")

    thresholds = None

    if config is not None:
        try:
            thresholds = load_regression_thresholds(config)
        except FileNotFoundError as exc:
            console.print("[bold red]Comparison failed.[/bold red]")
            console.print(f"- {exc}")
            raise typer.Exit(code=1) from exc
        except ValueError as exc:
            console.print("[bold red]Comparison failed.[/bold red]")
            console.print(f"- {exc}")
            raise typer.Exit(code=1) from exc

        console.print(f"[bold blue]Using regression config:[/bold blue] {config}")

    try:
        result = compare_run_dirs(
            baseline_run_dir=baseline_run,
            current_run_dir=current_run,
            thresholds=thresholds,
        )
    except FileNotFoundError as exc:
        console.print("[bold red]Comparison failed.[/bold red]")
        console.print(f"- {exc}")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        console.print("[bold red]Comparison failed.[/bold red]")
        console.print(f"- {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Regression Check")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Current", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Threshold", justify="right")
    table.add_column("Status", justify="center")

    for comparison in result.comparisons:
        status = "[green]PASS[/green]" if comparison.passed else "[red]FAIL[/red]"

        table.add_row(
            comparison.metric_name,
            f"{comparison.baseline_value:.4f}",
            f"{comparison.current_value:.4f}",
            f"{comparison.delta:.4f}",
            f"{comparison.threshold:.4f}",
            status,
        )

    console.print(table)

    if result.missing_metrics:
        console.print("[bold yellow]Missing metrics:[/bold yellow]")
        for metric_name in result.missing_metrics:
            console.print(f"- {metric_name}")

    if output is None:
        output = current_run / "comparison.json"

    saved_path = save_comparison_result(
        result=result,
        output_path=output,
    )

    console.print(f"[bold blue]Comparison result saved to:[/bold blue] {saved_path}")

    if result.passed:
        console.print("[bold green]Regression check passed.[/bold green]")
    else:
        console.print("[bold red]Regression check failed.[/bold red]")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
