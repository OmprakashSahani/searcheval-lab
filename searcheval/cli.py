"""Command-line interface for SearchEval Lab."""

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from searcheval.benchmarks.runner import benchmark_summary, run_benchmark
from searcheval.datasets.loader import load_dataset
from searcheval.datasets.validator import validate_dataset
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

    table = Table(title="Benchmark Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    for metric_name, value in summary.items():
        if isinstance(value, float):
            table.add_row(metric_name, f"{value:.4f}")
        else:
            table.add_row(metric_name, str(value))

    console.print(table)
    console.print("[bold green]Benchmark run completed.[/bold green]")


@app.command()
def compare(
    baseline_run: Path,
    current_run: Path,
) -> None:
    """Compare two benchmark runs."""
    console.print("[bold blue]Comparing benchmark runs[/bold blue]")
    console.print(f"Baseline run: {baseline_run}")
    console.print(f"Current run: {current_run}")

    console.print(
        "[yellow]Run comparison is not implemented yet. "
        "This command is a CLI placeholder for the MVP.[/yellow]"
    )


if __name__ == "__main__":
    app()
