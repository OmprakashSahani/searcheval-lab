"""Command-line interface for SearchEval Lab."""

from pathlib import Path

import typer
from rich.console import Console

from searcheval.datasets.validator import validate_dataset

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

    console.print(
        "[yellow]Benchmark runner is not implemented yet. "
        "This command is a CLI placeholder for the MVP.[/yellow]"
    )


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
