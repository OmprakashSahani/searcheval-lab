"""Command-line interface for SearchEval Lab."""

from pathlib import Path

import typer
from rich.console import Console

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

    required_files = [
        "documents.jsonl",
        "queries.jsonl",
        "qrels.jsonl",
    ]

    missing_files = [
        file_name for file_name in required_files if not (dataset_path / file_name).exists()
    ]

    if missing_files:
        console.print("[bold red]Dataset validation failed.[/bold red]")
        for file_name in missing_files:
            console.print(f"- Missing file: {file_name}")
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
