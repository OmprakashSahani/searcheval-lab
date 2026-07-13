"""Tests for the SearchEval Lab CLI."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from searcheval.cli import app

runner = CliRunner()


def write_summary(run_dir: Path, summary: dict[str, float]) -> None:
    """Write a saved benchmark summary.json file for CLI compare tests."""
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_dir.name,
        "engine": "tfidf",
        "k": 10,
        "summary": summary,
    }

    (run_dir / "summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def test_cli_version() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert "SearchEval Lab version:" in result.output


def test_cli_validate_sample_dataset() -> None:
    result = runner.invoke(app, ["validate", "data/search_eval_small"])

    assert result.exit_code == 0
    assert "Dataset validation passed." in result.output


def test_cli_run_tfidf_benchmark(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "data/search_eval_small",
            "--engine",
            "tfidf",
            "--k",
            "10",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Benchmark run completed." in result.output
    assert "Benchmark artifacts saved to:" in result.output
    assert "Markdown report saved to:" in result.output

    run_dirs = list(tmp_path.iterdir())

    assert len(run_dirs) == 1
    assert run_dirs[0].name.endswith("_tfidf")
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "report.md").exists()


def test_cli_run_bm25_benchmark(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "data/search_eval_small",
            "--engine",
            "bm25",
            "--k",
            "10",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Benchmark run completed." in result.output

    run_dirs = list(tmp_path.iterdir())

    assert len(run_dirs) == 1
    assert run_dirs[0].name.endswith("_bm25")


def test_cli_run_hybrid_benchmark(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "data/search_eval_small",
            "--engine",
            "hybrid",
            "--k",
            "10",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Benchmark run completed." in result.output

    run_dirs = list(tmp_path.iterdir())

    assert len(run_dirs) == 1
    assert run_dirs[0].name.endswith("_hybrid")
    assert (run_dirs[0] / "summary.json").exists()
    assert (run_dirs[0] / "report.md").exists()


def test_cli_run_rejects_unknown_engine(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "run",
            "data/search_eval_small",
            "--engine",
            "unknown",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 1
    assert "Unsupported engine:" in result.output


def test_cli_compare_saved_runs(tmp_path) -> None:
    baseline_run = tmp_path / "run_baseline_tfidf"
    current_run = tmp_path / "run_current_bm25"
    config_path = tmp_path / "regression.json"

    write_summary(
        run_dir=baseline_run,
        summary={
            "precision_at_10": 0.18,
            "recall_at_10": 0.95,
            "mrr_at_10": 0.95,
            "ndcg_at_10": 0.91,
            "latency_avg_ms": 1.0,
            "latency_max_ms": 2.0,
        },
    )

    write_summary(
        run_dir=current_run,
        summary={
            "precision_at_10": 0.19,
            "recall_at_10": 1.0,
            "mrr_at_10": 0.95,
            "ndcg_at_10": 0.92,
            "latency_avg_ms": 1.2,
            "latency_max_ms": 2.5,
        },
    )

    config_path.write_text(
        json.dumps(
            {
                "thresholds": {
                    "precision_at_10": -0.05,
                    "recall_at_10": -0.03,
                    "mrr_at_10": -0.03,
                    "ndcg_at_10": -0.02,
                    "latency_avg_ms": 5.0,
                    "latency_max_ms": 10.0,
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "compare",
            str(baseline_run),
            str(current_run),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0
    assert "Regression check passed." in result.output
    assert (current_run / "comparison.json").exists()
