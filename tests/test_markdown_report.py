"""Tests for Markdown benchmark report generation."""

from searcheval.benchmarks.runner import BenchmarkRun, QueryLatency
from searcheval.eval.evaluator import EvaluationReport, QueryEvaluation
from searcheval.reports.markdown import (
    format_metric_value,
    render_latency_table,
    render_markdown_report,
    render_per_query_table,
    render_summary_table,
    render_weak_query_table,
    save_markdown_report,
)


def sample_benchmark_run() -> BenchmarkRun:
    """Create a sample benchmark run for report tests."""
    evaluation = EvaluationReport(
        k=10,
        aggregate_metrics={
            "precision_at_10": 0.5,
            "recall_at_10": 1.0,
            "mrr_at_10": 1.0,
            "ndcg_at_10": 0.95,
        },
        per_query=[
            QueryEvaluation(
                query_id="q_001",
                metrics={
                    "precision_at_10": 0.5,
                    "recall_at_10": 1.0,
                    "mrr_at_10": 1.0,
                    "ndcg_at_10": 0.95,
                },
                retrieved_doc_ids=["doc_001", "doc_002"],
                relevant_doc_ids=["doc_001"],
            ),
            QueryEvaluation(
                query_id="q_002",
                metrics={
                    "precision_at_10": 0.1,
                    "recall_at_10": 0.5,
                    "mrr_at_10": 0.25,
                    "ndcg_at_10": 0.20,
                },
                retrieved_doc_ids=["doc_003", "doc_004"],
                relevant_doc_ids=["doc_005"],
            ),
        ],
    )

    return BenchmarkRun(
        engine_name="tfidf",
        k=10,
        evaluation=evaluation,
        query_latencies=[
            QueryLatency(query_id="q_001", latency_ms=1.25),
            QueryLatency(query_id="q_002", latency_ms=1.75),
        ],
    )


def test_format_metric_value_formats_float() -> None:
    assert format_metric_value(0.123456) == "0.1235"


def test_format_metric_value_formats_non_float() -> None:
    assert format_metric_value("tfidf") == "tfidf"


def test_render_summary_table_contains_metrics() -> None:
    run = sample_benchmark_run()

    table = render_summary_table(run)

    assert "| Metric | Value |" in table
    assert "| precision_at_10 | 0.5000 |" in table
    assert "| recall_at_10 | 1.0000 |" in table
    assert "| engine | tfidf |" in table


def test_render_per_query_table_contains_query_metrics() -> None:
    run = sample_benchmark_run()

    table = render_per_query_table(run)

    assert "| Query ID | precision_at_10 | recall_at_10 | mrr_at_10 | ndcg_at_10 |" in table
    assert "q_001" in table
    assert "q_002" in table
    assert "doc_001, doc_002" in table
    assert "doc_003, doc_004" in table


def test_render_weak_query_table_contains_lowest_scoring_query() -> None:
    run = sample_benchmark_run()

    table = render_weak_query_table(
        run=run,
        metric_name="ndcg_at_10",
        limit=1,
    )

    assert "| Query ID | Metric | Score | Retrieved Docs | Relevant Docs |" in table
    assert "q_002" in table
    assert "ndcg_at_10" in table
    assert "0.2000" in table
    assert "doc_003, doc_004" in table
    assert "doc_005" in table
    assert "q_001" not in table


def test_render_latency_table_contains_latency() -> None:
    run = sample_benchmark_run()

    table = render_latency_table(run)

    assert "| Query ID | Latency ms |" in table
    assert "| q_001 | 1.2500 |" in table
    assert "| q_002 | 1.7500 |" in table


def test_render_markdown_report_contains_sections() -> None:
    run = sample_benchmark_run()

    report = render_markdown_report(run=run, run_id="run_test_tfidf")

    assert "# SearchEval Lab Benchmark Report" in report
    assert "## Run Summary" in report
    assert "## Aggregate Metrics" in report
    assert "## Per-Query Metrics" in report
    assert "## Weak Query Analysis" in report
    assert "## Query Latency" in report
    assert "**Run ID:** `run_test_tfidf`" in report
    assert "**Engine:** `tfidf`" in report
    assert "**Top-K:** `10`" in report
    assert "Weakest queries ranked by `ndcg_at_10`." in report


def test_save_markdown_report_creates_file(tmp_path) -> None:
    run = sample_benchmark_run()
    output_path = tmp_path / "report.md"

    saved_path = save_markdown_report(
        run=run,
        output_path=output_path,
        run_id="run_test_tfidf",
    )

    assert saved_path == output_path
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    assert "# SearchEval Lab Benchmark Report" in content
    assert "**Run ID:** `run_test_tfidf`" in content
    assert "## Weak Query Analysis" in content
