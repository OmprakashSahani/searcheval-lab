# SearchEval Lab

Evaluation and benchmarking infrastructure for search and retrieval systems.

## Overview

SearchEval Lab is an ML systems and backend infrastructure project for evaluating search and retrieval quality.

It measures how well different search methods retrieve relevant documents for a set of benchmark queries. The project supports reproducible datasets, ranking metrics, benchmark runs, saved artifacts, Markdown reports, weak query analysis, configurable regression thresholds, and regression detection between runs.

This is not just a search demo. The main goal is to build the evaluation layer that helps search teams measure whether retrieval quality is improving or regressing over time.

## Project Positioning

SearchEval Lab is designed for roles related to:

- ML Systems Engineering
- Software Engineering
- Backend Engineering
- Search Infrastructure
- AI Infrastructure
- Research Evals
- Machine Learning Engineering

## Core Problem

Search systems need reliable ways to answer questions such as:

- Did the new retrieval method improve search quality?
- Are relevant documents appearing in the top results?
- Did a code change cause a search quality regression?
- Which queries are failing?
- How do different retrieval methods compare?
- What is the tradeoff between quality and latency?

SearchEval Lab answers these questions through benchmark-driven evaluation.

## Current MVP Features

The current version includes:

- JSONL dataset format for documents, queries, and relevance labels
- Dataset validation
- TF-IDF lexical search baseline
- BM25 lexical search baseline
- Ranking metrics:
  - Precision@K
  - Recall@K
  - MRR@K
  - NDCG@K
- Benchmark runner
- Query latency tracking
- Benchmark artifact storage
- Markdown benchmark report generation
- Weak query analysis
- Regression detection between benchmark runs
- Configurable regression thresholds
- CLI interface
- Example benchmark scripts
- Unit test suite
- GitHub Actions CI workflow

## High-Level Workflow

```text
Dataset
  ↓
Dataset Validation
  ↓
Search Method
  ↓
Top-K Retrieved Results
  ↓
Evaluation Metrics
  ↓
Benchmark Run
  ↓
Saved Artifacts
  ↓
Markdown Report
  ↓
Weak Query Analysis
  ↓
Regression Comparison
```

## Project Structure

```text
searcheval-lab/
│
├── README.md
├── pyproject.toml
│
├── configs/
│   └── regression.json
│
├── data/
│   └── search_eval_small/
│       ├── documents.jsonl
│       ├── queries.jsonl
│       └── qrels.jsonl
│
├── docs/
│   └── architecture.md
│
├── examples/
│   ├── compare_tfidf_bm25.sh
│   ├── run_bm25.sh
│   └── run_tfidf.sh
│
├── searcheval/
│   ├── __init__.py
│   ├── cli.py
│   │
│   ├── benchmarks/
│   │   ├── compare.py
│   │   ├── runner.py
│   │   └── store.py
│   │
│   ├── datasets/
│   │   ├── loader.py
│   │   ├── schema.py
│   │   └── validator.py
│   │
│   ├── eval/
│   │   ├── evaluator.py
│   │   ├── failure_analysis.py
│   │   └── metrics.py
│   │
│   ├── regression/
│   │   ├── config.py
│   │   └── detector.py
│   │
│   ├── reports/
│   │   └── markdown.py
│   │
│   └── search/
│       ├── base.py
│       ├── bm25.py
│       └── tfidf.py
│
├── tests/
│   ├── test_benchmark_compare.py
│   ├── test_benchmark_runner.py
│   ├── test_benchmark_store.py
│   ├── test_bm25_search.py
│   ├── test_evaluator.py
│   ├── test_failure_analysis.py
│   ├── test_markdown_report.py
│   ├── test_metrics.py
│   ├── test_regression_config.py
│   ├── test_regression_detector.py
│   └── test_tfidf_search.py
│
└── .github/
    └── workflows/
        └── ci.yml
```

## Dataset Format

SearchEval Lab uses three JSONL files.

### documents.jsonl

```json
{"doc_id": "doc_001", "title": "Transformer Architecture", "text": "Transformers use self-attention mechanisms for sequence modeling."}
```

### queries.jsonl

```json
{"query_id": "q_001", "query": "how do transformers use attention"}
```

### qrels.jsonl

```json
{"query_id": "q_001", "doc_id": "doc_001", "relevance": 3}
```

Relevance scale:

| Score | Meaning |
|---:|---|
| 0 | Not relevant |
| 1 | Weakly relevant |
| 2 | Relevant |
| 3 | Highly relevant |

## Installation

Install the project in editable mode:

```bash
python -m pip install -e ".[dev]"
```

## Developer Commands

Common commands are available through the Makefile.

Install dependencies:

```bash
make install
```

Run the full test suite:

```bash
make test
```

Run lint checks:

```bash
make lint
```

Check formatting:

```bash
make format-check
```

Validate the sample dataset:

```bash
make validate
```

Run TF-IDF benchmark:

```bash
make run-tfidf
```

Run BM25 benchmark:

```bash
make run-bm25
```

Run the TF-IDF vs BM25 comparison workflow:

```bash
make compare
```

Start the FastAPI backend:

```bash
make api
```

Run the API smoke test after the backend is running:

```bash
make api-smoke
```

Clean generated local artifacts:

```bash
make clean
```


## FastAPI Backend

SearchEval Lab also includes a FastAPI backend for running validation and benchmark workflows through HTTP endpoints.

Start the API server:

```bash
make api
```

In a second terminal, run the API smoke test:

```bash
make api-smoke
```

You can also run the script directly:

```bash
./examples/api_smoke_test.sh
```

The server runs at:

```txt
http://localhost:8000
```

Interactive API docs are available at:

```txt
http://localhost:8000/docs
```

### API Endpoints

Health check:

```bash
curl http://localhost:8000/health
```

List supported search engines:

```bash
curl http://localhost:8000/engines
```

Validate a dataset:

```bash
curl -X POST http://localhost:8000/datasets/validate \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "data/search_eval_small"}'
```

Run a benchmark without saving artifacts:

```bash
curl -X POST http://localhost:8000/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/search_eval_small",
    "engine": "tfidf",
    "k": 10,
    "save_artifacts": false
  }'
```

Run a BM25 benchmark and save artifacts:

```bash
curl -X POST http://localhost:8000/benchmarks/run \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_path": "data/search_eval_small",
    "engine": "bm25",
    "k": 10,
    "save_artifacts": true,
    "runs_dir": "runs/api"
  }'
```

List saved benchmark runs:

```bash
curl "http://localhost:8000/benchmarks/runs?runs_dir=runs/api"
```

Read one saved benchmark run:

```bash
curl "http://localhost:8000/benchmarks/runs/run_20260101_120000_tfidf?runs_dir=runs/api"
```

Read a saved Markdown benchmark report:

```bash
curl "http://localhost:8000/benchmarks/runs/run_20260101_120000_tfidf/report?runs_dir=runs/api"
```

This API layer makes the benchmark system usable as a backend service, not only as a CLI tool.

## Code Quality

SearchEval Lab uses Ruff for linting and formatting checks.

The GitHub Actions CI workflow runs:

```bash
make lint
make format-check
make test
```

This ensures that every pushed change passes code quality checks and the full test suite.

## CLI Usage

### Show version

```bash
python -m searcheval.cli version
```

### Validate dataset

```bash
python -m searcheval.cli validate data/search_eval_small
```

Expected result:

```text
Dataset validation passed.
```

### Run TF-IDF benchmark

```bash
python -m searcheval.cli run data/search_eval_small --engine tfidf --k 10
```

### Run BM25 benchmark

```bash
python -m searcheval.cli run data/search_eval_small --engine bm25 --k 10
```

The benchmark command prints a summary table with:

- Precision@10
- Recall@10
- MRR@10
- NDCG@10
- Average latency
- Minimum latency
- Maximum latency

It also saves benchmark artifacts under:

```text
runs/run_YYYYMMDD_HHMMSS_engine/
```

Each run includes:

```text
summary.json
metrics.json
per_query_metrics.json
latencies.json
report.md
```

## Example Scripts

Run the TF-IDF benchmark:

```bash
./examples/run_tfidf.sh
```

Run the BM25 benchmark:

```bash
./examples/run_bm25.sh
```

Run TF-IDF and BM25, then compare both benchmark runs:

```bash
./examples/compare_tfidf_bm25.sh
```

These scripts provide a simple end-to-end workflow for validating the dataset, running benchmarks, generating reports, and checking regressions.

## Benchmark Reports

Each generated Markdown report includes:

- Run summary
- Aggregate metrics
- Per-query metrics
- Weak query analysis
- Query latency table
- Notes about ranking metrics

The weak query analysis section ranks the lowest-performing queries by NDCG@K.

This helps identify where the retrieval method is failing and which queries need deeper investigation.

## Compare Search Methods

You can run two benchmark methods and compare their saved run directories.

Example:

```bash
python -m searcheval.cli run data/search_eval_small --engine tfidf --k 10
python -m searcheval.cli run data/search_eval_small --engine bm25 --k 10
```

Then compare the two saved runs:

```bash
python -m searcheval.cli compare \
  runs/run_baseline_tfidf \
  runs/run_current_bm25 \
  --config configs/regression.json
```

This checks whether the current run regressed beyond allowed thresholds.

## Regression Thresholds

Default regression thresholds are stored in:

```text
configs/regression.json
```

Example:

```json
{
  "thresholds": {
    "precision_at_10": -0.05,
    "recall_at_10": -0.03,
    "mrr_at_10": -0.03,
    "ndcg_at_10": -0.02,
    "latency_avg_ms": 5.0,
    "latency_max_ms": 10.0
  }
}
```

Negative thresholds are used for quality metrics where drops are bad.

Positive thresholds are used for latency metrics where increases are bad.

## Testing

Run the full test suite:

```bash
pytest
```

Current test coverage includes:

- Ranking metrics
- Dataset loading and validation
- TF-IDF search
- BM25 search
- Evaluation engine
- Query failure analysis
- Benchmark runner
- Benchmark artifact storage
- Markdown report generation
- Regression detection
- Regression config loading
- Benchmark run comparison

## Example Benchmark Output

Example TF-IDF result:

```text
precision_at_10  0.1800
recall_at_10     0.9500
mrr_at_10        0.9500
ndcg_at_10       0.9113
```

Example BM25 result:

```text
precision_at_10  0.1900
recall_at_10     1.0000
mrr_at_10        0.9500
ndcg_at_10       0.9198
```

Example TF-IDF to BM25 comparison:

```text
precision_at_10  +0.0100
recall_at_10     +0.0500
ndcg_at_10       +0.0085
```

## Why This Project Matters

Search quality is difficult to improve without reliable evaluation.

SearchEval Lab demonstrates how to build the infrastructure around search systems:

- reusable benchmark datasets
- consistent retrieval interfaces
- lexical search baselines
- ranking metrics
- reproducible benchmark runs
- weak query analysis
- regression detection
- automated reports
- CLI-based developer workflows

This makes the project relevant to ML systems, search infrastructure, backend engineering, and research evaluation roles.

## Future Work

Planned extensions:

- Embedding-based semantic search
- Hybrid lexical + vector search
- FastAPI backend
- HTML reports
- GitHub Actions quality gate for regression checks
- Dataset versioning
- Experiment comparison dashboard

## Author

Omprakash Sahani

ML Systems Engineer | Software Engineer | Distributed Systems
