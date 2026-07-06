# SearchEval Lab Architecture

SearchEval Lab is a benchmarking and evaluation system for search quality, retrieval baselines, latency tracking, regression detection, and reproducible benchmark reporting.

The system can be used in two ways:

1. As a developer CLI tool.
2. As a FastAPI backend service.

---

## High-Level Architecture

```txt
Dataset Files
documents.jsonl
queries.jsonl
qrels.jsonl
      |
      v
Dataset Loader + Validator
      |
      v
Search Engine Factory
      |
      +------------------+
      |                  |
      v                  v
TF-IDF Search        BM25 Search
      |                  |
      +--------+---------+
               |
               v
Benchmark Runner
               |
               v
Evaluation Engine
Precision@K, Recall@K, MRR@K, NDCG@K
               |
               v
Artifact Store
summary.json
metrics.json
per_query_metrics.json
latencies.json
report.md
comparison.json
               |
               +-------------------+
               |                   |
               v                   v
Regression Compare        Markdown Reports
               |
               v
CLI + FastAPI Access Layer
```

---

## Main Components

### 1. Dataset Layer

The dataset layer loads and validates search evaluation data.

Files:

```txt
searcheval/datasets/schema.py
searcheval/datasets/loader.py
searcheval/datasets/validator.py
```

Dataset format:

```txt
documents.jsonl
queries.jsonl
qrels.jsonl
```

Responsibilities:

- Parse JSONL files.
- Validate document IDs, query IDs, and relevance labels.
- Detect missing references.
- Detect duplicate records.
- Provide typed dataset objects to the benchmark runner.

---

### 2. Search Engine Layer

The search layer provides interchangeable retrieval methods.

Files:

```txt
searcheval/search/base.py
searcheval/search/tfidf.py
searcheval/search/bm25.py
searcheval/search/factory.py
```

Current engines:

```txt
tfidf
bm25
```

Responsibilities:

- Provide a common search interface.
- Rank documents for each query.
- Return top-k search results.
- Allow CLI and API workflows to select search engines dynamically.

The search engine factory keeps the CLI and API independent from specific search engine implementations.

---

### 3. Evaluation Layer

The evaluation layer computes search quality metrics.

Files:

```txt
searcheval/eval/metrics.py
searcheval/eval/evaluator.py
searcheval/eval/failure_analysis.py
```

Metrics:

```txt
Precision@K
Recall@K
MRR@K
NDCG@K
```

Responsibilities:

- Evaluate ranked results against relevance labels.
- Aggregate metrics across queries.
- Produce per-query metrics.
- Identify weak queries for failure analysis.

---

### 4. Benchmark Layer

The benchmark layer connects datasets, search engines, evaluation, and latency tracking.

Files:

```txt
searcheval/benchmarks/runner.py
searcheval/benchmarks/store.py
searcheval/benchmarks/compare.py
```

Responsibilities:

- Run benchmarks end to end.
- Measure query latency.
- Aggregate search quality metrics.
- Save benchmark artifacts.
- Compare two saved benchmark runs.

Saved artifacts:

```txt
summary.json
metrics.json
per_query_metrics.json
latencies.json
report.md
comparison.json
```

---

### 5. Regression Detection Layer

The regression layer detects quality and latency regressions between benchmark runs.

Files:

```txt
searcheval/regression/detector.py
searcheval/regression/config.py
configs/regression.json
```

Responsibilities:

- Compare baseline and current metrics.
- Apply configurable thresholds.
- Detect regressions in quality metrics.
- Detect regressions in latency metrics.
- Produce pass/fail comparison results.

Example threshold configuration:

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

---

### 6. Reporting Layer

The reporting layer generates Markdown benchmark reports.

File:

```txt
searcheval/reports/markdown.py
```

Report sections:

```txt
Run Summary
Aggregate Metrics
Per-Query Metrics
Weak Query Analysis
Query Latency
Notes
```

The report is designed to make benchmark results readable for engineers, reviewers, and hiring teams.

---

### 7. CLI Layer

The CLI provides a local developer workflow.

File:

```txt
searcheval/cli.py
```

Main commands:

```bash
python -m searcheval.cli version
python -m searcheval.cli validate data/search_eval_small
python -m searcheval.cli run data/search_eval_small --engine tfidf --k 10
python -m searcheval.cli run data/search_eval_small --engine bm25 --k 10
python -m searcheval.cli compare <baseline_run> <current_run>
```

The CLI is useful for local benchmarking, regression checks, and reproducible experiments.

---

### 8. FastAPI Backend Layer

The API layer exposes benchmark workflows over HTTP.

File:

```txt
searcheval/api.py
```

Main endpoints:

```txt
GET  /health
GET  /engines
POST /datasets/validate
POST /benchmarks/run
GET  /benchmarks/runs
GET  /benchmarks/runs/{run_id}
GET  /benchmarks/runs/{run_id}/report
GET  /benchmarks/runs/{run_id}/metrics
GET  /benchmarks/runs/{run_id}/per-query
GET  /benchmarks/runs/{run_id}/latencies
POST /benchmarks/compare
```

The API makes the benchmark system usable as a backend service and prepares the project for a future dashboard or automation layer.

---

## Developer Workflow

Common commands are available through the Makefile.

```bash
make install
make lint
make format-check
make test
make validate
make run-tfidf
make run-bm25
make compare
make api
make api-smoke
```

---

## CI Workflow

GitHub Actions runs:

```bash
make lint
make format-check
make test
```

This ensures that every pushed change passes linting, formatting, and the full test suite.

---

## Design Goals

SearchEval Lab is designed around the following goals:

1. Reproducibility  
   Benchmark inputs, outputs, and configuration are saved in a structured way.

2. Extensibility  
   New search engines can be added through the search engine interface and factory.

3. Observability  
   The system records aggregate metrics, per-query metrics, weak queries, and latency data.

4. Regression Safety  
   Saved benchmark runs can be compared to detect quality or latency regressions.

5. Backend Readiness  
   The FastAPI layer exposes the benchmark system as a service.

6. Portfolio Clarity  
   The project demonstrates backend engineering, ML systems thinking, benchmarking, and evaluation infrastructure.

---

## Future Extensions

Planned extensions:

```txt
Hybrid search engine
Semantic search baseline
Docker support
Example reports committed to examples/reports
Dashboard for saved benchmark runs
More datasets
CI benchmark workflow
```
