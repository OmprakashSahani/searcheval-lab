# SearchEval Lab Project Summary

## Overview

SearchEval Lab is a backend and ML systems project for evaluating search quality, benchmarking retrieval methods, tracking query latency, detecting regressions, and generating reproducible benchmark reports.

The project is designed around a realistic problem in AI search systems: how to measure whether a search or retrieval system is improving, regressing, or behaving inconsistently across benchmark runs.

It supports both a command-line workflow and a FastAPI backend workflow.

---

## Problem

Search and retrieval systems are often evaluated using multiple signals:

- Search quality
- Ranking relevance
- Per-query behavior
- Latency
- Regression over time
- Reproducibility of benchmark runs

A search method can appear good on average while still failing on specific queries. Similarly, a new implementation can improve quality while increasing latency, or reduce latency while hurting relevance.

SearchEval Lab addresses this by creating a structured system for running benchmarks, storing results, comparing runs, and inspecting failures.

---

## What the System Does

SearchEval Lab can:

- Load search evaluation datasets.
- Validate documents, queries, and relevance labels.
- Run retrieval baselines such as TF-IDF and BM25.
- Evaluate ranked results using Precision@K, Recall@K, MRR@K, and NDCG@K.
- Track query latency.
- Save benchmark artifacts.
- Generate Markdown benchmark reports.
- Compare two benchmark runs for regressions.
- Expose benchmark workflows through a CLI.
- Expose benchmark workflows through a FastAPI backend.
- Read saved benchmark artifacts through API endpoints.

---

## Key Features

### Dataset Validation

The dataset validator checks that benchmark inputs are well-formed before running evaluations.

It validates:

- Document IDs
- Query IDs
- Relevance labels
- Missing references
- Duplicate records

This helps prevent unreliable benchmark results caused by invalid input data.

---

### Search Baselines

The project currently supports:

- TF-IDF search
- BM25 search

Both engines use a shared search interface, making the system extensible for future retrieval methods such as hybrid search or semantic search.

---

### Evaluation Metrics

The evaluation engine computes common search-quality metrics:

- Precision@K
- Recall@K
- MRR@K
- NDCG@K

These metrics are computed per query and aggregated across the full benchmark run.

---

### Weak Query Analysis

SearchEval Lab identifies weak queries based on low ranking-quality scores.

This helps answer questions such as:

- Which queries are failing?
- Which queries need better retrieval behavior?
- Where does one search method underperform another?
- Which examples should be inspected manually?

---

### Benchmark Artifact Storage

Each benchmark run can save structured artifacts:

- summary.json
- metrics.json
- per_query_metrics.json
- latencies.json
- report.md

These artifacts make benchmark runs reproducible and inspectable after execution.

---

### Regression Detection

The regression system compares a baseline run against a current run using configurable thresholds.

It can detect regressions in:

- Precision@K
- Recall@K
- MRR@K
- NDCG@K
- Average latency
- Maximum latency

This is useful for performance-aware engineering workflows where search quality and latency must be protected over time.

---

### Markdown Reports

The report generator creates human-readable benchmark reports.

Reports include:

- Run summary
- Aggregate metrics
- Per-query metrics
- Weak query analysis
- Query latency
- Notes

This makes the system easier to review without manually inspecting JSON files.

---

### CLI Workflow

The CLI supports local developer workflows.

Example commands:

```bash
python -m searcheval.cli validate data/search_eval_small
python -m searcheval.cli run data/search_eval_small --engine tfidf --k 10
python -m searcheval.cli run data/search_eval_small --engine bm25 --k 10
python -m searcheval.cli compare <baseline_run> <current_run>
```

The CLI makes the project usable as a local benchmark tool.

---

### FastAPI Backend

The FastAPI backend exposes the benchmark system as an HTTP service.

Main API capabilities:

- Health checks
- List supported search engines
- Validate datasets
- Run benchmarks
- List saved benchmark runs
- Read saved benchmark summaries
- Read saved Markdown reports
- Read saved JSON artifacts
- Compare benchmark runs

This makes the project stronger for backend engineering and AI infrastructure roles.

---

## System Design

The system is organized into independent layers:

```txt
Dataset Layer
Search Engine Layer
Evaluation Layer
Benchmark Layer
Regression Layer
Reporting Layer
CLI Layer
FastAPI Layer
```

This design keeps the system modular and extensible.

For example:

- New search engines can be added without rewriting the benchmark runner.
- New metrics can be added without changing the API layer.
- New report formats can be added without changing search logic.
- The CLI and API reuse the same core benchmark code.

---

## Engineering Trade-Offs

### Why TF-IDF and BM25?

TF-IDF and BM25 are lightweight, interpretable baselines. They are useful starting points before adding semantic or hybrid retrieval.

They also keep the project easy to run locally without requiring large model downloads.

---

### Why Save JSON Artifacts?

Saved JSON artifacts make benchmark runs reproducible and easy to inspect.

They also allow later tools, dashboards, or APIs to read results without rerunning benchmarks.

---

### Why Include Latency?

Search quality alone is not enough. Real search systems must balance relevance and performance.

Latency tracking helps evaluate whether a retrieval method is practical for backend or production-like use.

---

### Why Add Regression Detection?

Benchmark results are most useful when they can detect change over time.

Regression detection makes SearchEval Lab closer to real engineering workflows where teams need to protect quality and performance after every change.

---

### Why Add a FastAPI Backend?

The API layer makes the system usable beyond the command line.

It prepares the project for:

- Dashboards
- Automation
- Internal evaluation services
- CI benchmark workflows
- Search quality monitoring tools

---

## Technical Stack

Core stack:

- Python
- Pydantic
- Scikit-learn
- Typer
- Rich
- FastAPI
- Uvicorn
- Pytest
- Ruff
- GitHub Actions

---

## Testing and Quality

The project includes tests for:

- Metrics
- Dataset loading and validation
- Search engines
- Benchmark runner
- Benchmark artifact storage
- Regression detection
- Markdown reports
- CLI workflows
- FastAPI endpoints
- Saved benchmark artifact access

The project also uses:

- Ruff linting
- Ruff formatting checks
- GitHub Actions CI

---

## Why This Project Matters

SearchEval Lab demonstrates practical engineering work across backend systems, ML systems, search evaluation, benchmarking, and reproducibility.

It shows the ability to:

- Build modular software systems.
- Design reusable interfaces.
- Evaluate search quality.
- Track performance metrics.
- Detect regressions.
- Build developer tooling.
- Build backend APIs.
- Write tests and CI workflows.
- Communicate system design clearly.

---

## Future Work

Planned extensions:

- Hybrid search engine
- Semantic search baseline
- Docker support
- Example benchmark reports committed to the repository
- Dashboard for saved benchmark runs
- CI benchmark workflow
- Larger evaluation datasets
- Query-level comparison reports

---

## Portfolio Positioning

SearchEval Lab is relevant for roles such as:

- Software Engineer Backend
- ML Systems Engineer
- AI Infrastructure Engineer
- Search Infrastructure Engineer
- Retrieval Evaluation Engineer
- Research Evals Engineer
- Performance Engineering roles

The project combines backend engineering, evaluation infrastructure, benchmarking, regression detection, and ML systems thinking in one cohesive system.
