# SearchEval Lab — System Architecture

## Purpose

SearchEval Lab is an evaluation and benchmarking system for search and retrieval pipelines.

The goal is to measure how well different search methods retrieve relevant documents for a set of benchmark queries.

This project is designed to show strong skills in:

- ML systems
- Search evaluation
- Backend engineering
- Benchmarking
- Reproducibility
- Regression detection
- Performance analysis

---

## Core Idea

SearchEval Lab should answer these questions:

1. Are relevant documents appearing in the top search results?
2. Which queries are performing well or poorly?
3. Did a new search method improve quality?
4. Did a code change create a search quality regression?
5. What is the tradeoff between retrieval quality and latency?

The project is not just a search demo.

The main focus is the evaluation infrastructure around search systems.

---

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
Regression Detection
  ↓
Benchmark Report
```

---

## High-Level Architecture

```text
                    ┌────────────────────────┐
                    │      Dataset Layer      │
                    │ docs / queries / qrels  │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │   Dataset Validator     │
                    │ schema + consistency    │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Search Interface     │
                    │ common retrieval API    │
                    └───────────┬────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ Lexical Search │ │ Vector Search  │ │ Hybrid Search  │
     │ TF-IDF / BM25  │ │ Embeddings     │ │ Lexical+Vector │
     └────────┬───────┘ └────────┬───────┘ └────────┬───────┘
              └─────────────────┼─────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │    Evaluation Engine    │
                    │ Recall / MRR / NDCG     │
                    └───────────┬────────────┘
                                │
                                ▼
                    ┌────────────────────────┐
                    │    Benchmark Runner     │
                    │ configs / runs / timing │
                    └───────────┬────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
     ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
     │ Benchmark Store│ │ Regression     │ │ Report         │
     │ run artifacts  │ │ Detector       │ │ Generator      │
     └────────────────┘ └────────────────┘ └────────────────┘
```

---

## Main Components

### 1. Dataset Layer

The dataset layer stores benchmark data.

The MVP dataset will use three files:

```text
documents.jsonl
queries.jsonl
qrels.jsonl
```

### documents.jsonl

Stores searchable documents.

Example:

```json
{"doc_id": "doc_001", "title": "Transformer Architecture", "text": "Transformers use self-attention mechanisms for sequence modeling."}
```

### queries.jsonl

Stores benchmark queries.

Example:

```json
{"query_id": "q_001", "query": "how do transformers use attention"}
```

### qrels.jsonl

Stores relevance labels.

Example:

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

---

### 2. Dataset Validator

The validator checks whether the dataset is usable before running benchmarks.

It should detect:

- Missing document IDs
- Missing query IDs
- Duplicate document IDs
- Duplicate query IDs
- Empty document text
- Empty query text
- Invalid relevance scores
- Relevance labels pointing to missing documents
- Relevance labels pointing to missing queries

This makes benchmark results more reliable and reproducible.

---

### 3. Search Interface

All search methods should follow one common interface.

Input:

```text
query
top_k
```

Output:

```json
{
  "query_id": "q_001",
  "results": [
    {
      "doc_id": "doc_001",
      "score": 0.92,
      "rank": 1
    },
    {
      "doc_id": "doc_005",
      "score": 0.74,
      "rank": 2
    }
  ]
}
```

This allows different search methods to be evaluated fairly.

---

### 4. Lexical Search

The first search baseline will be lexical search.

Version 1 can use:

- TF-IDF
- BM25

Lexical search is useful because it is simple, explainable, and strong for keyword matching.

---

### 5. Vector Search

Vector search will be added after the MVP.

It will use embeddings to retrieve semantically similar documents.

Example:

```text
Query: "how to scale model training across GPUs"

Could match:
"Distributed training improves scalability by splitting computation across multiple devices."
```

---

### 6. Hybrid Search

Hybrid search combines lexical and vector search.

Example:

```text
hybrid_score = alpha * lexical_score + (1 - alpha) * vector_score
```

This is useful because many real search systems combine keyword and semantic signals.

---

### 7. Evaluation Engine

The evaluation engine compares retrieved results against relevance labels.

The MVP should support:

- Precision@K
- Recall@K
- MRR@K
- NDCG@K

### Precision@K

Measures how many top-K results are relevant.

### Recall@K

Measures how many relevant documents were retrieved in the top-K results.

### MRR@K

Measures how early the first relevant result appears.

### NDCG@K

Measures ranking quality using graded relevance scores.

NDCG is important because it rewards highly relevant documents appearing near the top.

---

### 8. Benchmark Runner

The benchmark runner coordinates the full evaluation process.

It should:

1. Load the dataset
2. Validate the dataset
3. Initialize the selected search method
4. Run all benchmark queries
5. Collect top-K results
6. Measure query latency
7. Compute evaluation metrics
8. Save benchmark artifacts

A benchmark run should produce:

```text
runs/run_001/
  config.json
  metrics.json
  per_query_metrics.json
  results.json
```

---

### 9. Benchmark Store

The benchmark store saves outputs from each benchmark run.

Each run should include:

- Run ID
- Timestamp
- Dataset path
- Search method
- Top-K value
- Search configuration
- Aggregate metrics
- Per-query metrics
- Raw retrieved results
- Latency statistics

This makes experiments reproducible and comparable.

---

### 10. Regression Detector

The regression detector compares a current run against a baseline run.

Example:

```text
Baseline NDCG@10: 0.740
Current NDCG@10:  0.710
Delta:            -4.05%

Status: FAIL
Reason: NDCG@10 dropped beyond the allowed threshold.
```

Regression detection should support configurable thresholds.

Example:

```yaml
regression_thresholds:
  recall_at_10: -0.03
  ndcg_at_10: -0.02
  mrr_at_10: -0.02
  latency_p95_ms: 0.20
```

This turns SearchEval Lab into a quality gate for search systems.

---

### 11. Report Generator

The report generator creates benchmark summaries.

The first version should generate Markdown reports.

A report should include:

- Run summary
- Dataset summary
- Search method configuration
- Aggregate metrics
- Per-query metrics
- Weak queries
- Regression analysis
- Latency summary
- Recommendations

---

## CLI Architecture

The CLI should support the main project workflow.

Example commands:

```bash
searcheval validate data/search_eval_small
```

```bash
searcheval run --dataset data/search_eval_small --engine tfidf --k 10
```

```bash
searcheval compare --baseline runs/run_001 --current runs/run_002
```

```bash
searcheval report --run runs/run_002 --format markdown
```

```bash
searcheval check --baseline runs/run_001 --current runs/run_002 --thresholds configs/regression.yaml
```

---

## Planned API Architecture

The FastAPI backend will be added after the CLI MVP.

Planned endpoints:

```text
POST /datasets/validate
POST /benchmarks/run
GET  /benchmarks/{run_id}
POST /benchmarks/compare
GET  /reports/{run_id}
POST /search
```

The API layer should reuse the same internal modules as the CLI.

This avoids duplicate logic.

---

## MVP Architecture Summary

The MVP will focus on this flow:

```text
Dataset validation
  ↓
Lexical search baseline
  ↓
Ranking metrics
  ↓
Benchmark run storage
  ↓
Regression comparison
  ↓
Markdown report
```

The most important design decision is to keep evaluation independent from retrieval.

That means the evaluation engine should work with TF-IDF, BM25, vector search, hybrid search, or any future retrieval method.

---

## Architecture Status

Architecture defined.

Next step: create the initial project folder structure.
