# SearchEval Lab

Evaluation and benchmarking infrastructure for search and retrieval systems.

## Overview

SearchEval Lab is a portfolio-grade ML systems and backend infrastructure project for measuring search and retrieval quality.

The project focuses on evaluating how well different retrieval systems return relevant results for a given set of queries. It supports reproducible datasets, ranking metrics, benchmark runs, regression detection, and automated reports.

This is not just a search demo. The goal is to build the evaluation layer that helps search teams measure whether a retrieval system is improving or getting worse over time.

## Project Positioning

SearchEval Lab is designed for roles related to:

* ML Systems Engineering
* Software Engineering
* Backend Engineering
* Search Infrastructure
* AI Infrastructure
* Research Evals
* Machine Learning Engineering

## Core Problem

Search systems need reliable ways to answer questions such as:

* Did the new retrieval method improve search quality?
* Are relevant documents appearing in the top results?
* Did a code change cause a search quality regression?
* Which queries are failing?
* How do different retrieval methods compare?
* What is the tradeoff between quality and latency?

SearchEval Lab answers these questions through benchmark-driven evaluation.

## MVP Scope

The first version of SearchEval Lab will include:

1. A simple dataset format for documents, queries, and relevance labels
2. Dataset validation
3. Baseline lexical search using TF-IDF or BM25
4. Ranking metrics:

   * Recall@K
   * Precision@K
   * MRR@K
   * NDCG@K
5. Benchmark runner
6. Benchmark result storage
7. Regression detection between benchmark runs
8. Markdown benchmark reports
9. CLI interface for local usage

## Future Extensions

After the MVP is complete, the project can be extended with:

* Embedding-based semantic search
* Hybrid lexical + vector search
* FastAPI backend service
* HTML benchmark reports
* GitHub Actions quality gates
* Query failure analysis
* Latency benchmarking
* Dataset versioning
* Search experiment comparison dashboard

## Non-Goals for Version 1

The first version will not focus on:

* Training large models
* Building a full frontend dashboard
* Distributed deployment
* Kubernetes
* Complex authentication
* Large-scale production indexing
* Fine-tuning embedding models

The priority is to build a clean, reliable, and well-tested search evaluation system.

## High-Level Workflow

```text
Dataset
  ↓
Search Method
  ↓
Top-K Results
  ↓
Evaluation Metrics
  ↓
Benchmark Run
  ↓
Regression Check
  ↓
Report
```

## Example Use Case

A developer changes the search algorithm from lexical search to hybrid search.

SearchEval Lab can run both methods on the same benchmark dataset, compare ranking metrics, detect regressions, and generate a report showing whether the new method improved search quality.

## Current Status

Project initialized. MVP scope is being defined.

## Author

Omprakash Sahani

ML Systems Engineer | Software Engineer | Distributed Systems
