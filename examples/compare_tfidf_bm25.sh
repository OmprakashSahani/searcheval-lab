#!/usr/bin/env bash

set -euo pipefail

RUNS_DIR="runs/examples"

echo "Running SearchEval Lab TF-IDF vs BM25 comparison..."

python -m searcheval.cli validate data/search_eval_small

python -m searcheval.cli run \
  data/search_eval_small \
  --engine tfidf \
  --k 10 \
  --runs-dir "$RUNS_DIR"

TFIDF_RUN="$(ls -td "$RUNS_DIR"/*_tfidf | head -n 1)"

python -m searcheval.cli run \
  data/search_eval_small \
  --engine bm25 \
  --k 10 \
  --runs-dir "$RUNS_DIR"

BM25_RUN="$(ls -td "$RUNS_DIR"/*_bm25 | head -n 1)"

echo "Comparing runs..."
echo "Baseline: $TFIDF_RUN"
echo "Current:  $BM25_RUN"

python -m searcheval.cli compare \
  "$TFIDF_RUN" \
  "$BM25_RUN" \
  --config configs/regression.json

echo "TF-IDF vs BM25 comparison completed."
