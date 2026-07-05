#!/usr/bin/env bash

set -euo pipefail

echo "Running SearchEval Lab BM25 benchmark..."

python -m searcheval.cli validate data/search_eval_small

python -m searcheval.cli run \
  data/search_eval_small \
  --engine bm25 \
  --k 10

echo "BM25 benchmark completed."
