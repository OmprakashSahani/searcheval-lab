#!/usr/bin/env bash

set -euo pipefail

echo "Running SearchEval Lab TF-IDF benchmark..."

python -m searcheval.cli validate data/search_eval_small

python -m searcheval.cli run \
  data/search_eval_small \
  --engine tfidf \
  --k 10

echo "TF-IDF benchmark completed."
