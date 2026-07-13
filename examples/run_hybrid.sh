#!/usr/bin/env bash
set -euo pipefail

DATASET_PATH="${DATASET_PATH:-data/search_eval_small}"
K="${K:-10}"

echo "Validating dataset: ${DATASET_PATH}"
python -m searcheval.cli validate "${DATASET_PATH}"

echo
echo "Running Hybrid BM25 + TF-IDF benchmark"
python -m searcheval.cli run "${DATASET_PATH}" --engine hybrid --k "${K}"
