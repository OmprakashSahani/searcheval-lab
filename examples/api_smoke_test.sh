#!/usr/bin/env bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
DATASET_PATH="${DATASET_PATH:-data/search_eval_small}"
RUNS_DIR="${RUNS_DIR:-runs/api}"

echo "Using API URL: ${API_URL}"
echo

echo "1. Health check"
curl -sS "${API_URL}/health"
echo
echo

echo "2. Supported search engines"
curl -sS "${API_URL}/engines"
echo
echo

echo "3. Validate dataset"
curl -sS -X POST "${API_URL}/datasets/validate" \
  -H "Content-Type: application/json" \
  -d "{\"dataset_path\": \"${DATASET_PATH}\"}"
echo
echo

echo "4. Run TF-IDF benchmark without saving artifacts"
curl -sS -X POST "${API_URL}/benchmarks/run" \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_path\": \"${DATASET_PATH}\",
    \"engine\": \"tfidf\",
    \"k\": 10,
    \"save_artifacts\": false
  }"
echo
echo

echo "5. Run BM25 benchmark and save artifacts"
curl -sS -X POST "${API_URL}/benchmarks/run" \
  -H "Content-Type: application/json" \
  -d "{
    \"dataset_path\": \"${DATASET_PATH}\",
    \"engine\": \"bm25\",
    \"k\": 10,
    \"save_artifacts\": true,
    \"runs_dir\": \"${RUNS_DIR}\"
  }"
echo
echo

echo "6. List saved benchmark runs"
curl -sS "${API_URL}/benchmarks/runs?runs_dir=${RUNS_DIR}"
echo
echo

echo "API smoke test completed."
