.PHONY: install test lint format-check validate run-tfidf run-bm25 compare api clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .

format-check:
	ruff format --check .

validate:
	python -m searcheval.cli validate data/search_eval_small

run-tfidf:
	python -m searcheval.cli run data/search_eval_small --engine tfidf --k 10

run-bm25:
	python -m searcheval.cli run data/search_eval_small --engine bm25 --k 10

compare:
	./examples/compare_tfidf_bm25.sh

api:
	uvicorn searcheval.api:api_app --host 0.0.0.0 --port 8000 --reload

clean:
	rm -rf runs/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
