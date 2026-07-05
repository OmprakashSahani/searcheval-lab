.PHONY: install test validate run-tfidf run-bm25 compare clean

install:
	python -m pip install -e ".[dev]"

test:
	pytest

validate:
	python -m searcheval.cli validate data/search_eval_small

run-tfidf:
	python -m searcheval.cli run data/search_eval_small --engine tfidf --k 10

run-bm25:
	python -m searcheval.cli run data/search_eval_small --engine bm25 --k 10

compare:
	./examples/compare_tfidf_bm25.sh

clean:
	rm -rf runs/
	rm -rf .pytest_cache/
	rm -rf **/__pycache__/
