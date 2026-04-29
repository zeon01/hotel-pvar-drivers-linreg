.PHONY: help setup data features train evaluate report lint test clean all docker-build docker-run

help:
	@echo "Targets:"
	@echo "  setup         - uv sync + pre-commit install"
	@echo "  data          - generate synthetic dataset (default 200k rows)"
	@echo "  features      - feature engineering"
	@echo "  train         - fit OLS (HC3 + cluster-robust) and a parallel sklearn LinearRegression"
	@echo "  evaluate      - residual diagnostics + held-out R²"
	@echo "  report        - render figures into reports/figures"
	@echo "  lint          - ruff check + ruff format --check"
	@echo "  test          - pytest -q"
	@echo "  clean         - remove interim/processed/figures"
	@echo "  all           - data -> features -> train -> evaluate -> report"
	@echo "  docker-build  - build the Docker image"
	@echo "  docker-run    - run 'make all' inside the container"

setup:
	uv sync
	uv run pre-commit install

data:
	uv run python scripts/generate_data.py --rows 200000

features:
	uv run python -c "from pvar_linreg.features import build_feature_frame; build_feature_frame()"

train:
	uv run python -m pvar_linreg.modeling.train

evaluate:
	uv run python -m pvar_linreg.modeling.evaluate

report:
	uv run python scripts/make_dataset.py --report

lint:
	uv run ruff check .
	uv run ruff format --check .

test:
	uv run pytest

clean:
	rm -rf data/interim/* data/processed/* reports/figures/* docs/figures/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +

all: data features train evaluate report

docker-build:
	docker build -t pvar-linreg:latest .

docker-run:
	docker run --rm pvar-linreg:latest make all
