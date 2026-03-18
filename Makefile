.PHONY: install test lint format typecheck train serve deploy clean

install:
	uv sync --extra dev
	uv run pre-commit install

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check .
	uv run black --check .

format:
	uv run ruff check --fix .
	uv run black .

typecheck:
	uv run mypy .

train:
	uv run python training/train.py

serve:
	uv run uvicorn serving.api.main:app --reload --port 8080

deploy:
	./scripts/deploy.sh

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/