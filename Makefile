.PHONY: install dev test lint format run clean

install:
	uv pip install -e .

dev:
	uv pip install -e ".[dev,server]"

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	mypy src/videoclaw/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

run:
	claw generate "A 10-second demo video"

clean:
	rm -rf dist/ build/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
