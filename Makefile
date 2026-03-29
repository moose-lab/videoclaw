.PHONY: install dev test lint format run clean doctor

install:
	uv pip install -e .
	@echo ""
	@echo "Installed! Run with:"
	@echo "  uv run claw --help"
	@echo "  # or activate venv first: source .venv/bin/activate && claw --help"

dev:
	uv pip install -e ".[dev,server]"

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/
	uv run mypy src/videoclaw/

format:
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

run:
	uv run claw generate "A 10-second demo video"

doctor:
	uv run claw doctor

clean:
	rm -rf dist/ build/ *.egg-info .mypy_cache .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
