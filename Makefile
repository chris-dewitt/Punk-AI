.PHONY: help test lint fmt demo clean

help:
	@echo "make test   - run the test suite (no network, no GPU)"
	@echo "make lint   - ruff check"
	@echo "make fmt    - ruff format"
	@echo "make demo   - run the zero-dependency end-to-end demo"

test:
	python3 -m pytest -q

lint:
	python3 -m ruff check src tests examples

fmt:
	python3 -m ruff format src tests examples

demo:
	python3 examples/quickstart.py

clean:
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
