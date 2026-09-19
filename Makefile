# Makefile for ORBIT-X

.PHONY: setup install deps test lint typecheck format demo benchmark

setup: install deps

install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt

deps:
	@echo "Installing development tools..."
	pip install -e .[dev]

test:
	@echo "Running unit and integration tests..."
	pytest tests/

lint:
	@echo "Running linter (ruff)..."
	ruff check .

typecheck:
	@echo "Running mypy..."
	mypy .

format:
	@echo "Running code formatter (ruff --fix)..."
	ruff check --fix .

demo:
	@echo "Running deterministic demo..."
	python -m orbit_x.demo

benchmark:
	@echo "Running full benchmark suite..."
	python -m orbit_x.benchmark
