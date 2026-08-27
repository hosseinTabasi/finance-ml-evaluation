.PHONY: test lint toy figures install

PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

toy:
	$(PYTHON) -m finmlcv.experiments.synthetic_leakage
	$(PYTHON) -m finmlcv.experiments.run_benchmark
	$(PYTHON) scripts/render_figures.py

figures:
	$(PYTHON) scripts/render_figures.py
