.PHONY: help install install-dev test test-unit test-integration lint format typecheck check clean build docs demo

# ─────────────────────────────────────────────────────────────
#  llm-patch — Development Commands (uv)
# ─────────────────────────────────────────────────────────────

PYTHON     := python
UV         := uv
PYTEST     := $(UV) run pytest
RUFF       := $(UV) run ruff
MYPY       := $(UV) run mypy
PRE_COMMIT := $(UV) run pre-commit

SRC_DIRS   := src/ tests/ examples/
SRC_CODE   := src/ tests/

# ─────────────────────────────────────────────────────────────
#  Help
# ─────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo.
	@echo  llm-patch Development Commands
	@echo  ──────────────────────────────
	@echo.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo.

# ─────────────────────────────────────────────────────────────
#  Installation
# ─────────────────────────────────────────────────────────────

install: ## Install production dependencies
	$(UV) sync --no-dev

install-dev: ## Install all dependencies (production + dev)
	$(UV) sync
	$(PRE_COMMIT) install

# ─────────────────────────────────────────────────────────────
#  Testing
# ─────────────────────────────────────────────────────────────

test: ## Run all tests with coverage
	$(PYTEST) --cov=llm_patch --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTEST) tests/unit/ -v

test-integration: ## Run integration tests only
	$(PYTEST) tests/integration/ -v -m integration

test-fast: ## Run tests without coverage (faster)
	$(PYTEST) -x -q

# ─────────────────────────────────────────────────────────────
#  Code Quality
# ─────────────────────────────────────────────────────────────

lint: ## Run linter (ruff check)
	$(RUFF) check $(SRC_CODE)

format: ## Auto-format code (ruff format)
	$(RUFF) format $(SRC_CODE)
	$(RUFF) check --fix $(SRC_CODE)

typecheck: ## Run static type checker (mypy)
	$(MYPY) src/

check: lint typecheck test ## Run all checks (lint + typecheck + test)

# ─────────────────────────────────────────────────────────────
#  Build & Release
# ─────────────────────────────────────────────────────────────

build: clean ## Build distribution packages
	$(UV) build

publish-test: build ## Publish to TestPyPI
	$(UV) publish --index testpypi

publish: build ## Publish to PyPI
	$(UV) publish

# ─────────────────────────────────────────────────────────────
#  Demo & Examples
# ─────────────────────────────────────────────────────────────

demo: ## Run the end-to-end demo (no GPU required)
	cd examples && $(PYTHON) run_e2e.py --clean --aggregate

demo-batch: ## Run batch mode on example wiki
	cd examples && $(PYTHON) research_pipeline.py batch --wiki-dir wiki/ --aggregate

demo-watch: ## Run watch mode on example wiki (Ctrl-C to stop)
	cd examples && $(PYTHON) research_pipeline.py watch --wiki-dir wiki/

# ─────────────────────────────────────────────────────────────
#  Cleanup
# ─────────────────────────────────────────────────────────────

clean: ## Remove build artifacts and caches
	rm -rf dist/ build/ *.egg-info
	rm -rf .pytest_cache .pytest_tmp .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-adapters: ## Remove generated adapters from examples
	rm -rf examples/adapters/ examples/wiki/

clean-all: clean clean-adapters ## Remove everything (build artifacts + adapters)
