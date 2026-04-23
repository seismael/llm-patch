.PHONY: help sync test test-engine test-shared test-wiki-agent lint typecheck check check-layering check-coverage build clean adr

# ─────────────────────────────────────────────────────────────
#  llm-patch monorepo — workspace fan-out commands
#  Per-project Makefiles live in projects/<name>/Makefile.
# ─────────────────────────────────────────────────────────────

UV := uv

help: ## Show this help message
	@echo.
	@echo  llm-patch monorepo
	@echo  ──────────────────
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

sync: ## Sync the entire workspace
	$(UV) sync

# ── Tests ───────────────────────────────────────────────────

test-engine: ## Run engine tests
	cd projects/llm-patch && $(UV) run pytest -q

test-shared: ## Run shared-utils smoke tests
	cd projects/shared-utils && $(UV) run pytest -q

test-wiki-agent: ## Run wiki-agent smoke tests
	cd projects/wiki-agent && $(UV) run pytest -q

test: test-engine test-shared test-wiki-agent ## Run tests for every workspace member

# ── Quality ─────────────────────────────────────────────────

lint: ## Lint the entire workspace
	$(UV) run ruff check .

typecheck: ## Mypy strict for every project
	cd projects/llm-patch && $(UV) run mypy src
	cd projects/shared-utils && $(UV) run mypy src
	cd projects/wiki-agent && $(UV) run mypy src

check-layering: ## Architectural fitness check (ADR-0002)
	$(UV) run python tools/check_layering.py

check-coverage: ## Enforce workspace coverage thresholds
	cd projects/llm-patch && $(UV) run pytest --cov=llm_patch --cov-branch --cov-report=xml:coverage.xml -q
	$(UV) run python tools/check_coverage.py projects/llm-patch/coverage.xml

check: lint typecheck check-layering test check-coverage ## Lint + typecheck + layering + tests

# ── Build ───────────────────────────────────────────────────

build: ## Build distributions for every project
	cd projects/llm-patch && $(UV) build
	cd projects/shared-utils && $(UV) build
	cd projects/wiki-agent && $(UV) build

# ── ADR helper ──────────────────────────────────────────────

adr: ## Scaffold a new ADR: make adr title="my-decision"
	@test -n "$(title)" || (echo "Usage: make adr title=\"my-decision\""; exit 2)
	@next=$$(printf '%04d' $$(($$(ls docs/adr/[0-9]*.md 2>/dev/null | wc -l) + 1))); \
		cp docs/adr/0000-template.md "docs/adr/$$next-$(title).md"; \
		echo "Created docs/adr/$$next-$(title).md"

# ── Cleanup ─────────────────────────────────────────────────

clean: ## Remove caches and build artifacts
	rm -rf dist build .pytest_cache .pytest_tmp .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
