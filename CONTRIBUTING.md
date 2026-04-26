# Contributing to llm-patch

Thank you for your interest in contributing to llm-patch! This project is open to everyone — whether you're fixing a typo, adding a storage backend, or proposing a new feature.

---

## Code of Conduct

Be respectful, constructive, and inclusive. We welcome contributors of all experience levels and backgrounds.

---

## How to Contribute

### Reporting Bugs

1. Search [existing issues](https://github.com/your-org/llm-patch/issues) to avoid duplicates
2. Open a new issue with:
   - A clear title and description
   - Steps to reproduce
   - Expected vs. actual behavior
   - Python version, OS, and PyTorch version

### Suggesting Features

Open an issue with the `enhancement` label. Include:
- The problem you're trying to solve
- Your proposed solution
- Alternative approaches you've considered

### Submitting Code

1. **Fork** the repository
2. **Clone** your fork and install dev dependencies:
   ```bash
   git clone https://github.com/your-username/llm-patch.git
   cd llm-patch
   make install-dev
   ```
3. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/my-new-source
   ```
4. **Make your changes** with tests
5. **Run checks** before committing:
   ```bash
   make check
   ```
6. **Commit** with a clear message:
   ```bash
   git commit -m "feat: add ConfluenceKnowledgeSource"
   ```
7. **Push** and open a pull request

---

## Development Setup

### Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/) ≥ 0.4

### Installation

```bash
# Install all dependencies + pre-commit hooks
uv sync
uv run pre-commit install
```

### Running Checks

```bash
make check          # Runs lint + typecheck + tests
make test           # Tests with coverage
make lint           # Ruff linter
make typecheck      # Mypy strict mode
make format         # Auto-format code
```

---

## Code Standards

### Style

- **Formatter/Linter:** [Ruff](https://github.com/astral-sh/ruff) — configured in `pyproject.toml`
- **Line length:** 100 characters
- **Quote style:** Double quotes
- **Imports:** Sorted by Ruff (isort-compatible)

### Type Hints

- All public functions and methods must have complete type annotations
- The project uses `mypy --strict` — your code must pass strict type checking
- The `py.typed` marker is present for downstream PEP 561 compliance

### Testing

- All new features must include tests
- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/` and should be marked with `@pytest.mark.integration`
- Use `pytest-mock` for mocking — prefer dependency injection over patching
- Target test coverage above 90% for new code

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add S3AdapterRepository
fix: handle empty documents in generator
docs: update USAGE.md with watch mode examples
test: add unit tests for WikiDocumentAggregator
refactor: extract frontmatter parsing into utility
```

---

## Areas for Contribution

Here are some areas where contributions are especially welcome:

### Plugins (`pip install`-able add-ons)

`llm-patch` 0.3.0 ships a discovery mechanism (env var + entry point)
so plugins live outside the engine. See
[docs/EXTENDING.md](docs/EXTENDING.md) for the contract and
[docs/adr/0008-plugin-discovery.md](docs/adr/0008-plugin-discovery.md)
for the design rationale. To announce a plugin, open an issue using
the **New source plugin** or **New registry-client plugin** template.

### New Knowledge Sources (`IDataSource`)

- Confluence / Notion integration
- Database table watcher (PostgreSQL, SQLite)
- RSS/Atom feed ingestion
- Git repository diff watcher

### New Storage Backends (`IAdapterRepository`)

- AWS S3
- Google Cloud Storage
- Azure Blob Storage
- HuggingFace Hub (push/pull)

### New Weight Generators (`IWeightGenerator`)

- Alternative hypernetwork architectures
- Distillation-based generators
- Adapter quality scoring and validation

### Infrastructure

- REST API server for on-demand generation
- Docker / Docker Compose setup
- CI/CD pipeline templates (GitHub Actions, GitLab CI)
- Web UI dashboard for monitoring

### Documentation

- Additional use case tutorials
- Video walkthroughs
- Translations
- API reference generation (Sphinx / MkDocs)

---

## Pull Request Checklist

Before submitting a PR, ensure:

- [ ] Code follows the project's style (run `make format`)
- [ ] All checks pass (run `make check`)
- [ ] New code has test coverage
- [ ] Public APIs have type annotations
- [ ] Documentation is updated if applicable
- [ ] Commit messages follow Conventional Commits
- [ ] PR description explains the change and links to related issues

---

## License

By contributing to llm-patch, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE). This means your contributions are free to use, modify, and distribute by anyone, for any purpose, including commercial use.

---

## Questions?

If you're unsure about anything, open an issue or start a discussion. We're happy to help you get started.
