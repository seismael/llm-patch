# Usage Guide

This guide covers how to install, configure, and use **llm-patch** in common scenarios — from quick local demos to production pipelines.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start (No GPU)](#quick-start-no-gpu)
- [Batch Processing](#batch-processing)
- [Live Watch Mode](#live-watch-mode)
- [Wiki-Based Pipelines](#wiki-based-pipelines)
- [Loading Adapters at Inference](#loading-adapters-at-inference)
- [Configuration](#configuration)
- [Using the Makefile](#using-the-makefile)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | Required |
| PyTorch | ≥ 2.1 | CUDA recommended for real weight generation |
| Poetry | ≥ 1.7 | Dependency management |
| T2L Checkpoint | — | Required for real (non-mock) generation |

### Install with Poetry

```bash
git clone https://github.com/your-org/llm-patch.git
cd llm-patch
poetry install
```

### Install with Dev Dependencies

```bash
poetry install --with dev
```

### Install Wiki Extras (Optional)

For robust YAML frontmatter parsing:

```bash
poetry install --extras wiki
```

---

## Quick Start (No GPU)

The fastest way to see llm-patch in action uses mock components — no GPU or T2L checkpoint required:

```bash
cd examples
python run_e2e.py --clean --aggregate
```

This will:

1. Copy sample ML papers from `raw/papers/` into a simulated `wiki/` directory
2. Add wiki-style frontmatter and create entity stub pages
3. Run the full pipeline with `MockWeightGenerator` and `MockAdapterRepository`
4. Report all generated adapter manifests

Expected output:

```
Phase 1: Simulating LLM Wiki Agent output...
  Created wiki/sources/attention-is-all-you-need.md
  Created wiki/sources/lora-low-rank-adaptation.md
  Created wiki/sources/gpt3-few-shot-learners.md
  Created wiki/entities/transformer.md
  ...
Phase 2: Running adapter generation pipeline...
  Compiled: attention-is-all-you-need → adapters/sources/attention-is-all-you-need
  Compiled: lora-low-rank-adaptation → adapters/sources/lora-low-rank-adaptation
  ...
Phase 3: Complete. Generated 6 adapter manifests.
```

---

## Batch Processing

Process all documents in a directory at once and exit:

### Using the Library API

```python
from llm_patch import (
    KnowledgeFusionOrchestrator,
    SakanaT2LGenerator,
    LocalSafetensorsRepository,
    MarkdownDirectoryWatcher,
    GeneratorConfig,
    StorageConfig,
    WatcherConfig,
)

watcher = MarkdownDirectoryWatcher(WatcherConfig(directory="./docs"))
generator = SakanaT2LGenerator(GeneratorConfig(checkpoint_dir="./models/t2l"))
repository = LocalSafetensorsRepository(StorageConfig(output_dir="./adapters"))

orchestrator = KnowledgeFusionOrchestrator(
    source=watcher, generator=generator, repository=repository
)

# Process everything and get manifests
manifests = orchestrator.compile_all()
for m in manifests:
    print(f"  {m.adapter_id}: rank={m.rank}, path={m.storage_uri}")
```

### Using the Example CLI

```bash
cd examples
python research_pipeline.py batch --wiki-dir ./wiki --output-dir ./adapters
```

With wikilink aggregation (follows `[[links]]` to enrich documents):

```bash
python research_pipeline.py batch --wiki-dir ./wiki --output-dir ./adapters --aggregate
```

---

## Live Watch Mode

Monitor a directory for changes. Every new or modified Markdown file automatically triggers adapter generation:

### Using the Library API

```python
orchestrator = KnowledgeFusionOrchestrator(
    source=watcher, generator=generator, repository=repository
)

# Context manager handles start/stop
with orchestrator:
    # Watcher is running — edit files in ./docs and adapters are generated
    # Press Ctrl-C to stop
    import time
    while True:
        time.sleep(1)
```

### Using the Example CLI

```bash
cd examples
python research_pipeline.py watch --wiki-dir ./wiki
```

Add or modify Markdown files in the watched directory while the watcher is running. Press `Ctrl-C` to stop.

---

## Wiki-Based Pipelines

`WikiKnowledgeSource` provides enhanced features for structured wiki directories:

### Features

- **YAML frontmatter parsing** — Metadata like `title`, `authors`, `tags` is extracted into `DocumentContext.metadata`
- **Wikilink extraction** — `[[Entity Name]]` and `[[Entity Name|display text]]` references are captured
- **Cross-page aggregation** — Optionally follow wikilinks to concatenate linked entity/concept pages into enriched documents

### Wiki Directory Structure

```
wiki/
├── sources/              # One page per ingested document
│   ├── api-v1.md
│   └── api-v2.md
├── entities/             # Auto-extracted entity pages
│   ├── authentication.md
│   └── rate-limiting.md
└── concepts/             # Concept pages
    └── rest-api.md
```

### Frontmatter Format

```yaml
---
title: API v2 Documentation
authors: Platform Team
version: 2.0
tags: api, rest, authentication
---

# API v2

Content here...
```

### Using WikiKnowledgeSource

```python
from llm_patch import WikiKnowledgeSource, WatcherConfig

source = WikiKnowledgeSource(
    WatcherConfig(directory="./wiki"),
    aggregate=True,  # Follow [[wikilinks]] for richer context
)

# Scan all existing pages
docs = source.scan_existing()
for doc in docs:
    print(f"{doc.document_id}: {doc.metadata.get('title', 'untitled')}")
    print(f"  Wikilinks: {doc.metadata.get('wikilinks', [])}")
```

---

## Loading Adapters at Inference

Generated adapters are PEFT-compatible and can be loaded with a single line of code:

### Basic Loading

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

# Load adapter
model = PeftModel.from_pretrained(base_model, "./adapters/api-v2")

# Generate
inputs = tokenizer("How do I authenticate with API v2?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Dynamic Adapter Swapping

```python
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")

# Load first adapter
model = PeftModel.from_pretrained(base_model, "./adapters/api-v1", adapter_name="v1")

# Load second adapter
model.load_adapter("./adapters/api-v2", adapter_name="v2")

# Switch between them per request
model.set_adapter("v1")
# ... generate for v1 customer ...

model.set_adapter("v2")
# ... generate for v2 customer ...
```

### Merging Multiple Adapters

```python
# Combine multiple document adapters into a single domain adapter
model.add_weighted_adapter(
    adapters=["api-v2", "auth-guide", "rate-limits"],
    weights=[1.0, 0.8, 0.5],
    adapter_name="api-v2-complete",
)
model.set_adapter("api-v2-complete")
```

### Adapter Validation

Compare base model output vs. adapter-enhanced output:

```bash
cd examples
python validate_adapter.py \
    --adapter-dir ./adapters/attention-paper \
    --base-model google/gemma-2-2b-it
```

This requires a CUDA-capable GPU with `torch`, `transformers`, and `peft` installed.

---

## Configuration

### GeneratorConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoint_dir` | `str` | — | Path to the T2L checkpoint directory containing `hypermod.pt`, `args.yaml`, `adapter_config.json` |
| `device` | `str` | `"cuda"` | PyTorch device (`"cuda"`, `"cpu"`, `"mps"`) |

### WatcherConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `directory` | `str` | — | Directory to monitor for documents |
| `patterns` | `list[str]` | `["*.md"]` | Glob patterns to match files |
| `recursive` | `bool` | `True` | Whether to watch subdirectories |
| `debounce_seconds` | `float` | `0.5` | Debounce interval to prevent duplicate callbacks on rapid saves |

### StorageConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `str` | — | Directory where adapters are written |

---

## Using the Makefile

The project includes a Makefile for common development tasks:

```bash
make help             # Show all available commands
make install-dev      # Install dependencies + pre-commit hooks
make test             # Run all tests with coverage
make test-unit        # Run unit tests only
make test-fast        # Quick test run, stop on first failure
make lint             # Run ruff linter
make format           # Auto-format code
make typecheck        # Run mypy type checker
make check            # Run lint + typecheck + test (full CI check)
make demo             # Run end-to-end demo
make clean            # Remove build artifacts and caches
make build            # Build distribution packages
```

---

## Running Tests

### All Tests with Coverage

```bash
make test
# or
poetry run pytest --cov=llm_patch --cov-report=term-missing
```

### Unit Tests Only

```bash
make test-unit
# or
poetry run pytest tests/unit/ -v
```

### Integration Tests Only

```bash
make test-integration
# or
poetry run pytest tests/integration/ -v -m integration
```

### Quick Run (Stop on First Failure)

```bash
make test-fast
# or
poetry run pytest -x -q
```

---

## Troubleshooting

### Common Issues

**"No module named 'hyper_llm_modulator'"**

The `SakanaT2LGenerator` requires the Sakana AI hypernetwork library. Install it from the [text-to-lora repository](https://github.com/SakanaAI/text-to-lora). For testing without GPU or the Sakana library, use the mock components in `examples/`.

**"CUDA out of memory"**

The T2L hypernetwork requires GPU memory for inference. Try:
- Setting `device="cpu"` in `GeneratorConfig` (slower but works without GPU)
- Using a smaller base model checkpoint
- Closing other GPU-intensive processes

**"FileNotFoundError: checkpoint directory not found"**

Ensure `checkpoint_dir` in `GeneratorConfig` points to a directory containing `hypermod.pt`, `args.yaml`, and `adapter_config.json` from the T2L checkpoint.

**Adapters not regenerating on file changes**

- Check that the file matches the configured `patterns` (default: `["*.md"]`)
- Ensure the watcher is running (`orchestrator.start()` or `with orchestrator:`)
- Check the `debounce_seconds` setting — rapid saves within the debounce window are collapsed into one event

**Type checking errors with torch**

Add the following to your `mypy` configuration to ignore missing stubs for PyTorch and related libraries:

```toml
[[tool.mypy.overrides]]
module = ["torch.*", "safetensors.*", "watchdog.*", "hyper_llm_modulator.*"]
ignore_missing_imports = true
```
