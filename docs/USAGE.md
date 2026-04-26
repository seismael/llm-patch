# Usage Guide

This guide covers how to install, configure, and use **llm-patch** — a CLI-first toolkit (with a stable Python API underneath) that converts text into LoRA adapter weights, attaches them to HuggingFace models, and serves the patched model.

> **New here?** Start with the 5-minute walkthrough in [docs/QUICKSTART.md](QUICKSTART.md). This document is the long-form reference.

---

## Table of Contents

- [Installation](#installation)
- [CLI Reference](#cli-reference)
- [Publishing & consuming adapters](#publishing--consuming-adapters)
- [Architecture Overview](#architecture-overview)
- [Quick Start (No GPU)](#quick-start-no-gpu)
- [Data Sources](#data-sources)
- [Compile Pipeline](#compile-pipeline)
- [Attach & Runtime](#attach--runtime)
- [Wiki Pipeline](#wiki-pipeline)
- [HTTP API Server](#http-api-server)
- [Configuration](#configuration)
- [Using the Makefile](#using-the-makefile)
- [Running Tests](#running-tests)
- [Troubleshooting](#troubleshooting)
- [Python API (advanced)](#python-api-advanced)

> The CLI is the recommended surface. The Python API examples in the second half of this document target plugin authors and library integrators — not first-time users.

---

## Installation

If you are working from this monorepo checkout, prefer the workspace flow:

```pwsh
uv sync
```

This installs the engine plus the workspace use-cases declared under
`projects/*`. The engine console script is `llm-patch`; the wiki use-case
console script is `llm-patch-wiki-agent`.

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.11 | Required |
| PyTorch | ≥ 2.1 | CUDA recommended for real weight generation |
| T2L Checkpoint | — | Required for real (non-mock) generation |

### Install with pip

```bash
git clone https://github.com/seismael/llm-patch.git
cd llm-patch
pip install -e .
```

### Install Extras

llm-patch uses optional extras to keep the core lightweight:

```bash
# Wiki features (YAML frontmatter parsing)
pip install -e '.[wiki]'

# CLI interface
pip install -e '.[cli]'

# PDF document ingestion
pip install -e '.[pdf]'

# HTTP API data source
pip install -e '.[http]'

# FastAPI server
pip install -e '.[server]'

# LLM-powered wiki agents
pip install -e '.[anthropic]'

# Everything
pip install -e '.[all]'
```

### Install Dev Dependencies

```bash
pip install -e '.[all]' --group dev
```

---

## Architecture Overview

llm-patch follows a four-stage pipeline:

```
Ingest → Compile → Attach → Use
```

| Stage | What it does | Key classes |
|---|---|---|
| **Ingest** | Pull documents from markdown dirs, JSONL files, PDFs, HTTP APIs, or wiki structures | `MarkdownDataSource`, `JsonlDataSource`, `PdfDataSource`, `HttpApiDataSource`, `CompositeDataSource` |
| **Compile** | Convert documents into LoRA adapter weights and store them | `CompilePipeline`, `SakanaT2LGenerator`, `LocalSafetensorsRepository` |
| **Attach** | Load a base HuggingFace model and attach compiled adapters | `HFModelProvider`, `PeftAdapterLoader`, `UsePipeline` |
| **Use** | Generate text or chat with the patched model | `PeftAgentRuntime`, `ChatSession` |

---

## Quick Start (No GPU)

The fastest way to see llm-patch in action uses mock components:

```bash
cd examples
python run_e2e.py --clean --aggregate
```

This will:

1. Copy sample ML papers from `raw/papers/` into a simulated `wiki/` directory
2. Add wiki-style frontmatter and create entity stub pages
3. Run the full pipeline with mock generator and repository
4. Report all generated adapter manifests

---

## Data Sources

All data sources implement `IDataSource` with `fetch_all()` and `fetch_one()` methods.

### Markdown Directory

Read `.md` files from a directory:

```python
from llm_patch.sources.markdown import MarkdownDataSource

source = MarkdownDataSource(
    directory="./docs",
    patterns=["*.md"],   # default
    recursive=True,      # default
)

for doc in source.fetch_all():
    print(f"{doc.document_id}: {len(doc.content)} chars")
```

### JSONL File

Read documents from a JSON Lines file:

```python
from llm_patch.sources.jsonl import JsonlDataSource

source = JsonlDataSource(
    path="./data/corpus.jsonl",
    text_field="text",  # JSON key for content
    id_field="id",      # JSON key for document ID
)
```

### PDF Directory

Requires `pip install 'llm-patch[pdf]'`:

```python
from llm_patch.sources.pdf import PdfDataSource

source = PdfDataSource(directory="./papers", recursive=True)

for doc in source.fetch_all():
    print(f"{doc.document_id}: {doc.metadata.get('page_count')} pages")
```

### HTTP API

Requires `pip install 'llm-patch[http]'`:

```python
from llm_patch.sources.http_api import HttpApiDataSource

source = HttpApiDataSource(
    url="https://api.example.com/documents",
    headers={"Authorization": "Bearer ..."},
    text_path="content.body",  # dot-path into JSON
    id_path="meta.id",
)
```

### Composite Source

Merge multiple sources into one, with namespaced IDs:

```python
from llm_patch.sources.composite import CompositeDataSource

combined = CompositeDataSource(
    markdown_source,
    jsonl_source,
    pdf_source,
    namespace_ids=True,  # IDs become "markdown:doc1", "jsonl:doc2", etc.
)

# fetch_one routes to the correct source via namespace prefix
doc = combined.fetch_one("pdf:research-paper")
```

### Wiki Source

Structured wiki directories with YAML frontmatter and `[[wikilink]]` extraction:

```python
from llm_patch.sources.wiki import WikiDataSource

source = WikiDataSource(
    directory="./wiki",
    aggregate=True,  # follow [[wikilinks]] to enrich documents
)

for doc in source.fetch_all():
    print(doc.metadata.get("title"))
    print(doc.metadata.get("wikilinks"))
```

---

## Compile Pipeline

`CompilePipeline` connects a data source, weight generator, and adapter repository:

```python
from llm_patch import CompilePipeline
from llm_patch.sources.markdown import MarkdownDataSource
from llm_patch.generators.sakana_t2l import SakanaT2LGenerator
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
from llm_patch.core.config import GeneratorConfig, StorageConfig

source = MarkdownDataSource(directory="./docs")
generator = SakanaT2LGenerator(GeneratorConfig(checkpoint_dir="./models/t2l"))
repository = LocalSafetensorsRepository(StorageConfig(output_dir="./adapters"))

pipeline = CompilePipeline(source, generator, repository)

# Batch compile all documents
manifests = pipeline.compile_all()
for m in manifests:
    print(f"  {m.adapter_id}: rank={m.rank}, path={m.storage_uri}")
```

### Single Document

```python
from llm_patch.core.models import DocumentContext

doc = DocumentContext(document_id="my-doc", content="Document text here...")
manifest = pipeline.process_document(doc)
```

### Live Watch Mode

Pass an `IKnowledgeStream` to auto-compile on file changes:

```python
from llm_patch.sources.markdown import MarkdownWatcher

watcher = MarkdownWatcher(directory="./docs")
pipeline = CompilePipeline(source, generator, repository, stream=watcher)

with pipeline:
    # Watcher is running — edits to ./docs trigger auto-compilation
    input("Press Enter to stop...\n")
```

---

## Attach & Runtime

### Loading a Model with Adapters

`UsePipeline` loads a base model, attaches adapters, and optionally wraps it in an agent:

```python
from llm_patch import UsePipeline
from llm_patch.attach.model_provider import HFModelProvider
from llm_patch.attach.peft_loader import PeftAdapterLoader
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository
from llm_patch.core.config import StorageConfig

provider = HFModelProvider()
loader = PeftAdapterLoader()
repo = LocalSafetensorsRepository(StorageConfig(output_dir="./adapters"))

use = UsePipeline(provider, loader, repo)

# Load model + attach specific adapters
handle = use.load_and_attach("google/gemma-2-2b-it", adapter_ids=["my-doc"])

# Or load all available adapters
handle = use.load_and_attach("google/gemma-2-2b-it")
```

### Building an Agent

```python
agent = use.build_agent("google/gemma-2-2b-it", adapter_ids=["my-doc"])

# Single generation
response = agent.generate("Explain the key concepts in this document")

# Multi-turn chat
from llm_patch.core.models import ChatMessage, ChatRole

reply = agent.chat([
    ChatMessage(role=ChatRole.USER, content="What are the main findings?"),
])
print(reply.message.content)
```

### ChatSession (Stateful Conversation)

```python
from llm_patch.runtime.session import ChatSession

session = ChatSession(
    runtime=agent,
    system_prompt="You are a domain expert.",
    max_history=20,  # keep last 20 messages (0 = unlimited)
)

answer = session.say("What is LoRA?")
print(answer)

answer = session.say("How does it relate to transformers?")
print(answer)

# Access conversation history
for msg in session.history:
    print(f"{msg.role}: {msg.content[:80]}")

session.clear()  # reset conversation
```

### Merging Adapters

```python
from llm_patch.attach.merger import merge_into_base, weighted_blend

# Blend multiple adapters with different weights
blended = weighted_blend(handle, {
    "api-v2": 1.0,
    "auth-guide": 0.8,
    "rate-limits": 0.5,
}, combined_name="api-expert")

# Merge active adapter into base weights (creates standalone model)
from pathlib import Path
saved_path = merge_into_base(blended, Path("./merged-model"))
```

---

## Wiki Pipeline

`WikiPipeline` manages an LLM-maintained wiki and optionally triggers compilation:

```python
from llm_patch import WikiPipeline
from llm_patch.core.config import WikiConfig
from llm_patch.wiki.agents.anthropic_agent import AnthropicWikiAgent

agent = AnthropicWikiAgent(api_key="sk-...")
config = WikiConfig(base_dir="./wiki", schema_path="./schema.md")

wiki = WikiPipeline(agent, config)

# Initialize wiki directory structure
wiki.init()

# Ingest a raw source
result = wiki.ingest(Path("./raw/papers/attention-paper.md"))

# Query the wiki
answer = wiki.query("How does self-attention work?")
print(answer.answer)
print(f"Cited pages: {answer.cited_pages}")

# Run health check
report = wiki.lint()

# Get wiki status
status = wiki.status()
print(f"Pages: {status}")
```

### Wiki Directory Structure

```
my-wiki/
├── raw/                  # Immutable source documents
│   └── papers/
├── wiki/                 # LLM-maintained wiki (llm-patch manages this)
│   ├── sources/          # Summaries of ingested documents
│   ├── entities/         # Entity pages (people, tools, models)
│   └── concepts/         # Concept pages
└── schema.md             # Instructions for the wiki agent
```

---

## HTTP API Server

Requires `pip install 'llm-patch[server]'`.

### Start the Server

```bash
# Via CLI
llm-patch serve --host 0.0.0.0 --port 8000 --adapter-dir ./adapters

# Via Python
import uvicorn
uvicorn.run("llm_patch.server.app:app", host="0.0.0.0", port=8000)
```

### Environment Variables

| Variable | Description |
|---|---|
| `LLM_PATCH_MODEL_ID` | Auto-load this HuggingFace model on startup |
| `LLM_PATCH_ADAPTER_DIR` | Adapter storage directory (default: `./adapters`) |

### Endpoints

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Health check + version |
| GET | `/adapters` | List all stored adapters |
| GET | `/adapters/{id}` | Get adapter details |
| DELETE | `/adapters/{id}` | Delete an adapter |
| POST | `/compile` | Compile a document into an adapter |
| POST | `/generate` | Single-prompt text generation |
| POST | `/chat` | Multi-turn chat completion |

### Example Requests

```bash
# Health check
curl http://localhost:8000/health

# Compile a document
curl -X POST http://localhost:8000/compile \
  -H "Content-Type: application/json" \
  -d '{"document_id": "my-doc", "content": "Document text..."}'

# Generate text
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain the key concepts", "max_new_tokens": 256}'

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What is LoRA?"}]}'
```

---

## CLI Reference

Install the CLI extra: `pip install 'llm-patch[cli]'`

### Wiki Management

```bash
# Initialize wiki
llm-patch wiki --base-dir ./wiki init

# Ingest a source file
llm-patch wiki --base-dir ./wiki ingest ./raw/paper.md

# Query the wiki
llm-patch wiki --base-dir ./wiki query "How does attention work?"

# Lint (health check)
llm-patch wiki --base-dir ./wiki lint

# Status overview
llm-patch wiki --base-dir ./wiki status

# Batch compile all unprocessed sources
llm-patch wiki --base-dir ./wiki compile
```

### Data Source Inspection

```bash
# List documents from a markdown directory
llm-patch source list --kind markdown --path ./docs

# Count documents
llm-patch source count --kind jsonl --path ./data.jsonl

# Preview a specific document
llm-patch source preview --kind markdown --path ./docs my-document-id
```

### Adapter Compilation (Legacy)

```bash
# Batch compile
llm-patch adapter compile --source-dir ./docs --output-dir ./adapters --checkpoint-dir ./models/t2l

# Watch mode
llm-patch adapter watch --source-dir ./docs --output-dir ./adapters --checkpoint-dir ./models/t2l
```

### Model & Inference

```bash
# List adapters
llm-patch model info --adapter-dir ./adapters

# One-shot generation
llm-patch model generate --model-id google/gemma-2-2b-it --adapter-dir ./adapters "Explain LoRA"

# Interactive chat
llm-patch model chat --model-id google/gemma-2-2b-it --adapter-dir ./adapters
```

### Server

```bash
llm-patch serve --host 0.0.0.0 --port 8000 --adapter-dir ./adapters
```

---

## Publishing & consuming adapters

> Available from `llm-patch` v0.2.0. Full design context: [AGENTIC_AI_INTEGRATION.md](AGENTIC_AI_INTEGRATION.md), [REGISTRY_PROTOCOL.md](REGISTRY_PROTOCOL.md).

The engine ships the **interface** for distributing adapters across
hubs but no concrete network client. Operators wire one with the
`LLM_PATCH_PLUGIN_REGISTRY` environment variable, pointing to a `module:factory`
that returns an `IAdapterRegistryClient` instance.

```pwsh
# Tell llm-patch which registry implementation to use
$Env:LLM_PATCH_PLUGIN_REGISTRY = "my_org_registry:build_registry"

# Search the hub
llm-patch hub search "react" --limit 5
llm-patch hub search "react" --json   # machine-readable

# Pull an adapter (verifies SHA-256 from manifest)
llm-patch pull hub://acme/react-19:1.2.0

# Push a locally compiled adapter
llm-patch push ./out/my-adapter --target hub://acme/my-adapter:0.1.0
```

`hf://owner/repo` and `s3://bucket/key` are reserved schemes — they are
recognized by the CLI but dispatched to whichever
`IAdapterRegistryClient` understands that scheme. If no client is
registered, the command exits cleanly with a `RegistryUnavailableError`
message and a link to [REGISTRY_PROTOCOL.md](REGISTRY_PROTOCOL.md).

### GitHub Actions (template — deferred)

A reference workflow `.github/workflows/llm-patch-publish.yml` will be
added as a commented-out template. Real publishing logic is disabled
by default; users wire credentials and uncomment as needed.

```yaml
# .github/workflows/llm-patch-publish.yml (planned)
name: Publish llm-patch adapter
on:
  push:
    tags: ["adapter-v*"]
jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync
      - env:
          LLM_PATCH_PLUGIN_REGISTRY: my_org_registry:build_registry
          HUB_TOKEN: ${{ secrets.HUB_TOKEN }}
        run: |
          uv run llm-patch compile ./docs --output ./out
          uv run llm-patch push ./out --target hub://acme/docs:${{ github.ref_name }}
```

---

## Configuration

### Config Dataclasses

All configuration uses Pydantic models from `llm_patch.core.config`.

#### GeneratorConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `checkpoint_dir` | `Path` | — | Path to T2L checkpoint directory |
| `device` | `str` | `"cuda"` | PyTorch device |

#### StorageConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `Path` | — | Directory for adapter storage |

#### WatcherConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `directory` | `Path` | — | Directory to monitor |
| `patterns` | `list[str]` | `["*.md"]` | File glob patterns |
| `recursive` | `bool` | `True` | Watch subdirectories |
| `debounce_seconds` | `float` | `0.5` | Debounce interval |

#### WikiConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `base_dir` | `Path` | — | Wiki root directory |
| `schema_path` | `Path \| None` | `None` | Path to wiki schema file |
| `obsidian` | `bool` | `False` | Obsidian vault mode |

#### ModelSpec

| Field | Type | Default | Description |
|---|---|---|---|
| `model_id` | `str` | — | HuggingFace model identifier |
| `dtype` | `str` | `"float16"` | Model data type |
| `device_map` | `str` | `"auto"` | Device placement strategy |
| `trust_remote_code` | `bool` | `False` | Trust remote model code |

#### AttachConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `adapter_dir` | `Path` | — | Directory containing compiled adapters |
| `adapter_name` | `str \| None` | `None` | Specific adapter to load |

#### AgentConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `model_spec` | `ModelSpec` | — | Base model specification |
| `adapter_ids` | `list[str]` | `[]` | Adapters to attach |
| `generation_max_new_tokens` | `int` | `256` | Max tokens per generation |
| `generation_temperature` | `float` | `0.7` | Sampling temperature |
| `system_prompt` | `str \| None` | `None` | Default system prompt |

#### ServerConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `host` | `str` | `"127.0.0.1"` | Server bind address |
| `port` | `int` | `8000` | Server port |
| `adapter_dir` | `Path` | `Path("adapters")` | Adapter storage directory |
| `cors_origins` | `list[str]` | `["*"]` | Allowed CORS origins |

### Data Source Configs

Use the discriminated union `DataSourceConfig` to configure sources declaratively:

```python
from llm_patch.core.config import MarkdownSourceConfig, JsonlSourceConfig

md_config = MarkdownSourceConfig(
    type="markdown",
    directory="./docs",
    patterns=["*.md"],
    recursive=True,
)

jsonl_config = JsonlSourceConfig(
    type="jsonl",
    path="./data.jsonl",
    text_field="text",
    id_field="id",
)
```

---

## Using the Makefile

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
python -m pytest --cov=llm_patch --cov-report=term-missing
```

### Unit Tests Only

```bash
python -m pytest tests/unit/ -v
```

### Integration Tests Only

```bash
python -m pytest tests/integration/ -v
```

### Quick Run (Stop on First Failure)

```bash
python -m pytest -x -q
```

---

## Troubleshooting

### Common Issues

**"No module named 'hyper_llm_modulator'"**

The `SakanaT2LGenerator` requires the Sakana AI hypernetwork library. For testing without GPU, use mock components in `examples/`.

**"CUDA out of memory"**

Try `device="cpu"` in `GeneratorConfig` or use a smaller model.

**"ModuleNotFoundError: No module named 'pypdf'"**

Install the PDF extra: `pip install 'llm-patch[pdf]'`

**"ModuleNotFoundError: No module named 'httpx'"**

Install the HTTP extra: `pip install 'llm-patch[http]'`

**"ModuleNotFoundError: No module named 'fastapi'"**

Install the server extra: `pip install 'llm-patch[server]'`

**Adapters not regenerating on file changes**

- Check that files match configured `patterns` (default: `["*.md"]`)
- Ensure the watcher/stream is running via context manager or `start()`
- Check `debounce_seconds` — rapid saves within the debounce window are collapsed

**Type checking errors with torch**

Add to your `mypy` config:

```toml
[[tool.mypy.overrides]]
module = ["torch.*", "safetensors.*", "watchdog.*", "hyper_llm_modulator.*"]
ignore_missing_imports = true
```