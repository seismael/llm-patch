<div align="center">

# llm-patch

**Instant Knowledge Internalization for Large Language Models**

*Turn any text document into LoRA adapter weights in a single forward pass — no fine-tuning, no retraining, no waiting.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-≥2.1-ee4c2c.svg)](https://pytorch.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

`llm-patch` converts text documents into [LoRA](https://arxiv.org/abs/2106.09685) adapter weights using [Sakana AI's Text-to-LoRA](https://github.com/SakanaAI/text-to-lora) hypernetworks. Point it at a directory of Markdown files — API docs, research papers, wiki pages, internal knowledge bases — and it produces small, composable adapter weights (~2–5 MB each) that can be dynamically loaded into any compatible LLM at inference time.

The result: **your language model learns new knowledge instantly**, without gradient-based training, GPU hours, or model redeployment.

### Highlights

- **Zero fine-tuning** — A single forward pass through a hypernetwork synthesizes LoRA weights from text embeddings. No training loop, no optimizer, no loss function.
- **Real-time watch mode** — Monitor directories for document changes. New or updated files automatically trigger adapter generation within seconds.
- **Composable knowledge** — Each document becomes an independent adapter. Load one, stack several, or swap them dynamically per request.
- **Production-ready output** — Adapters are stored in the HuggingFace-standard `safetensors` format with PEFT-compatible configuration. Load them with one line of code.
- **Pluggable architecture** — Swap sources, generators, and storage backends independently. Built on SOLID principles with clean interfaces.
- **Wiki-aware ingestion** — First-class support for structured wiki directories with YAML frontmatter parsing, `[[wikilink]]` resolution, and cross-page content aggregation.

---

## Table of Contents

- [Why llm-patch?](#why-llm-patch)
- [How It Works](#how-it-works)
- [Use Cases](#use-cases)
- [Building Self-Improving AI Agents](#building-self-improving-ai-agents)
- [Getting Started](#getting-started)
- [Architecture](#architecture)
- [Configuration Reference](#configuration-reference)
- [Extending](#extending)
- [End-to-End Example](#end-to-end-example)
- [Roadmap](#roadmap)
- [Development](#development)
- [Contributing](#contributing)
- [Community & Links](#community--links)
- [Documentation](#documentation)
- [License](#license)

---

## Why llm-patch?

Large language models are powerful, but they're frozen at training time. When your knowledge changes — new API versions ship, policies update, research advances — you have limited options:

### The Problem with Current Approaches

| | **Traditional Fine-Tuning** | **RAG (Retrieval-Augmented Generation)** | **llm-patch** |
|---|---|---|---|
| **Speed** | Hours to days | Real-time retrieval | Seconds (single forward pass) |
| **Cost** | High (GPU compute for training) | Medium (embedding + vector DB) | Low (inference-only) |
| **Knowledge depth** | Deep (baked into weights) | Shallow (context window only) | Deep (baked into weights) |
| **Freshness** | Stale until retrained | Always current | Always current (watch mode) |
| **Hallucination risk** | Low for trained topics | Medium (retrieval noise) | Low (weight-level injection) |
| **Context window** | Not consumed | Consumed (limits reasoning space) | Not consumed |
| **Composability** | Requires re-training | Per-query retrieval | Mix-and-match adapters |
| **Offline capable** | Yes | Needs retrieval infrastructure | Yes |

**Fine-tuning** gives deep knowledge but is slow, expensive, and static. Every knowledge update requires a new training run.

**RAG** is fast and fresh but operates at the context level — retrieved chunks consume your context window, introduce retrieval noise, and the model doesn't truly *understand* the knowledge; it's just reading it at inference time.

**llm-patch** offers a third path: **instant weight-level knowledge injection**. Documents are converted into LoRA adapter weights through a hypernetwork in a single forward pass. The knowledge is embedded directly into the model's parameters — as deep as fine-tuning, as fresh as RAG, with neither the cost of one nor the limitations of the other.

---

## How It Works

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Documents   │────►│  Text        │────►│ Hypernetwork │────►│ LoRA Adapter │
│  (Markdown)  │     │  Embedding   │     │ (T2L)        │     │  Weights     │
└──────────────┘     └──────────────┘     └──────────────┘     └──────┬───────┘
                                                                      │
                                                                      ▼
                                                               ┌──────────────┐
                                                               │ Load into    │
                                                               │ any LLM via  │
                                                               │ PEFT         │
                                                               └──────────────┘
```

### The Pipeline in Plain Language

1. **Ingest** — llm-patch watches a directory (or wiki) for Markdown files. When a document appears or changes, it's captured as a `DocumentContext` with metadata, frontmatter, and wikilink references.

2. **Embed** — The document text is encoded into a dense vector representation using a sentence embedding model (e.g., all-MiniLM-L6-v2).

3. **Generate** — The embedding is fed through a [Text-to-LoRA (T2L) hypernetwork](https://github.com/SakanaAI/text-to-lora) — a meta-network trained by Sakana AI that *generates* LoRA weight matrices from text embeddings in a single forward pass. No gradient descent, no training loop. Pure inference.

4. **Store** — The resulting LoRA weights (~2–5 MB) are saved as `safetensors` files alongside a PEFT-compatible `adapter_config.json`. Each adapter is stored in its own directory, ready for loading.

5. **Load** — At inference time, load any adapter into your base model with a single line of PEFT code. Swap adapters per request, stack multiple adapters, or serve different adapters to different users.

### Why Hypernetworks?

Traditional fine-tuning adjusts model weights through iterative gradient descent over many examples. A **hypernetwork** takes a shortcut: it's a neural network that has been trained to *predict* what the fine-tuned weights would look like, given a text description. The result is adapter generation in ~100ms instead of hours — at the cost of some precision compared to full fine-tuning, but with massive gains in speed and flexibility.

---

## Use Cases

llm-patch enables a fundamentally new pattern: **models that learn from documents in real-time**. Here are concrete scenarios where this capability transforms workflows:

### 1. Self-Improving AI Agents

AI agents that operate over a knowledge base (wiki, Confluence, Notion) can use llm-patch to **continuously internalize their own knowledge source**. As the knowledge base evolves — through the agent's own actions or human edits — new adapters are generated automatically. The agent's underlying LLM always reflects the latest state of knowledge without redeployment or retraining.

> *An agent that writes wiki pages about customer issues can then load adapters generated from those very pages, becoming progressively smarter about the problem domain it documents.*

### 2. Personal Research Assistant

Researchers and knowledge workers can point llm-patch at their local notes directory — Obsidian vaults, Markdown collections, Zettlekasten systems. As they read papers and take notes, adapters are generated in real-time, and their local LLM **absorbs what they're actively researching**. The model doesn't just retrieve your notes — it *internalizes* them.

> *A machine learning researcher reading papers on diffusion models gets a local LLM that progressively develops deep understanding of the topic, going beyond what RAG can achieve with context-window stuffing.*

### 3. Enterprise Support Agents

Support teams maintain documentation per product version (API v1, v2, v3). llm-patch watches each version's wiki and generates **version-specific adapters**. When a customer asks a question, the support LLM dynamically loads the adapter matching their product version — delivering precise, version-accurate answers with zero cross-version hallucination.

> *A SaaS company with 5 API versions maintains 5 wiki directories. llm-patch produces 5 adapters. The support chatbot loads the right one per customer, eliminating "that feature was removed in v3" errors entirely.*

### 4. Living Documentation QA

Engineering teams point llm-patch at their documentation repository. Every commit to the docs triggers adapter regeneration. Internal chatbots and developer tools always answer questions based on **the current state of documentation** — not a stale training snapshot from months ago.

> *A platform team's internal "ask the docs" chatbot never gives outdated answers because its adapter is regenerated on every docs PR merge.*

### 5. Education & Tutoring

Course instructors convert their curriculum — lecture notes, textbooks, problem sets — into adapters. Students interact with an LLM tutor that has **deep, curriculum-specific knowledge** rather than generic training data. Different courses produce different adapters; students load the one matching their class.

> *A university's "AI Teaching Assistant" loads the adapter for CS 301 when a student asks about database normalization, and switches to the CS 201 adapter for data structures questions.*

### 6. Compliance & Legal

Legal and compliance teams convert regulation documents, internal policies, and contractual language into adapters. The resulting LLM has the **exact text of policies baked into its weights** — not retrieved from a vector database where chunking and retrieval errors can cause critical misquotes.

> *A compliance officer asks "what are our data retention requirements for EU customers?" and gets an answer grounded in the actual policy text, not a paraphrased retrieval result.*

### 7. Multi-Tenant SaaS Knowledge

SaaS platforms where each customer has their own knowledge base can generate **per-tenant adapters**. One base model serves all tenants; the correct adapter is loaded dynamically per API request. This is dramatically more efficient than running separate fine-tuned models per tenant.

> *A customer success platform with 200 enterprise clients maintains 200 knowledge bases. llm-patch generates 200 adapters (~1 GB total). One base model + dynamic adapter loading serves all clients.*

### 8. CI/CD-Driven Model Updates

Integrate llm-patch into your CI/CD pipeline. Every documentation commit triggers adapter regeneration as a pipeline step. The freshest adapters are deployed alongside your application code. **Your model's knowledge is as current as your last merged pull request.**

> *A GitHub Action runs `llm-patch compile` on every push to `docs/`. The resulting adapters are uploaded to S3 and hot-loaded by the production inference server.*

### 9. Competitive Intelligence

Product and strategy teams monitor competitor documentation, changelogs, and public knowledge bases. llm-patch watches these sources and generates adapters that keep an analysis LLM **perpetually current on the competitive landscape** — no manual re-ingestion or periodic retraining required.

> *A product manager asks "what features did Competitor X release this quarter?" and gets answers grounded in the competitor's actual documentation, updated automatically as their docs change.*

### 10. Domain-Specific Copilots

Build specialized coding assistants by feeding API documentation, framework guides, and internal libraries into llm-patch. The resulting adapters create an LLM copilot that **deeply understands your specific tech stack** — not just public training data, but your internal SDKs, your API conventions, your architecture decisions.

> *A development team working with a proprietary internal framework gets a coding assistant that knows framework-specific patterns, avoiding the generic suggestions that public models provide.*

---

## Building Self-Improving AI Agents

One of the most powerful applications of llm-patch is enabling **agents that continuously improve their own model** based on evolving knowledge. This goes beyond RAG-augmented agents — the knowledge isn't just retrieved at query time; it's **internalized into the model's weights**.

### The Self-Improvement Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    CONTINUOUS IMPROVEMENT LOOP               │
│                                                              │
│   ┌──────────┐     ┌──────────────┐     ┌───────────────┐   │
│   │  Agent   │────►│ Knowledge    │────►│  llm-patch    │   │
│   │  Acts    │     │ Base Updates │     │  Watches &    │   │
│   │          │     │ (wiki, docs) │     │  Generates    │   │
│   └──────────┘     └──────────────┘     └───────┬───────┘   │
│        ▲                                        │           │
│        │           ┌──────────────┐              │           │
│        │           │  New LoRA    │              │           │
│        └───────────│  Adapter     │◄─────────────┘           │
│                    │  Loaded      │                          │
│                    └──────────────┘                          │
│                                                              │
│   Result: Agent becomes progressively smarter with each      │
│   knowledge base update — no retraining, no redeployment.    │
└─────────────────────────────────────────────────────────────┘
```

### How It Differs from RAG-Based Agents

| Aspect | RAG-Based Agent | llm-patch Agent |
|---|---|---|
| **Knowledge location** | External (retrieved at query time) | Internal (baked into model weights) |
| **Context window** | Consumed by retrieved chunks | Fully available for reasoning |
| **Reasoning depth** | Surface-level (reading retrieved text) | Deep (knowledge is part of model parameters) |
| **Latency** | Retrieval + generation | Generation only (adapter pre-loaded) |
| **Offline capability** | Requires vector DB infrastructure | Fully offline after adapter generation |
| **Knowledge conflicts** | Possible (retrieval noise) | Resolved at generation time |

### Concrete Example: Wiki-Powered Agent

llm-patch ships with `WikiKnowledgeSource` — an `IKnowledgeSource` implementation designed specifically for wiki-style knowledge bases (like those produced by [llm-wiki-agent](https://github.com/SamurAIGPT/llm-wiki-agent)):

```python
from llm_patch import (
    KnowledgeFusionOrchestrator,
    WikiKnowledgeSource,
    SakanaT2LGenerator,
    LocalSafetensorsRepository,
    GeneratorConfig, StorageConfig, WatcherConfig,
)

# Watch a wiki directory with cross-page aggregation
source = WikiKnowledgeSource(
    WatcherConfig(directory="./agent_wiki"),
    aggregate=True,  # Follow [[wikilinks]] for richer context
)

orchestrator = KnowledgeFusionOrchestrator(
    source=source,
    generator=SakanaT2LGenerator(GeneratorConfig(checkpoint_dir="./t2l_checkpoint")),
    repository=LocalSafetensorsRepository(StorageConfig(output_dir="./agent_adapters")),
)

# The agent's knowledge base is now a live pipeline:
# Wiki page created/updated → adapter generated → agent loads new adapter
with orchestrator:
    ...  # Agent operates while adapters are updated in the background
```

The `WikiKnowledgeSource` parses YAML frontmatter, extracts `[[wikilinks]]`, and optionally aggregates linked entity/concept pages into enriched documents — giving the hypernetwork richer context for more accurate weight generation.

---

## Getting Started

### Installation

```bash
# Clone and install with uv
git clone https://github.com/seismael/llm-patch.git
cd llm-patch
uv sync
```

### Prerequisites

- Python ≥ 3.11
- PyTorch ≥ 2.1
- A pre-trained T2L checkpoint from [SakanaAI/text-to-lora](https://github.com/SakanaAI/text-to-lora)

### Quickstart

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

# 1. Configure components
watcher = MarkdownDirectoryWatcher(
    WatcherConfig(directory="./api_docs")
)
generator = SakanaT2LGenerator(
    GeneratorConfig(checkpoint_dir="./models/gemma_2b_t2l")
)
repository = LocalSafetensorsRepository(
    StorageConfig(output_dir="./compiled_adapters")
)

# 2. Create the orchestrator (auto-registers as observer)
orchestrator = KnowledgeFusionOrchestrator(
    source=watcher,
    generator=generator,
    repository=repository,
)

# 3. Compile all existing documents into adapters
manifests = orchestrator.compile_all()
for m in manifests:
    print(f"  {m.adapter_id} → {m.storage_uri}")

# 4. Watch for future changes (blocking via context manager)
with orchestrator:
    # New/modified .md files automatically trigger weight generation
    ...
```

### Loading Adapters at Inference

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
model = PeftModel.from_pretrained(base_model, "./compiled_adapters/api_v2")

# The model now has "api_v2" knowledge baked into its weights
response = model.generate(...)
```

---

## Architecture

The library follows SOLID principles with three pluggable layers coordinated by a central Facade. For a deep dive into the design patterns, data flow, and extensibility points, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
┌─────────────────────────────────────────────────┐
│         KnowledgeFusionOrchestrator (Facade)    │
├──────────────┬─────────────────┬────────────────┤
│ IKnowledge   │ IWeight         │ IAdapter       │
│ Source       │ Generator       │ Repository     │
│ (Observer)   │ (Strategy)      │ (Repository)   │
├──────────────┼─────────────────┼────────────────┤
│ Markdown     │ SakanaT2L       │ LocalSafe      │
│ Directory    │ Generator       │ tensors        │
│ Watcher /    │                 │ Repository     │
│ Wiki Source  │                 │                │
└──────────────┴─────────────────┴────────────────┘
```

### Design Patterns

- **Facade Pattern** — `KnowledgeFusionOrchestrator` provides a single entry point. Users interact with the orchestrator, not the individual layers.
- **Strategy Pattern** — Swap weight generation backends by implementing `IWeightGenerator`. Ship with `SakanaT2LGenerator`; plug in custom hypernetworks or future backends.
- **Observer Pattern** — `IKnowledgeSource` implementations watch for document changes and notify the orchestrator via callbacks. No polling required.
- **Repository Pattern** — `IAdapterRepository` decouples adapter persistence from business logic. Store locally with `LocalSafetensorsRepository`, or implement S3, GCS, or HuggingFace Hub backends.

### Adapter Output Structure

Each generated adapter is stored as a self-contained directory:

```
compiled_adapters/
└── {document_id}/
    ├── adapter_model.safetensors   # LoRA weights (2-5 MB)
    ├── adapter_config.json          # PEFT LoraConfig metadata
    └── manifest.json                # Generation metadata & timestamps
```

---

## Configuration Reference

| Config | Field | Default | Description |
|---|---|---|---|
| `GeneratorConfig` | `checkpoint_dir` | — | Path to T2L checkpoint directory |
| | `device` | `"cuda"` | PyTorch device (`"cuda"`, `"cpu"`, `"mps"`) |
| `WatcherConfig` | `directory` | — | Directory to monitor |
| | `patterns` | `["*.md"]` | Glob patterns to match |
| | `recursive` | `True` | Watch subdirectories |
| | `debounce_seconds` | `0.5` | Debounce interval for rapid saves |
| `StorageConfig` | `output_dir` | — | Adapter output directory |

---

## Extending

Build custom backends by implementing the core interfaces:

```python
from llm_patch import IWeightGenerator, IAdapterRepository, IKnowledgeSource
```

**Custom knowledge sources** — Ingest from databases, APIs, CMS platforms, or any structured text source by implementing `IKnowledgeSource`.

**Custom generators** — Integrate alternative hypernetworks, distillation methods, or future text-to-weight approaches by implementing `IWeightGenerator`.

**Custom storage** — Write adapters to S3, Google Cloud Storage, HuggingFace Hub, or any remote storage by implementing `IAdapterRepository`.

---

## End-to-End Example

The `examples/` directory contains a complete tutorial that chains an [LLM Wiki Agent](https://github.com/SamurAIGPT/llm-wiki-agent) with llm-patch to build domain-specialized LoRA adapters from ML research papers:

```
raw/papers/*.md ──► LLM Wiki Agent ──► wiki/ ──► WikiKnowledgeSource ──► adapters/
```

### Quick Demo (No GPU Required)

```bash
cd examples
python run_e2e.py --clean --aggregate
```

This runs the full pipeline end-to-end using mock components — ingesting sample papers, simulating wiki generation, and producing adapter manifests. See [examples/README.md](examples/README.md) for the full tutorial, including batch mode, watch mode, wikilink aggregation, and adapter validation.

---

## Roadmap

llm-patch is under active development. Here's what's planned:

- [ ] **Cloud storage backends** — S3, GCS, and Azure Blob adapters for `IAdapterRepository`
- [ ] **Adapter merging** — Combine multiple per-document adapters into a single domain adapter using PEFT's `add_weighted_adapter()`
- [ ] **Multi-model support** — Extend beyond Gemma to LLaMA, Mistral, Phi, and other architectures
- [ ] **REST API server** — HTTP service for on-demand adapter generation and serving
- [ ] **HuggingFace Hub integration** — Push/pull adapters directly from the Hub
- [ ] **Incremental updates** — Delta adapter generation when documents are partially updated
- [ ] **Adapter quality scoring** — Automated evaluation of generated adapter quality against ground-truth QA pairs
- [ ] **Batch inference optimization** — Process multiple documents in parallel through the hypernetwork
- [ ] **Web UI dashboard** — Monitor pipeline status, adapter inventory, and generation history

---

## Development

A `Makefile` is provided for common tasks:

```bash
make install-dev      # Install all dependencies + pre-commit hooks
make check            # Run lint + typecheck + test (full CI check)
make test             # Run all tests with coverage
make test-unit        # Run unit tests only
make lint             # Run ruff linter
make format           # Auto-format code
make typecheck        # Run mypy strict mode
make demo             # Run end-to-end demo (no GPU)
make clean            # Remove build artifacts and caches
```

Or use Poetry directly:

```bash
poetry install --with dev
poetry run pytest
poetry run ruff check src/ tests/
poetry run mypy src/
poetry run pre-commit install
```

See [docs/USAGE.md](docs/USAGE.md) for the complete usage guide, including configuration, troubleshooting, and advanced adapter loading patterns.

---

## Contributing

Contributions are welcome! Whether it's a new `IKnowledgeSource` for your favorite CMS, a storage backend for cloud providers, bug fixes, or documentation improvements — we'd love your help.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-new-source`)
3. Write tests for your changes
4. Run `make check` to verify lint, types, and tests pass
5. Open a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines, code standards, and areas where contributions are especially welcome.

---

## Community & Links

### Related Projects

- **[Sakana AI — Text-to-LoRA](https://github.com/SakanaAI/text-to-lora)** — The hypernetwork backend that powers llm-patch's weight generation
- **[PEFT](https://github.com/huggingface/peft)** — Parameter-Efficient Fine-Tuning library for loading and managing LoRA adapters
- **[safetensors](https://github.com/huggingface/safetensors)** — Safe, fast tensor serialization format used for adapter storage
- **[llm-wiki-agent](https://github.com/SamurAIGPT/llm-wiki-agent)** — LLM-powered wiki generation agent that pairs naturally with llm-patch

### Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

---

## Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | Project overview, use cases, and quickstart |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, design patterns, and data flow |
| [docs/USAGE.md](docs/USAGE.md) | Installation, configuration, and usage guide |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, code standards, and PR checklist |
| [examples/README.md](examples/README.md) | End-to-end tutorial with example scripts |
| [NOTES.md](NOTES.md) | Design decisions and implementation notes |

---

## License

Apache-2.0 — See [LICENSE](LICENSE) for the full text.

Free to use, modify, and distribute for any purpose, including commercial use.
