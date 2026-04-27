<div align="center">

# llm-patch

**The instant LoRA toolkit — turn any folder into a model adapter in seconds.**

*A CLI-first OSS tool for compiling text into LoRA weights, attaching them to any HuggingFace model, and serving the patched model.*

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-≥2.1-ee4c2c.svg)](https://pytorch.org/)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

</div>

---

## 60-second quickstart

```pwsh
pip install "llm-patch[cli]"
llm-patch doctor                                     # verify env (Python/torch/CUDA/extras)
llm-patch init                                       # interactive scaffold → .llm-patch.toml
llm-patch compile ./docs --output ./adapters         # folder → LoRA adapter
llm-patch chat --base google/gemma-2-2b-it --adapter ./adapters/my-notes
```

That's the whole pipeline. Full walkthrough in [docs/QUICKSTART.md](docs/QUICKSTART.md).

`llm-patch` converts text from **any source** (Markdown, Wiki, PDF, JSONL, HTTP APIs) into [LoRA](https://arxiv.org/abs/2106.09685) adapter weights and serves the patched model via CLI, HTTP API, or MCP. The default weight-generation backend is [Sakana AI's Text-to-LoRA](https://github.com/SakanaAI/text-to-lora) — a single forward pass, no training loop.

The result: **your language model learns new knowledge instantly**, without gradient-based fine-tuning, GPU hours, or model redeployment.

### Highlights

- **CLI-first** — `init`, `compile`, `watch`, `chat`, `push`, `pull`, `serve`, `hub`, `doctor`. `--help` is fast (no torch on cold start).
- **Zero fine-tuning** — single hypernetwork forward pass. No training loop, no optimizer.
- **Multi-source ingestion** — Markdown, wikis, PDFs, JSONL, HTTP APIs, or any custom `IDataSource`. Compose with `CompositeDataSource`.
- **Real-time watch mode** — adapters regenerate within seconds of a file change.
- **Composable knowledge** — load one adapter, stack several, merge them, hot-swap per request.
- **Adapter Market** — `push`/`pull` adapters across hubs ([docs/AGENTIC_AI_INTEGRATION.md](docs/AGENTIC_AI_INTEGRATION.md)).
- **Pluggable architecture** — sources, generators, registries, runtimes are all ABCs ([docs/EXTENDING.md](docs/EXTENDING.md)).

---

## Table of Contents

- [Workspace Layout](#workspace-layout)
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

---

## Workspace Layout

This repository is a **uv workspace monorepo**. The structure is:

```
projects/
  llm-patch/         engine — generic Ingest → Compile → Attach → Use framework
  utils/             cross-project, stdlib-only utilities (llm_patch_utils)
  wiki-agent/        first downstream use-case (llm_patch_wiki_agent)
docs/
  adr/               Architecture Decision Records (MADR format)
  ARCHITECTURE.md, USAGE.md, E2E_WALKTHROUGH.md
tools/               cross-project scripts (scaffolding, layering, coverage)
.github/
  instructions/      scoped agent instructions (applyTo)
  workflows/         CI / release pipelines
SPEC.md              binding engineering specification
AGENTS.md            root agent contract
```

| Project | Description | Test command |
|---|---|---|
| [`projects/llm-patch`](projects/llm-patch) | The generic engine. | `uv run --package llm-patch pytest` |
| [`projects/utils`](projects/utils) | Cross-project utilities (stdlib-only). | `uv run --package llm-patch-utils pytest` |
| [`projects/wiki-agent`](projects/wiki-agent) | Wiki-specialized agent with `compile`, one-shot `chat`, and `info` commands built on the engine. | `uv run --package llm-patch-wiki-agent pytest` |

Dependencies flow **one-way**: use-cases → engine → utils.
Enforced by [tools/check_layering.py](tools/check_layering.py).

### Authoritative Documents

- [SPEC.md](SPEC.md) — binding engineering specification (SOLID, GoF, TDD, layering).
- [AGENTS.md](AGENTS.md) — root agent contract and pre-change checklist.
- [docs/adr/README.md](docs/adr/README.md) — Architecture Decision Records.
- Scoped agent rules under `.github/instructions/*.instructions.md`
  (applied automatically via `applyTo` globs; see ADR-0009).

### Adding a New Use-Case Project

```pwsh
uv run python tools/scaffold_project.py <name>
```

This materializes the standardized layout (`src/`, `tests/`,
`pyproject.toml`, `README.md`, `CHANGELOG.md`). The project is
automatically picked up by the workspace. No per-project `AGENTS.md`
is emitted — the single root `AGENTS.md` governs all projects.

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
┌──────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  IDataSource     │────►│  Text        │────►│ Hypernetwork │────►│ LoRA Adapter │
│  (Markdown, Wiki,│     │  Embedding   │     │ (T2L)        │     │  Weights     │
│   PDF, JSONL,    │     └──────────────┘     └──────────────┘     └──────┬───────┘
│   HTTP API, ...) │                                                      │
└──────────────────┘                                                      ▼
                                                               ┌──────────────────┐
                                                               │ IModelProvider   │
                                                               │ + IAdapterLoader │
                                                               │ → ModelHandle    │
                                                               └────────┬─────────┘
                                                                        │
                                                                        ▼
                                                               ┌──────────────────┐
                                                               │ IAgentRuntime    │
                                                               │ generate / chat  │
                                                               │ / stream         │
                                                               └──────────────────┘
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

### 0. The Adapter Market — Distributed Knowledge Registry & Agentic Runtime (v0.2.0)

Treat LoRA adapters as **versioned, immutable artifacts** — like Docker images or NPM packages. Agents discover their own knowledge gaps, pull a specialized adapter from a hub, hot-swap it into the live model, and generate from a now-pristine context window. The engine ships the four primitives (manifest v2, registry-client ABC, LRU cache, hot-swap controller) and the CLI/MCP/server hooks; operators plug in their transport.

```pwsh
$Env:LLM_PATCH_PLUGIN_REGISTRY = "my_org_registry:build_registry"
llm-patch hub search "react"
llm-patch pull hub://acme/react-19:1.2.0
```

→ See [docs/AGENTIC_AI_INTEGRATION.md](docs/AGENTIC_AI_INTEGRATION.md) for the full design (sequence diagram + per-requirement Status Matrix), [docs/REGISTRY_PROTOCOL.md](docs/REGISTRY_PROTOCOL.md) for the wire format, and [docs/SERVER_ARCHITECTURE.md](docs/SERVER_ARCHITECTURE.md) for the hot-swap concurrency model.

### 1. LLM Wiki Knowledge Specialization

The **[LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** pattern (described by Andrej Karpathy) proposes using LLMs to build and maintain a persistent, structured wiki — a compounding knowledge base where every source you add is synthesized, cross-referenced, and interlinked by the LLM. LLM Wikis are emerging as a powerful alternative to RAG — instead of retrieving raw chunks at query time, the knowledge is already compiled, synthesized, and interlinked. llm-patch takes this a step further: **convert the entire accumulated wiki into LoRA adapter weights** and bake that knowledge directly into a local model like Gemma.

The result: a locally-running LLM that is a **genuine domain expert** on everything in your wiki — your research, your notes, your curated knowledge — without consuming context window tokens and without requiring cloud APIs.

> *A researcher maintains an LLM Wiki on machine learning over six months — hundreds of interlinked pages covering papers, concepts, entities, and evolving syntheses. llm-patch watches the wiki directory and generates adapters as pages are created and updated. The researcher loads the merged adapter into a local Gemma model that now has deep, weight-level understanding of the entire knowledge base — answering questions with the synthesized insight of hundreds of sources, not just retrieving fragments.*

**Why this matters — LLM Wiki vs RAG vs llm-patch:**

| | **RAG** | **LLM Wiki alone** | **LLM Wiki + llm-patch** |
|---|---|---|---|
| **Knowledge structure** | Raw chunks in vector DB | Synthesized, interlinked pages | Baked into model weights |
| **Context window** | Consumed by retrieved chunks | Consumed by retrieved pages | Fully available for reasoning |
| **Knowledge depth** | Shallow (reading fragments) | Medium (reading syntheses) | Deep (internalized in parameters) |
| **Offline capability** | Needs vector DB infrastructure | Needs LLM + wiki access | Adapter loaded once, fully offline |
| **Compounding** | No accumulation | Knowledge compounds over time | Compounded knowledge in weights |
| **Scalability** | Degrades with retrieval noise | Degrades beyond context window | Adapters compress knowledge into ~2-5 MB |
| **Privacy** | May hit cloud APIs | May hit cloud APIs | Fully local with models like Gemma |

**How to set it up:**

```python
from llm_patch import (
    KnowledgeFusionOrchestrator,
    WikiKnowledgeSource,
    SakanaT2LGenerator,
    LocalSafetensorsRepository,
    GeneratorConfig, StorageConfig, WatcherConfig,
)

# Point at your LLM Wiki directory (Obsidian vault, markdown wiki, etc.)
source = WikiKnowledgeSource(
    WatcherConfig(directory="./my-llm-wiki/wiki"),
    aggregate=True,  # Follow [[wikilinks]] for cross-page synthesis
)

orchestrator = KnowledgeFusionOrchestrator(
    source=source,
    generator=SakanaT2LGenerator(GeneratorConfig(checkpoint_dir="./models/gemma_2b_t2l")),
    repository=LocalSafetensorsRepository(StorageConfig(output_dir="./wiki_adapters")),
)

# Compile all existing wiki pages into adapters
manifests = orchestrator.compile_all()

# Then watch for ongoing wiki updates
with orchestrator:
    # As your LLM Wiki agent adds/updates pages, adapters regenerate automatically
    ...
```

Then load the adapters into your local model:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

# Load Gemma (or any supported model) with your wiki knowledge
base = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it")
model = PeftModel.from_pretrained(base, "./wiki_adapters/merged")

# Your local LLM is now an expert on your entire knowledge base
```

This bridges the gap between **knowledge curation** (what LLM Wiki does well — synthesizing, cross-referencing, maintaining) and **knowledge internalization** (what llm-patch does — converting that curated knowledge into model weights). Together, they create a fully local, continuously-improving, domain-expert LLM. See [docs/USAGE.md](docs/USAGE.md#llm-wiki-integration) for the complete integration guide.

### 2. Self-Improving AI Agents

AI agents that operate over a knowledge base (wiki, Confluence, Notion) can use llm-patch to **continuously internalize their own knowledge source**. As the knowledge base evolves — through the agent's own actions or human edits — new adapters are generated automatically. The agent's underlying LLM always reflects the latest state of knowledge without redeployment or retraining.

> *An agent that writes wiki pages about customer issues can then load adapters generated from those very pages, becoming progressively smarter about the problem domain it documents.*

### 3. Personal Research Assistant

Researchers and knowledge workers can point llm-patch at their local notes directory — Obsidian vaults, Markdown collections, Zettlekasten systems. As they read papers and take notes, adapters are generated in real-time, and their local LLM **absorbs what they're actively researching**. The model doesn't just retrieve your notes — it *internalizes* them.

> *A machine learning researcher reading papers on diffusion models gets a local LLM that progressively develops deep understanding of the topic, going beyond what RAG can achieve with context-window stuffing.*

### 4. Enterprise Support Agents

Support teams maintain documentation per product version (API v1, v2, v3). llm-patch watches each version's wiki and generates **version-specific adapters**. When a customer asks a question, the support LLM dynamically loads the adapter matching their product version — delivering precise, version-accurate answers with zero cross-version hallucination.

> *A SaaS company with 5 API versions maintains 5 wiki directories. llm-patch produces 5 adapters. The support chatbot loads the right one per customer, eliminating "that feature was removed in v3" errors entirely.*

### 5. Living Documentation QA

Engineering teams point llm-patch at their documentation repository. Every commit to the docs triggers adapter regeneration. Internal chatbots and developer tools always answer questions based on **the current state of documentation** — not a stale training snapshot from months ago.

> *A platform team's internal "ask the docs" chatbot never gives outdated answers because its adapter is regenerated on every docs PR merge.*

### 6. Education & Tutoring

Course instructors convert their curriculum — lecture notes, textbooks, problem sets — into adapters. Students interact with an LLM tutor that has **deep, curriculum-specific knowledge** rather than generic training data. Different courses produce different adapters; students load the one matching their class.

> *A university's "AI Teaching Assistant" loads the adapter for CS 301 when a student asks about database normalization, and switches to the CS 201 adapter for data structures questions.*

### 7. Compliance & Legal

Legal and compliance teams convert regulation documents, internal policies, and contractual language into adapters. The resulting LLM has the **exact text of policies baked into its weights** — not retrieved from a vector database where chunking and retrieval errors can cause critical misquotes.

> *A compliance officer asks "what are our data retention requirements for EU customers?" and gets an answer grounded in the actual policy text, not a paraphrased retrieval result.*

### 8. Multi-Tenant SaaS Knowledge

SaaS platforms where each customer has their own knowledge base can generate **per-tenant adapters**. One base model serves all tenants; the correct adapter is loaded dynamically per API request. This is dramatically more efficient than running separate fine-tuned models per tenant.

> *A customer success platform with 200 enterprise clients maintains 200 knowledge bases. llm-patch generates 200 adapters (~1 GB total). One base model + dynamic adapter loading serves all clients.*

### 9. CI/CD-Driven Model Updates

Integrate llm-patch into your CI/CD pipeline. Every documentation commit triggers adapter regeneration as a pipeline step. The freshest adapters are deployed alongside your application code. **Your model's knowledge is as current as your last merged pull request.**

> *A GitHub Action runs `llm-patch compile` on every push to `docs/`. The resulting adapters are uploaded to S3 and hot-loaded by the production inference server.*

### 10. Competitive Intelligence

Product and strategy teams monitor competitor documentation, changelogs, and public knowledge bases. llm-patch watches these sources and generates adapters that keep an analysis LLM **perpetually current on the competitive landscape** — no manual re-ingestion or periodic retraining required.

> *A product manager asks "what features did Competitor X release this quarter?" and gets answers grounded in the competitor's actual documentation, updated automatically as their docs change.*

### 11. Domain-Specific Copilots

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
    CompilePipeline,
    MarkdownDataSource,
    GeneratorConfig,
    StorageConfig,
)
from llm_patch.generators.sakana_t2l import SakanaT2LGenerator
from llm_patch.storage.local_safetensors import LocalSafetensorsRepository

# 1. Set up the source, generator, and storage
source = MarkdownDataSource(directory="./api_docs")
generator = SakanaT2LGenerator(
    GeneratorConfig(checkpoint_dir="./models/gemma_2b_t2l")
)
repository = LocalSafetensorsRepository(
    StorageConfig(output_dir="./compiled_adapters")
)

# 2. Compile all documents into adapter weights
pipeline = CompilePipeline(source=source, generator=generator, repository=repository)
manifests = pipeline.compile_all()
for m in manifests:
    print(f"  {m.adapter_id} → {m.storage_uri}")
```

### Loading and Using Adapters

```python
from llm_patch.pipelines import UsePipeline
from llm_patch.attach import HFModelProvider, PeftAdapterLoader
from llm_patch.runtime.agent import PeftAgentRuntime
from llm_patch.runtime.session import ChatSession

# Load model and attach all compiled adapters
use = UsePipeline(
    model_provider=HFModelProvider(),
    adapter_loader=PeftAdapterLoader(),
    repository=repository,
)
agent = use.build_agent("google/gemma-2-2b-it")

# Single generation
print(agent.generate("Explain the API v2 authentication flow"))

# Interactive chat
session = ChatSession(agent, system_prompt="You are a helpful API expert.")
print(session.say("How do I authenticate with OAuth2?"))
```

### CLI

```bash
# Inspect sources
llm-patch source list --kind markdown --path ./api_docs

# Compile adapters
llm-patch adapter compile --source-dir ./api_docs --output-dir ./adapters --checkpoint-dir ./t2l

# Generate text with a patched model
llm-patch model generate --model-id google/gemma-2-2b-it --adapter-dir ./adapters "Explain OAuth2"

# Interactive chat
llm-patch model chat --model-id google/gemma-2-2b-it --adapter-dir ./adapters

# Start HTTP API server
llm-patch serve --model-id google/gemma-2-2b-it --adapter-dir ./adapters
```

---

## Architecture

The library is architected as a generic **Ingest → Compile → Attach → Use** pipeline with pluggable interfaces at every layer. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design deep-dive.

```
┌─────────────────────────────────────────────────────────────┐
│                      Pipelines Layer                         │
├──────────────────┬──────────────────┬───────────────────────┤
│  CompilePipeline │   UsePipeline    │    WikiPipeline       │
│  (ingest→store)  │  (load→serve)    │   (wiki→compile)      │
├──────────────────┴──────────────────┴───────────────────────┤
│                      Core Interfaces                         │
├──────────┬────────────┬──────────┬──────────┬───────────────┤
│IDataSource│IWeight    │IModel    │IAdapter  │IAgentRuntime  │
│IKnowledge│ Generator  │Provider  │Loader    │               │
│ Stream   │            │          │          │               │
├──────────┼────────────┼──────────┼──────────┼───────────────┤
│Markdown  │SakanaT2L   │HFModel   │Peft     │PeftAgent      │
│Wiki, PDF │Generator   │Provider  │Adapter  │Runtime +      │
│JSONL,HTTP│            │          │Loader   │ChatSession    │
│Composite │            │          │         │               │
└──────────┴────────────┴──────────┴──────────┴───────────────┘
```

### Design Patterns

- **Pipeline Composition** — `CompilePipeline`, `UsePipeline`, and `WikiPipeline` compose interfaces into end-to-end workflows.
- **Strategy Pattern** — Swap weight generation backends by implementing `IWeightGenerator`.
- **Data Source Pattern** — `IDataSource` (pull/batch) and `IKnowledgeStream` (push/live). Compose with `CompositeDataSource`.
- **Repository Pattern** — `IAdapterRepository` decouples adapter persistence from business logic.
- **Provider Pattern** — `IModelProvider` + `IAdapterLoader` abstract model loading and adapter attachment.
- **Runtime Pattern** — `IAgentRuntime` provides generate/chat/stream over any patched model.

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
| `ModelSpec` | `model_id` | — | HuggingFace model ID or local path |
| | `dtype` | `"auto"` | Model dtype (auto, float16, bfloat16) |
| | `device_map` | `"auto"` | Device mapping for model sharding |
| `AttachConfig` | `model` | — | `ModelSpec` for base model |
| | `adapter_ids` | `[]` | Adapter IDs to attach (empty = all) |
| `AgentConfig` | `system_prompt` | `None` | System instruction for chat |
| | `max_history` | `0` | Chat history limit (0 = unlimited) |
| `ServerConfig` | `host` | `"127.0.0.1"` | Server bind host |
| | `port` | `8000` | Server bind port |

---

## Extending

Build custom backends by implementing the core interfaces:

```python
from llm_patch.core.interfaces import (
    IDataSource, IKnowledgeStream,  # Data ingestion
    IWeightGenerator,                # Weight generation
    IAdapterRepository,              # Adapter storage
    IModelProvider,                  # Model loading
    IAdapterLoader,                  # Adapter attachment
    IAgentRuntime,                   # Inference / chat
)
```

**Custom data sources** — Ingest from databases, APIs, CMS platforms, or any structured text source by implementing `IDataSource`. For live monitoring, also implement `IKnowledgeStream`. Compose multiple sources with `CompositeDataSource`.

**Custom generators** — Integrate alternative hypernetworks, distillation methods, or future text-to-weight approaches by implementing `IWeightGenerator`.

**Custom storage** — Write adapters to S3, Google Cloud Storage, HuggingFace Hub, or any remote storage by implementing `IAdapterRepository`.

**Custom model providers** — Load models from custom registries or specialized frameworks by implementing `IModelProvider`.

**Custom runtimes** — Build alternative inference backends (vLLM, TGI, GGML) by implementing `IAgentRuntime`.

---

## End-to-End Example

The `examples/` directory contains a complete tutorial that chains an [LLM Wiki Agent](https://github.com/SamurAIGPT/llm-wiki-agent) with llm-patch to build domain-specialized LoRA adapters from ML research papers:

```
examples/data/papers/*.md ──► LLM Wiki Agent ──► wiki/ ──► WikiKnowledgeSource ──► adapters/
```

### Quick Demo (No GPU Required)

```bash
cd examples
python run_e2e.py --clean --aggregate
```

This runs the full pipeline end-to-end using mock components — ingesting sample papers, simulating wiki generation, and producing adapter manifests. See [examples/README.md](examples/README.md) for the full tutorial, including batch mode, watch mode, wikilink aggregation, and adapter validation.

### Before/After Knowledge Comparison (Gemini)

A live comparison using `gemini/gemini-2.0-flash` on 2 ML research papers demonstrated:

| Metric | Before (raw LLM) | After (wiki-enhanced) |
|---|---|---|
| Answer length | 642 chars | 1,848 chars (+188%) |
| Domain terms | 3/5 | 5/5 |
| Citations | 0 | 6 wiki pages |
| Math formulas | No | Yes |

The wiki-enhanced answer includes specific formulas (attention equation, LoRA decomposition), architectural details (encoder/decoder sub-layers, residual connections), and traceable citations to wiki pages. See [docs/E2E_WALKTHROUGH.md](docs/E2E_WALKTHROUGH.md) for full results and insights.

---

## Roadmap

llm-patch is under active development. Here's what's been done and what's planned:

- [x] **Generic data source framework** — `IDataSource` / `IKnowledgeStream` with Markdown, Wiki, PDF, JSONL, HTTP API, and Composite sources
- [x] **Adapter attachment pipeline** — `IModelProvider` + `IAdapterLoader` with HuggingFace + PEFT support
- [x] **Agent runtime** — `IAgentRuntime` with generate/chat/stream and `ChatSession` history management
- [x] **Adapter merging** — `merge_into_base()` and `weighted_blend()` for combining adapters
- [x] **CLI surface** — `source`, `model`, `adapter`, `wiki`, `serve` command groups
- [x] **REST API server** — FastAPI server with adapter management and inference endpoints
- [ ] **Cloud storage backends** — S3, GCS, and Azure Blob adapters for `IAdapterRepository`
- [ ] **Multi-model support** — Extend beyond Gemma to LLaMA, Mistral, Phi, and other architectures
- [ ] **HuggingFace Hub integration** — Push/pull adapters directly from the Hub
- [ ] **Incremental updates** — Delta adapter generation when documents are partially updated
- [ ] **Adapter quality scoring** — Automated evaluation of generated adapter quality
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

---

## Community & Links

- **[Discussions](https://github.com/yondonfu/llm-patch/discussions)** — questions, ideas, show-and-tell.
- **[Issues](https://github.com/yondonfu/llm-patch/issues)** — bug reports, feature requests, plugin proposals (use templates).
- **[docs/COMMUNITY.md](docs/COMMUNITY.md)** — full community channel reference, plugin gallery, and roadmap labels.
- **[CONTRIBUTING.md](CONTRIBUTING.md)** — contribution guide, code standards, and PR checklist.
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** — community standards.

### Related Projects

- **[Sakana AI — Text-to-LoRA](https://github.com/SakanaAI/text-to-lora)** — The hypernetwork backend that powers llm-patch's weight generation
- **[PEFT](https://github.com/huggingface/peft)** — Parameter-Efficient Fine-Tuning library for loading and managing LoRA adapters
- **[safetensors](https://github.com/huggingface/safetensors)** — Safe, fast tensor serialization format used for adapter storage
- **[llm-wiki-agent](https://github.com/SamurAIGPT/llm-wiki-agent)** — LLM-powered wiki generation agent that pairs naturally with llm-patch
- **[Karpathy's LLM Wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)** — The pattern for building persistent, compounding knowledge bases with LLMs — a natural knowledge source for llm-patch
- **[danvega/karpathy-wiki](https://github.com/danvega/karpathy-wiki)** — Community implementation of the LLM Wiki pattern

### Papers

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

---

## Documentation

| Document | Description |
|---|---|
| [README.md](README.md) | Project overview, use cases, and quickstart |
| [docs/QUICKSTART.md](docs/QUICKSTART.md) | 5-minute CLI-first walkthrough (`init` → `compile` → `chat`) |
| [docs/USAGE.md](docs/USAGE.md) | CLI reference + publishing/consuming adapters (Python API in appendix) |
| [docs/EXTENDING.md](docs/EXTENDING.md) | Authoring sources, generators, and registry-client plugins |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, design patterns, and data flow |
| [docs/E2E_WALKTHROUGH.md](docs/E2E_WALKTHROUGH.md) | Step-by-step pipeline walkthrough with Gemini comparison results |
| [docs/AGENTIC_AI_INTEGRATION.md](docs/AGENTIC_AI_INTEGRATION.md) | Adapter Market use case — sequence + Status Matrix |
| [docs/REGISTRY_PROTOCOL.md](docs/REGISTRY_PROTOCOL.md) | Wire format for any llm-patch–compatible adapter hub |
| [docs/SERVER_ARCHITECTURE.md](docs/SERVER_ARCHITECTURE.md) | Hot-swap server concurrency, eviction, and failure mapping |
| [docs/COMMUNITY.md](docs/COMMUNITY.md) | Community channels, plugin gallery, and roadmap labels |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Versioning promise, frozen v1 surface, deprecations slated for v2.0 |
| [docs/LIMITATIONS.md](docs/LIMITATIONS.md) | Non-goals, platform support, and known limits |
| [CHANGELOG.md](CHANGELOG.md) | Version history and release notes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute, code standards, and PR checklist |
| [examples/quickstart/README.md](examples/quickstart/README.md) | Demo notes + run script powering the 60-second quickstart |
| [examples/README.md](examples/README.md) | End-to-end tutorial with example scripts |

---

## License

Apache-2.0 — See [LICENSE](LICENSE) for the full text.

Free to use, modify, and distribute for any purpose, including commercial use.
