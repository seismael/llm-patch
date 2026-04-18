# Architecture

This document describes the internal architecture of **llm-patch**, the design decisions behind it, and how the components fit together. It is intended for contributors and developers who want to understand the system deeply or extend it with custom implementations.

---

## System Overview

llm-patch is a generic **Ingest → Compile → Attach → Use** framework that converts text documents into LoRA adapter weights, attaches them to any HuggingFace model, and serves the patched model for inference. The system is built around SOLID principles and clean interfaces to maximize extensibility.

```
┌─────────────────────────────────────────────────────────────────────┐
│                      User / Application / CLI                       │
└───────────────┬──────────────────┬──────────────────┬───────────────┘
                │                  │                  │
                ▼                  ▼                  ▼
     ┌──────────────────┐ ┌──────────────┐  ┌──────────────────┐
     │  CompilePipeline │ │  UsePipeline │  │   WikiPipeline   │
     │  (ingest→store)  │ │ (load→serve) │  │  (wiki→compile)  │
     └──────┬───────────┘ └───────┬──────┘  └──────────────────┘
            │                     │
     ┌──────┴──────┐       ┌─────┴──────┐
     │             │       │            │
     ▼             ▼       ▼            ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐
│IDataSource│ │IWeight   │ │IModel    │ │IAdapter      │
│           │ │Generator │ │Provider  │ │Loader        │
├──────────┤ ├──────────┤ ├──────────┤ ├──────────────┤
│Markdown  │ │SakanaT2L │ │HFModel   │ │PeftAdapter   │
│Wiki      │ │Generator │ │Provider  │ │Loader        │
│PDF       │ │          │ │          │ │              │
│JSONL     │ └──────────┘ └──────────┘ └──────────────┘
│HTTP API  │
│Composite │       ┌──────────────┐        ┌──────────────┐
└──────────┘       │IAdapter      │        │IAgentRuntime │
                   │Repository    │        ├──────────────┤
                   ├──────────────┤        │PeftAgent     │
                   │LocalSafe     │        │Runtime       │
                   │tensors       │        │ + ChatSession│
                   └──────────────┘        └──────────────┘
```

---

## Design Patterns

### Pipeline Composition — `CompilePipeline`, `UsePipeline`, `WikiPipeline`

Pipelines compose the core interfaces into end-to-end workflows:

- **`CompilePipeline`** — binds `IDataSource` → `IWeightGenerator` → `IAdapterRepository`. Supports batch (`compile_all`) and live stream (`IKnowledgeStream`) compilation.
- **`UsePipeline`** — binds `IModelProvider` → `IAdapterLoader` → `IAdapterRepository`. Loads a base model, attaches adapters, and optionally wraps in a `PeftAgentRuntime`.
- **`WikiPipeline`** — bridges `WikiManager` with an optional `CompilePipeline` for the wiki → adapter closed loop.

```python
from llm_patch.pipelines import CompilePipeline, UsePipeline

compile_pl = CompilePipeline(source=my_source, generator=gen, repository=repo)
manifests = compile_pl.compile_all()

use_pl = UsePipeline(model_provider=provider, adapter_loader=loader, repository=repo)
agent = use_pl.build_agent("google/gemma-2-2b-it")
```

### Strategy — `IWeightGenerator`

The weight generation backend is interchangeable. The `IWeightGenerator` interface defines a single contract:

```python
class IWeightGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, context: DocumentContext) -> dict[str, torch.Tensor]: ...

    @abc.abstractmethod
    def get_peft_config(self) -> Any: ...
```

**Current implementation:** `SakanaT2LGenerator` wraps the `hyper_llm_modulator` library from Sakana AI.

### Data Source — `IDataSource` / `IKnowledgeStream`

Data sources implement either pull-based (`IDataSource`) or push-based (`IKnowledgeStream`) ingestion:

```python
class IDataSource(abc.ABC):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def fetch_all(self) -> Iterable[DocumentContext]: ...

    def fetch_one(self, document_id: str) -> DocumentContext | None: ...
```

```python
class IKnowledgeStream(abc.ABC):
    @abc.abstractmethod
    def subscribe(self, callback: Callable[[DocumentContext], None]) -> None: ...

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...
```

**Current implementations:**
- `MarkdownDataSource` / `MarkdownWatcher` — Markdown directory batch and live monitoring.
- `WikiDataSource` / `WikiWatcher` — Wiki-structured markdown with frontmatter and wikilinks.
- `PdfDataSource` — PDF directory ingestion via `pypdf`.
- `JsonlDataSource` — JSONL file ingestion.
- `HttpApiDataSource` — REST API document fetching via `httpx`.
- `CompositeDataSource` — Merge multiple `IDataSource` implementations with ID namespacing.

### Model Loading & Adapter Attachment

```python
class IModelProvider(abc.ABC):
    @abc.abstractmethod
    def load(self, model_id: str, **kwargs) -> ModelHandle: ...

class IAdapterLoader(abc.ABC):
    @abc.abstractmethod
    def attach(self, handle: ModelHandle, manifest: AdapterManifest) -> ModelHandle: ...
```

**Current implementations:** `HFModelProvider` (transformers), `PeftAdapterLoader` (PEFT).

### Agent Runtime — `IAgentRuntime`

```python
class IAgentRuntime(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abc.abstractmethod
    def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse: ...

    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]: ...
```

**Current implementation:** `PeftAgentRuntime` wraps a `ModelHandle` with tokenize → generate → decode.
`ChatSession` manages conversation history, system prompts, and history trimming.

### Repository — `IAdapterRepository`

Adapter persistence is abstracted behind a clean CRUD interface:

```python
class IAdapterRepository(abc.ABC):
    @abc.abstractmethod
    def save(self, adapter_id: str, weights: dict[str, torch.Tensor], peft_config: Any) -> AdapterManifest: ...

    @abc.abstractmethod
    def load(self, adapter_id: str) -> dict[str, torch.Tensor]: ...

    @abc.abstractmethod
    def exists(self, adapter_id: str) -> bool: ...

    @abc.abstractmethod
    def list_adapters(self) -> list[AdapterManifest]: ...

    @abc.abstractmethod
    def delete(self, adapter_id: str) -> None: ...
```

**Current implementation:** `LocalSafetensorsRepository` — Writes to the local filesystem using the `safetensors` format.

**Extension point:** Implement this interface for S3, Google Cloud Storage, Azure Blob, HuggingFace Hub, or any persistent storage backend.

---

## Domain Models

All domain models are defined as Pydantic models in `core/models.py`:

### `DocumentContext`

Represents an ingested document ready for weight generation:

| Field | Type | Description |
|---|---|---|
| `document_id` | `str` | Unique identifier (derived from filename stem) |
| `content` | `str` | Raw text content of the document |
| `metadata` | `dict[str, Any]` | Arbitrary metadata (source path, modification time, frontmatter, wikilinks) |

### `AdapterManifest`

Tracks a generated adapter and its location:

| Field | Type | Description |
|---|---|---|
| `adapter_id` | `str` | Matches the source `document_id` |
| `rank` | `int` | LoRA rank (r parameter) |
| `target_modules` | `list[str]` | Model layers the adapter affects |
| `storage_uri` | `str` | Path or URI to the stored adapter directory |
| `created_at` | `datetime` | UTC timestamp of generation |

### `ModelHandle`

Wraps a loaded model + tokenizer for use by the attach and runtime layers:

| Field | Type | Description |
|---|---|---|
| `model` | `Any` | The loaded model object (transformers `PreTrainedModel`) |
| `tokenizer` | `Any` | The loaded tokenizer object |
| `model_id` | `str` | Model identifier |
| `active_adapters` | `list[str]` | List of currently active adapter IDs |

### `ChatMessage` / `ChatResponse`

For the agent runtime chat interface:

| Field | Type | Description |
|---|---|---|
| `ChatMessage.role` | `ChatRole` | `system`, `user`, or `assistant` |
| `ChatMessage.content` | `str` | Message text |
| `ChatResponse.message` | `ChatMessage` | The assistant's reply |

### `GenerationOptions`

Controls text generation parameters:

| Field | Type | Default |
|---|---|---|
| `max_new_tokens` | `int` | `256` |
| `temperature` | `float` | `0.7` |
| `top_p` | `float` | `0.9` |
| `top_k` | `int` | `50` |
| `do_sample` | `bool` | `True` |
| `repetition_penalty` | `float` | `1.0` |

---

## Configuration

Configuration is managed through Pydantic models in `core/config.py`:

| Model | Purpose | Key Fields |
|---|---|---|
| `GeneratorConfig` | T2L hypernetwork settings | `checkpoint_dir`, `device` |
| `WatcherConfig` | Directory monitoring settings | `directory`, `patterns`, `recursive`, `debounce_seconds` |
| `StorageConfig` | Adapter output settings | `output_dir` |
| `WikiConfig` | Wiki workspace settings | `base_dir`, `schema_path` |
| `ModelSpec` | Base model specification | `model_id`, `dtype`, `device_map`, `trust_remote_code` |
| `AttachConfig` | Adapter attachment settings | `model`, `adapter_ids` |
| `AgentConfig` | Agent runtime settings | `model`, `attach`, `system_prompt`, `max_history`, `generation` |
| `ServerConfig` | HTTP server settings | `host`, `port`, `reload`, `cors_origins` |
| `MarkdownSourceConfig` | Markdown source params | `kind="markdown"`, `directory`, `patterns`, `recursive` |
| `WikiSourceConfig` | Wiki source params | `kind="wiki"`, `directory`, `aggregate` |
| `PdfSourceConfig` | PDF source params | `kind="pdf"`, `directory` |
| `JsonlSourceConfig` | JSONL source params | `kind="jsonl"`, `path`, `text_field`, `id_field` |
| `HttpSourceConfig` | HTTP API source params | `kind="http"`, `base_url`, `documents_endpoint` |

`DataSourceConfig` is a discriminated union type: `MarkdownSourceConfig | WikiSourceConfig | PdfSourceConfig | JsonlSourceConfig | HttpSourceConfig`.

All configs support environment variable overrides and validation via Pydantic.

---

## Data Flow

### Compile Pipeline — Batch Mode (`compile_all`)

```
1. compile_pipeline.compile_all()
2. → source.fetch_all()                   # Pull all documents from IDataSource
3. → For each DocumentContext:
4.   → generator.generate(context)         # Embed text → hypernetwork → LoRA weights
5.   → repository.save(id, weights, cfg)   # Write safetensors + config + manifest
6. → Return list[AdapterManifest]
```

### Compile Pipeline — Live Stream Mode

```
1. compile_pipeline(source=..., stream=watcher)
2. → stream.subscribe(on_document_changed)
3. → stream.start()                        # Begin filesystem monitoring
4. → On file change detected:
5.   → callback fires with DocumentContext
6.   → generator.generate(context)
7.   → repository.save(id, weights, cfg)
8. → compile_pipeline.stop()               # On exit / Ctrl-C
```

### Use Pipeline — Load → Attach → Infer

```
1. use_pipeline.load_and_attach(model_id, adapter_ids)
2. → model_provider.load(model_id)         # Load base HF model
3. → For each adapter:
4.   → adapter_loader.attach(handle, manifest)  # Attach via PEFT
5. → Return ModelHandle
6.
7. runtime = PeftAgentRuntime(handle)
8. response = runtime.generate(prompt)      # Tokenize → generate → decode
9. (or) response = runtime.chat(messages)   # Apply chat template → generate
```

### Weight Generation Pipeline (Inside `SakanaT2LGenerator`)

```
1. Document text input
2. → Sentence embedding model (e.g., all-MiniLM-L6-v2)
3. → Dense vector representation (768-d)
4. → Task encoder (transformer block)
5. → Hypernetwork forward pass
6. → LoRA A/B weight matrices for all target model layers
7. → State dict: {"base_model.model.layers.{i}.self_attn.q_proj.lora_A.weight": tensor, ...}
```

### Wiki Aggregation Pipeline (Inside `WikiDocumentAggregator`)

```
1. Source page parsed (frontmatter + body + wikilinks)
2. → Extract [[wikilink]] references
3. → Resolve links to entity/concept page files
4. → Read and concatenate linked page content
5. → Return enriched DocumentContext with aggregated text
```

---

## Obsidian Vault Integration

The wiki directory can optionally function as an [Obsidian](https://obsidian.md) vault, enabling a powerful visual workflow: **Obsidian is the IDE; the LLM is the programmer; the wiki is the codebase** (per the [Karpathy LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)).

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Obsidian (Viewer)                           │
│  Graph view · Page browsing · Web Clipper · Dataview queries        │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ reads .md files
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Wiki Directory (= Vault Root)                    │
│                                                                     │
│  .obsidian/                                                         │
│  ├── app.json              (attachmentFolderPath, userIgnoreFilters) │
│  ├── appearance.json       (theme settings)                         │
│  └── community-plugins.json                                         │
│                                                                     │
│  raw/        ← immutable sources (ignored in graph view)            │
│  wiki/       ← LLM-generated markdown (visible in graph view)      │
│    ├── summaries/, concepts/, entities/, syntheses/, journal/        │
│    ├── index.md, log.md                                             │
│    └── (all pages with YAML frontmatter + [[wikilinks]])            │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Module | Responsibility |
|---|---|---|
| `ObsidianConfig` | `wiki/obsidian.py` | Pydantic model for vault settings (attachment folder, ignore filters, Dataview toggle) |
| `ObsidianVault` | `wiki/obsidian.py` | Vault detection, `.obsidian/` initialization, config management, graph export |
| `GraphData` / `GraphNode` / `GraphEdge` | `wiki/obsidian.py` | Lightweight graph snapshot derived from wiki page `[[wikilinks]]` |

### How It Connects

- **WikiManager** holds an optional `ObsidianVault` instance.  When `obsidian_enabled=True` in the schema (or `--obsidian` is passed to `init`), the vault is created alongside the wiki directories.
- **`userIgnoreFilters`** in `.obsidian/app.json` exclude `raw/`, `.claude/`, `.git/` etc. from the graph view — keeping it clean with only wiki content nodes.
- **`attachmentFolderPath`** directs Obsidian Web Clipper downloads to `raw/assets/` so images are version-controlled alongside sources.
- **Graph export** (`wiki.graph()` / `wiki.export_graph()`) builds a JSON representation of nodes and edges that can be used by external visualization tools or for LLM-driven analysis.
- **Dataview compatibility** — all wiki pages emit YAML frontmatter (title, type, tags, created, updated, confidence, sources) that the Obsidian Dataview plugin can query.

### CLI & MCP

The CLI exposes an `obsidian` subcommand group:

```bash
llm-patch wiki obsidian init          # Set up .obsidian/ config
llm-patch wiki obsidian graph -o g.json  # Export knowledge graph
llm-patch wiki obsidian status        # Show vault + graph metrics
llm-patch wiki init --obsidian        # Init wiki + vault in one step
```

The MCP server adds three tools: `obsidian_init`, `obsidian_graph`, `obsidian_status`.

---

## Directory Structure

```
llm-patch/
├── src/
│   └── llm_patch/
│       ├── __init__.py              # Public API exports
│       ├── py.typed                 # PEP 561 typed package marker
│       ├── orchestrator.py          # Legacy KnowledgeFusionOrchestrator shim
│       ├── wiki_pipeline.py         # Legacy WikiPipelineOrchestrator shim
│       ├── cli/
│       │   ├── __init__.py          # Top-level `llm-patch` CLI group
│       │   ├── wiki.py              # `llm-patch wiki` subcommands
│       │   ├── adapter.py           # `llm-patch adapter` subcommands (legacy)
│       │   ├── source.py            # `llm-patch source` subcommands
│       │   ├── model.py             # `llm-patch model` subcommands
│       │   └── serve.py             # `llm-patch serve` subcommand
│       ├── core/
│       │   ├── __init__.py
│       │   ├── interfaces.py        # IDataSource, IKnowledgeStream, IWeightGenerator,
│       │   │                        # IAdapterRepository, IModelProvider, IAdapterLoader,
│       │   │                        # IAgentRuntime
│       │   ├── models.py            # DocumentContext, AdapterManifest, ModelHandle,
│       │   │                        # ChatMessage, ChatResponse, GenerationOptions, etc.
│       │   └── config.py            # All Pydantic config models
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── compile.py           # CompilePipeline (ingest → generate → store)
│       │   ├── wiki.py              # WikiPipeline (wiki lifecycle + compile)
│       │   └── use.py               # UsePipeline (load → attach → agent)
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── markdown.py          # MarkdownDataSource, MarkdownWatcher
│       │   ├── wiki.py              # WikiDataSource, WikiWatcher, WikiDocumentAggregator
│       │   ├── pdf.py               # PdfDataSource (requires pypdf)
│       │   ├── jsonl.py             # JsonlDataSource
│       │   ├── http_api.py          # HttpApiDataSource (requires httpx)
│       │   ├── composite.py         # CompositeDataSource (multi-source merge)
│       │   ├── markdown_watcher.py  # Backward-compat re-export
│       │   └── wiki_source.py       # Backward-compat re-export
│       ├── generators/
│       │   ├── __init__.py
│       │   └── sakana_t2l.py        # SakanaT2LGenerator (Strategy)
│       ├── attach/
│       │   ├── __init__.py
│       │   ├── model_provider.py    # HFModelProvider (IModelProvider)
│       │   ├── peft_loader.py       # PeftAdapterLoader (IAdapterLoader)
│       │   └── merger.py            # merge_into_base(), weighted_blend()
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── agent.py             # PeftAgentRuntime (IAgentRuntime)
│       │   └── session.py           # ChatSession (conversation management)
│       ├── server/
│       │   ├── __init__.py
│       │   ├── app.py               # FastAPI application
│       │   └── schemas.py           # Request/response Pydantic schemas
│       ├── mcp/
│       │   ├── __init__.py
│       │   └── server.py            # MCP tool server
│       ├── wiki/
│       │   ├── __init__.py
│       │   ├── page.py, schema.py, index.py, log.py, linker.py
│       │   ├── operations.py, interfaces.py, manager.py, obsidian.py
│       │   └── agents/              # Wiki agent implementations
│       └── storage/
│           ├── __init__.py
│           └── local_safetensors.py # LocalSafetensorsRepository
├── tests/
│   ├── conftest.py
│   ├── unit/                        # Fast, isolated tests with mocks
│   └── integration/                 # End-to-end pipeline tests
├── examples/
│   └── ...                          # Example scripts and demo data
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   ├── E2E_WALKTHROUGH.md           # Step-by-step pipeline guide with results
│   └── USAGE.md                     # Usage guide
├── scripts/
│   └── run_gemini_comparison.py     # Before/after Gemini comparison script
├── pyproject.toml
├── Makefile
├── README.md
└── LICENSE

---

## Key Dependencies

| Package | Role | Version |
|---|---|---|
| `torch` | Tensor operations, model inference | ≥ 2.1 |
| `transformers` | Pre-trained model loading | ≥ 4.40 |
| `peft` | LoRA adapter management | ≥ 0.12 |
| `safetensors` | Safe, fast tensor serialization | ≥ 0.4 |
| `pydantic` | Configuration and data validation | ≥ 2.0 |
| `watchdog` | Cross-platform filesystem monitoring | ≥ 4.0 |
| `click` | CLI framework | ≥ 8.0 |
| `fastapi` | HTTP API server (optional `[server]` extra) | ≥ 0.100 |
| `uvicorn` | ASGI server (optional `[server]` extra) | ≥ 0.20 |
| `pypdf` | PDF source (optional `[pdf]` extra) | ≥ 4.0 |
| `httpx` | HTTP API source (optional `[http]` extra) | ≥ 0.24 |
| `hyper_llm_modulator` | Sakana AI T2L hypernetwork (external) | — |

---

## Testing Strategy

### Unit Tests (274 tests)

- Test each layer in isolation using mocks
- Verify the orchestrator calls the correct methods in the correct order
- Validate tensor shapes and adapter config correctness
- Test edge cases (empty documents, missing files, duplicate IDs)
- Wiki module: manager, index, linker, log, page, schema, obsidian, agents

### Integration Tests

- Test the full pipeline: source → generator → repository
- Use `MockWeightGenerator` and `MockAdapterRepository` for GPU-free testing
- Verify filesystem interactions (file creation, manifest writing)
- Test watch mode with synthetic file events
- E2E phases 1–10 covering the complete layer stack

### Running Tests

```bash
make test          # All tests with coverage
make test-unit     # Unit tests only
make test-fast     # Quick run, stop on first failure
```

---

## Layer Summary

The system is organized into dependency layers where each layer depends
only on layers below it. Layer 0 has zero internal dependencies.

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 5: Entry Points                                           │
│  CLI (click) · HTTP Server (FastAPI) · MCP Server                │
├──────────────────────────────────────────────────────────────────┤
│  Layer 4: Pipelines                                              │
│  CompilePipeline · UsePipeline · WikiPipeline                    │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: Concrete Implementations                               │
│  Sources: Markdown, Wiki, PDF, JSONL, HTTP, Composite            │
│  Generators: SakanaT2LGenerator                                  │
│  Attach: HFModelProvider, PeftAdapterLoader, Merger              │
│  Runtime: PeftAgentRuntime, ChatSession                          │
│  Storage: LocalSafetensorsRepository                             │
│  Wiki: WikiManager, Obsidian, Agents (Anthropic, LiteLLM, Mock) │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: Wiki Primitives                                        │
│  WikiPage · WikiIndex · WikiLinker · WikiLog · WikiSchema        │
│  Operations (IngestResult, QueryResult, LintReport)              │
├──────────────────────────────────────────────────────────────────┤
│  Layer 1: Domain Models & Config                                 │
│  DocumentContext · AdapterManifest · ModelHandle                  │
│  ChatMessage · ChatResponse · GenerationOptions                  │
│  GeneratorConfig · StorageConfig · WikiConfig · etc.             │
├──────────────────────────────────────────────────────────────────┤
│  Layer 0: Interfaces (zero dependencies)                         │
│  IDataSource · IKnowledgeStream · IWeightGenerator               │
│  IAdapterRepository · IModelProvider · IAdapterLoader            │
│  IAgentRuntime · IWikiAgent                                      │
└──────────────────────────────────────────────────────────────────┘
```

| Layer | Module(s) | Responsibility | Test Count |
|---|---|---|---|
| 0 | `core.interfaces`, `wiki.interfaces` | Abstract contracts; Dependency Inversion | Tested via implementations |
| 1 | `core.models`, `core.config` | Pydantic domain objects and configuration | 16 |
| 2 | `wiki.page`, `wiki.index`, `wiki.linker`, `wiki.log`, `wiki.schema` | Wiki primitives (parsing, indexing, linking) | 45+ |
| 3 | `sources.*`, `generators.*`, `attach.*`, `runtime.*`, `storage.*`, `wiki.manager`, `wiki.agents.*` | All concrete implementations | 150+ |
| 4 | `pipelines.*` | Pipeline composition and orchestration | 28+ |
| 5 | `cli`, `server`, `mcp` | User-facing entry points | 28+ |

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Pydantic for models** | Immutable, validated, serializable domain objects with minimal boilerplate |
| **ABC interfaces** | Enforce contracts at the type level; enable dependency injection and testing with mocks |
| **safetensors format** | HuggingFace standard; safe (no code execution), fast, cross-platform |
| **Watchdog for filesystem** | Battle-tested, cross-platform, supports debouncing natively |
| **Facade pattern** | Users interact with one class; complexity is hidden behind `KnowledgeFusionOrchestrator` |
| **document_id from filename** | Simple, deterministic, human-readable; avoids UUID complexity for local use |
| **No YAML dependency for frontmatter** | Regex-based parsing avoids adding PyYAML to core dependencies; available as optional extra |
| **py.typed marker** | PEP 561 compliance for downstream type checking |
