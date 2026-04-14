# Architecture

This document describes the internal architecture of **llm-patch**, the design decisions behind it, and how the components fit together. It is intended for contributors and developers who want to understand the system deeply or extend it with custom implementations.

---

## System Overview

llm-patch converts text documents into LoRA adapter weights through a three-layer pipeline coordinated by a central orchestrator. The system is designed around SOLID principles and well-known design patterns to maximize extensibility, testability, and separation of concerns.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User / Application                         │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│              KnowledgeFusionOrchestrator  (Facade)                  │
│                                                                     │
│  Coordinates the full pipeline:                                     │
│  • compile_all() — batch process all existing documents             │
│  • start()/stop() — begin/end live file watching                    │
│  • process_document() — single document → adapter                   │
├──────────────────┬──────────────────┬───────────────────────────────┤
│  IKnowledgeSource│ IWeightGenerator │    IAdapterRepository         │
│  (Observer)      │ (Strategy)       │    (Repository)               │
├──────────────────┼──────────────────┼───────────────────────────────┤
│ MarkdownDir      │ SakanaT2L        │    LocalSafetensors           │
│ Watcher          │ Generator        │    Repository                 │
│                  │                  │                               │
│ WikiKnowledge    │ (Future:         │    (Future:                   │
│ Source           │  custom backends)│     S3, GCS, Hub)             │
└──────────────────┴──────────────────┴───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Output: Adapter Directory                       │
│                                                                     │
│  {adapter_id}/                                                      │
│  ├── adapter_model.safetensors    (LoRA weight tensors)             │
│  ├── adapter_config.json          (PEFT LoraConfig)                 │
│  └── manifest.json                (Generation metadata)             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Design Patterns

### Facade — `KnowledgeFusionOrchestrator`

The orchestrator is the single entry point for users. It hides the complexity of coordinating three independent layers behind a simple API:

```python
orchestrator = KnowledgeFusionOrchestrator(source, generator, repository)
manifests = orchestrator.compile_all()
```

Users never need to call the source, generator, or repository directly. The orchestrator also acts as a context manager for live watching (`with orchestrator: ...`).

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

**Extension point:** Implement this interface to plug in custom hypernetworks, distillation-based generators, or any future text-to-weight approach. The orchestrator and storage layers remain untouched.

### Observer — `IKnowledgeSource`

Knowledge sources observe a data source (filesystem, API, database) and notify the orchestrator when documents change:

```python
class IKnowledgeSource(abc.ABC):
    @abc.abstractmethod
    def register_callback(self, callback: Callable[[DocumentContext], None]) -> None: ...

    @abc.abstractmethod
    def start(self) -> None: ...

    @abc.abstractmethod
    def stop(self) -> None: ...

    @abc.abstractmethod
    def scan_existing(self) -> list[DocumentContext]: ...
```

**Current implementations:**
- `MarkdownDirectoryWatcher` — Uses `watchdog` for filesystem monitoring with configurable glob patterns and debouncing.
- `WikiKnowledgeSource` — Extends markdown watching with YAML frontmatter parsing, `[[wikilink]]` extraction, and optional cross-page aggregation via `WikiDocumentAggregator`.

**Extension point:** Implement this interface for Confluence, Notion, database tables, RSS feeds, or any structured text source.

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

All domain models are defined as immutable Pydantic models in `core/models.py`:

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

---

## Configuration

Configuration is managed through Pydantic models in `core/config.py`:

| Model | Purpose | Key Fields |
|---|---|---|
| `GeneratorConfig` | T2L hypernetwork settings | `checkpoint_dir`, `device` |
| `WatcherConfig` | Directory monitoring settings | `directory`, `patterns`, `recursive`, `debounce_seconds` |
| `StorageConfig` | Adapter output settings | `output_dir` |

All configs support environment variable overrides and validation via Pydantic.

---

## Data Flow

### Batch Mode (`compile_all`)

```
1. orchestrator.compile_all()
2. → source.scan_existing()               # Glob for all matching files
3. → For each DocumentContext:
4.   → generator.generate(context)         # Embed text → hypernetwork → LoRA weights
5.   → repository.save(id, weights, cfg)   # Write safetensors + config + manifest
6. → Return list[AdapterManifest]
```

### Watch Mode (`start` / context manager)

```
1. orchestrator.start()  (or `with orchestrator:`)
2. → source.start()                        # Begin filesystem monitoring
3. → On file change detected:
4.   → source callback fires with DocumentContext
5.   → orchestrator._on_document_changed(context)
6.   → generator.generate(context)
7.   → repository.save(id, weights, cfg)
8. → orchestrator.stop()                   # On exit / Ctrl-C
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

## Directory Structure

```
llm-patch/
├── src/
│   └── llm_patch/
│       ├── __init__.py              # Public API exports
│       ├── py.typed                 # PEP 561 typed package marker
│       ├── orchestrator.py          # KnowledgeFusionOrchestrator (Facade)
│       ├── core/
│       │   ├── __init__.py
│       │   ├── interfaces.py        # IWeightGenerator, IKnowledgeSource, IAdapterRepository
│       │   ├── models.py            # DocumentContext, AdapterManifest (Pydantic)
│       │   └── config.py            # GeneratorConfig, WatcherConfig, StorageConfig
│       ├── generators/
│       │   ├── __init__.py
│       │   └── sakana_t2l.py        # SakanaT2LGenerator (Strategy)
│       ├── sources/
│       │   ├── __init__.py
│       │   ├── markdown_watcher.py  # MarkdownDirectoryWatcher (Observer)
│       │   └── wiki_source.py       # WikiKnowledgeSource + WikiDocumentAggregator
│       └── storage/
│           ├── __init__.py
│           └── local_safetensors.py # LocalSafetensorsRepository (Repository)
├── tests/
│   ├── conftest.py                  # Shared fixtures
│   ├── unit/                        # Fast, isolated tests with mocks
│   │   ├── test_generator.py
│   │   ├── test_models.py
│   │   ├── test_orchestrator.py
│   │   ├── test_sources.py
│   │   └── test_storage.py
│   └── integration/                 # End-to-end pipeline tests
│       ├── test_pipeline.py
│       └── test_wiki_pipeline.py
├── examples/
│   ├── README.md                    # Tutorial documentation
│   ├── research_pipeline.py         # Batch/watch mode CLI
│   ├── run_e2e.py                   # Full demo (no GPU required)
│   ├── validate_adapter.py          # Inference validation (GPU required)
│   └── raw/papers/                  # Sample ML paper summaries
├── docs/
│   ├── ARCHITECTURE.md              # This file
│   └── USAGE.md                     # Usage guide
├── pyproject.toml                   # PEP 621 project config, tool settings
├── Makefile                         # Development commands
├── README.md                        # Project overview and use cases
├── CONTRIBUTING.md                  # Contribution guidelines
├── LICENSE                          # Apache-2.0
├── NOTES.md                         # Design notes and decisions
└── .pre-commit-config.yaml          # Pre-commit hooks (Ruff, formatting)
```

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
| `hyper_llm_modulator` | Sakana AI T2L hypernetwork (external) | — |

---

## Testing Strategy

### Unit Tests

- Test each layer in isolation using mocks
- Verify the orchestrator calls the correct methods in the correct order
- Validate tensor shapes and adapter config correctness
- Test edge cases (empty documents, missing files, duplicate IDs)

### Integration Tests

- Test the full pipeline: source → generator → repository
- Use `MockWeightGenerator` and `MockAdapterRepository` for GPU-free testing
- Verify filesystem interactions (file creation, manifest writing)
- Test watch mode with synthetic file events

### Running Tests

```bash
make test          # All tests with coverage
make test-unit     # Unit tests only
make test-fast     # Quick run, stop on first failure
```

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
