# Research Papers → Wiki → LoRA: End-to-End Tutorial

This example demonstrates the full **llm-patch** pipeline: ingesting structured
wiki pages (produced by an LLM Wiki Agent) and converting them into LoRA
adapter weights that can be applied to a base language model.

## Architecture

```
data/papers/*.md ──► LLM Wiki Agent ──► wiki/ (structured markdown)
                       (ingest)          │
                                          ▼
                                    llm-patch library
                                    ┌──────────────┐
                                    │ WikiSource   │ (IKnowledgeSource)
                                    │  watches     │
                                    │  wiki/       │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │ Weight       │ (IWeightGenerator)
                                    │ Generator    │
                                    └──────┬───────┘
                                           ▼
                                    ┌──────────────┐
                                    │ Adapter      │ (IAdapterRepository)
                                    │ Repository   │
                                    └──────┬───────┘
                                           ▼
                                    adapters/{id}/
```

## Layout

```
examples/
├── data/                 # Read-only sample corpora used by the demos
│   ├── papers/           # 3 sample ML paper summaries (raw input)
│   └── wiki/             # Pre-built wiki snapshot (sources/ + entities/)
├── e2e/                  # End-to-end demo scripts
│   ├── run_e2e.py            # Full pipeline: simulate wiki → generate → validate
│   ├── run_wiki_e2e.py       # Wiki-agent variant (mock or Anthropic)
│   ├── demo_e2e_scenario.py  # Scripted 5-step scenario for documentation/tests
│   ├── research_pipeline.py  # Core batch/watch pipeline implementation
│   └── validate_adapter.py   # GPU-only adapter validation script
└── quickstart/           # Smallest possible getting-started example
```

## Quick Start

### 1. Run the end-to-end demo (no GPU needed)

```bash
python examples/e2e/run_e2e.py --clean
```

This will:
1. Copy raw papers from `examples/data/papers/` into a simulated `wiki/` directory
2. Add wiki-style frontmatter and create entity stub pages
3. Run the WikiKnowledgeSource → MockWeightGenerator → MockAdapterRepository pipeline
4. Report all generated adapter manifests

### 2. Batch mode — process an existing wiki

```bash
python examples/e2e/research_pipeline.py batch \
    --wiki-dir examples/data/wiki/ --output-dir adapters/
```

### 3. Batch mode with wikilink aggregation

```bash
python examples/e2e/research_pipeline.py batch \
    --wiki-dir examples/data/wiki/ --aggregate
```

When `--aggregate` is enabled, each source page follows its `[[wikilinks]]`
to entity and concept pages, concatenating linked content into a single
enriched document before weight generation.

### 4. Watch mode — live monitoring

```bash
python examples/e2e/research_pipeline.py watch --wiki-dir wiki/
```

Add or modify wiki pages while the watcher is running. Each change triggers
automatic adapter generation. Press Ctrl-C to stop.

### 5. Validate adapters (GPU required)

```bash
python examples/e2e/validate_adapter.py \
    --adapter-dir adapters/sources/attention-paper \
    --base-model google/gemma-2-2b-it
```

This loads the base model, applies the LoRA adapter, and runs a side-by-side
inference comparison. Requires `torch`, `transformers`, and `peft` with a
working CUDA installation.

## Wiki Directory Structure

The LLM Wiki Agent (or the `run_e2e.py` simulator) produces:

```
wiki/
├── sources/              # One page per ingested paper
│   ├── attention-is-all-you-need.md
│   ├── lora-low-rank-adaptation.md
│   └── gpt3-few-shot-learners.md
├── entities/             # Auto-extracted entity pages
│   ├── transformer.md
│   ├── self-attention.md
│   └── ...
└── concepts/             # Concept pages
    ├── lora.md
    └── ...
```

Each page has YAML frontmatter:

```yaml
---
title: Attention Is All You Need
authors: Vaswani et al.
year: 2017
tags: transformer, attention
---
```

## WikiKnowledgeSource

`WikiKnowledgeSource` is a new `IKnowledgeSource` that:

- Watches wiki directories recursively for `.md` file changes
- Parses YAML frontmatter into `DocumentContext.metadata`
- Extracts `[[wikilinks]]` and stores them in metadata
- Optionally aggregates linked pages into enriched documents

```python
from llm_patch import WikiKnowledgeSource, WatcherConfig

config = WatcherConfig(directory="wiki/")
source = WikiKnowledgeSource(config, aggregate=True)

# Scan existing pages
docs = source.scan_existing()
for doc in docs:
    print(doc.document_id, doc.metadata.get("title"))

# Or use with the orchestrator for live monitoring
from llm_patch import KnowledgeFusionOrchestrator

orchestrator = KnowledgeFusionOrchestrator(source, generator, repository)
with orchestrator:
    # Watcher is running — changes trigger automatic processing
    ...
```

## Extending

- **Real weight generation**: Replace `MockWeightGenerator` with
  `SakanaT2LGenerator` when a Sakana T2L checkpoint is available.
- **Real storage**: Replace `MockAdapterRepository` with
  `LocalSafetensorsRepository` for persistent safetensors output.
- **LLM Wiki Agent**: Point `--wiki-dir` at the actual output of
  `SamurAIGPT/llm-wiki-agent` instead of using the simulator.
- **Adapter merging**: Use PEFT's `add_weighted_adapter()` to combine
  multiple per-document adapters into a single domain adapter.
