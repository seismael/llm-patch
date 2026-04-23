# llm-patch

> **Engine** — the generic Ingest → Compile → Attach → Use framework that converts text documents into LoRA adapter weights and attaches them to HuggingFace models.

This is the engine package of the [llm-patch monorepo](../../README.md).
For the high-level project overview, use cases, and quickstart, see the
[repository root README](../../README.md). This file documents the
engine package itself.

## Install

From the workspace root:

```pwsh
uv sync
```

Or as a standalone dependency in another project:

```toml
dependencies = ["llm-patch>=0.1"]
```

## Public API

The public surface is everything re-exported from
[`src/llm_patch/__init__.py`](src/llm_patch/__init__.py).
See [ADR-0003](../../docs/adr/0003-public-api-policy.md) for the policy.

```python
from llm_patch import (
    CompilePipeline, UsePipeline, WikiPipeline,
    MarkdownDataSource, WikiKnowledgeSource,
    SakanaT2LGenerator, LocalSafetensorsRepository,
    GeneratorConfig, StorageConfig, WatcherConfig,
)
```

## Tests

```pwsh
uv run --package llm-patch pytest -q
# expected baseline: 216 passed, 3 skipped
```

## Contracts

- Per-project agent contract: [AGENTS.md](AGENTS.md).
- Engine boundary rules: [.github/instructions/engine-boundary.instructions.md](../../.github/instructions/engine-boundary.instructions.md).
- Engineering specification: [SPEC.md](../../SPEC.md).
