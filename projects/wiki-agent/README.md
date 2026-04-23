# llm-patch-wiki-agent

Wiki-specialized agent built on the [llm-patch](../llm-patch) engine.

Compiles a wiki vault (Obsidian-style or Karpathy-style LLM Wiki) into LoRA
adapters via the engine's `WikiKnowledgeSource` + `SakanaT2LGenerator`,
attaches them to a base LLM, and serves a domain-expert chat agent.

> **Status**: initial v1 surface. The package now provides a real
> `WikiAgent` orchestration layer plus `compile`, `chat`, and `info`
> CLI commands. `serve` remains intentionally out of scope for this
> release.

## Prerequisites

- `compile` requires a Sakana Text-to-LoRA checkpoint directory and the
	`hyper_llm_modulator` runtime used by `SakanaT2LGenerator`.
- `chat` requires compiled adapters in the configured adapter directory
	plus a loadable HuggingFace-compatible base model.

## CLI

```pwsh
uv run llm-patch-wiki-agent info --adapter-dir .\artifacts\adapters
uv run llm-patch-wiki-agent compile --wiki-dir .\wiki --adapter-dir .\artifacts\adapters --checkpoint-dir .\models\t2l
uv run llm-patch-wiki-agent chat --adapter-dir .\artifacts\adapters --model-id google/gemma-2-2b-it "Summarize the wiki"
```

`chat` is intentionally a one-shot interaction in v1. Long-running HTTP
serving stays in the engine's server layer and a future dedicated use-case.

See the root [SPEC.md](../../SPEC.md) for the governing engineering
specification and [AGENTS.md](AGENTS.md) for the per-project agent contract.
