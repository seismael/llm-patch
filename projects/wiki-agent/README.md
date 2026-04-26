# llm-patch-wiki-agent

Wiki-specialized agent built on the [llm-patch](../llm-patch) engine.

Compiles a wiki vault (Obsidian-style or Karpathy-style LLM Wiki) into LoRA
adapters via the engine's `WikiKnowledgeSource` + `SakanaT2LGenerator`,
attaches them to a base LLM, and serves a domain-expert chat agent.

> **Status**: v1 ships the full **Living Wiki Copilot** use case
> (`NOTES.md`) — a continuous compile daemon, an HTTP inference gateway
> with metadata-based routing, and an MCP server exposing
> `internalize_knowledge`. All three are wiki-agent-specific glue;
> the engine and utils stay generic and untouched.

## Prerequisites

- `compile` / `daemon` require a Sakana Text-to-LoRA checkpoint directory
  and the `hyper_llm_modulator` runtime used by `SakanaT2LGenerator`.
- `chat` / `serve` / `mcp` require compiled adapters in the configured
  adapter directory plus a loadable HuggingFace-compatible base model.
- `serve` requires the optional `[server]` extra (`fastapi`, `uvicorn`).
- `mcp` requires the optional `[mcp]` extra.

## CLI

### Inspect & one-shot (always available)

```pwsh
uv run llm-patch-wiki-agent info --adapter-dir .\artifacts\adapters
uv run llm-patch-wiki-agent compile --wiki-dir .\wiki --adapter-dir .\artifacts\adapters --checkpoint-dir .\models\t2l
uv run llm-patch-wiki-agent chat --adapter-dir .\artifacts\adapters --model-id google/gemma-2-2b-it "Summarize the wiki"
```

### Phase 1 — continuous compile daemon

Compiles a wiki vault into LoRA adapters and writes a routing-metadata
sidecar (`<adapter_id>.meta.json`) next to each adapter. Run as a
CI/CD step (`--once`) or as a long-running watcher (`--watch`).

```pwsh
uv run llm-patch-wiki-agent daemon `
  --wiki-dir .\docs --adapter-dir .\artifacts\adapters `
  --checkpoint-dir .\models\t2l --once
```

### Phase 2 — HTTP inference gateway

The gateway selects the correct adapter via an `IAdapterRouter` strategy
(default: `MetadataExactMatchRouter`, exact match on `context_id`),
loads it on-demand, and reuses the runtime via a per-adapter cache.

```pwsh
uv run llm-patch-wiki-agent serve `
  --adapter-dir .\artifacts\adapters `
  --model-id google/gemma-2-2b-it `
  --host 127.0.0.1 --port 8765
```

```pwsh
curl -s -X POST http://127.0.0.1:8765/v1/chat -H "content-type: application/json" -d '{
  "messages": [{"role": "user", "content": "How do I implement v2 auth?"}],
  "metadata":  {"context_id": "api-v2-auth"}
}'
```

Endpoints: `GET /health`, `GET /v1/adapters`, `POST /v1/route`, `POST /v1/chat`.

### Phase 3 — MCP server

Exposes the pipeline to autonomous agents (Claude Desktop, etc.) over MCP.
Tools: `internalize_knowledge(path, context_id?, tags?, summary?)`,
`list_adapters()`, `chat_with_adapter(adapter_id, query)`.

```pwsh
uv run llm-patch-wiki-agent mcp `
  --adapter-dir .\artifacts\adapters `
  --checkpoint-dir .\models\t2l `
  --model-id google/gemma-2-2b-it `
  --transport stdio
```

See the root [SPEC.md](../../SPEC.md) for the governing engineering
specification, [NOTES.md](../../NOTES.md) for the use-case blueprint,
and [AGENTS.md](AGENTS.md) for the per-project agent contract.
