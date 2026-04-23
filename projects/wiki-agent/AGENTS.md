# AGENTS — `llm-patch-wiki-agent`

Per-project agent contract for the wiki-specialized use-case. Read with
the root [SPEC.md](../../SPEC.md) and root [AGENTS.md](../../AGENTS.md).

## Goal

Demonstrate the **engine → use-case** layering by composing the public
`llm_patch` API into a wiki-specialized chat agent with its own CLI.

## Public API

- `llm_patch_wiki_agent.WikiAgent`, `WikiAgentConfig`, and `WikiAgentInfo`.
- Console script `llm-patch-wiki-agent` (entry: `llm_patch_wiki_agent.cli:main`).

## Allowed Dependencies

- `llm-patch` — **only** symbols re-exported from `llm_patch.__init__`.
- `llm-patch-shared` — for cross-project utilities and error types.
- `click>=8.0` — CLI framework.
- Adding any other runtime dependency requires an ADR.

## Do

- Compose the engine via dependency injection (pass `IDataSource`,
  `IWeightGenerator`, `IAdapterRepository`, etc. into use-case classes).
- Write tests against the **public** engine API, not internals.
- Keep CLI commands thin — delegate to `WikiAgent` methods.

## Don't

- Don't import from `llm_patch.core.*`, `llm_patch.wiki.*` internals,
  `llm_patch.runtime.*`, or any other module not re-exported from
  `llm_patch.__init__`. The engine boundary is enforced by
  [tools/check_layering.py](../../tools/check_layering.py).
- Don't subclass engine classes; compose them.
- Don't add per-tenant or deployment glue here — that belongs in a
  separate use-case project (e.g., a future `chat-web` or `agent-deployer`).

## Test

```pwsh
uv run --package llm-patch-wiki-agent pytest -q
```
