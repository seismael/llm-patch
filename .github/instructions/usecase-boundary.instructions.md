---
applyTo: "projects/wiki-agent/**"
description: "Use-case boundary rules for wiki-agent: import only from llm_patch's public top-level API; no engine internals."
---

# Use-Case Boundary Rules — wiki-agent

These rules apply to every file under `projects/wiki-agent/`. They
implement the use-case layer of the architecture in
[SPEC.md §1](../../SPEC.md#1-architectural-layering),
[ADR-0002](../../docs/adr/0002-layered-architecture.md), and
[ADR-0003](../../docs/adr/0003-public-api-policy.md).

## Hard Rules

- **Import only from the top-level engine package**:
  ```python
  # OK
  from llm_patch import CompilePipeline, WikiKnowledgeSource, IAgentRuntime
  # NOT OK
  from llm_patch.pipelines.compile import CompilePipeline
  from llm_patch.core.interfaces import IAgentRuntime
  from llm_patch.wiki.manager import WikiManager  # internal — use llm_patch.WikiManager
  ```
- **Compose, don't subclass** engine classes. The engine ABCs exist so
  you can plug in a new implementation; subclassing a concrete engine
  class couples you to its internals.
- **Stay thin**: CLI commands delegate to a single `WikiAgent` method.
  Keep business logic in `WikiAgent`, not in `cli.py`.
- **Add a test** for every new public symbol, importing it from
  `llm_patch_wiki_agent` (the top-level), not from a submodule.

## Allowed Runtime Dependencies

- `llm-patch` (workspace).
- `llm-patch-shared` (workspace).
- `click>=8.0`.
- Anything else requires an ADR.
