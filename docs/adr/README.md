# Architecture Decision Records

This directory holds the project's [Architecture Decision Records](https://adr.github.io/)
in the [MADR](https://adr.github.io/madr/) format. Each ADR captures one
significant decision: its context, the decision itself, the consequences,
and the alternatives that were considered.

## How to add an ADR

1. Copy [0000-template.md](0000-template.md) to `NNNN-short-title.md`
   where `NNNN` is the next free 4-digit number.
2. Set status to `Proposed`. Discuss in a PR.
3. On merge, update status to `Accepted`. Reference the ADR from the
   relevant `AGENTS.md` or `SPEC.md` section.
4. Superseding an ADR: keep the old file, set its status to
   `Superseded by NNNN`, add a back-link from the new ADR.

## Index

| # | Title | Status |
|---|---|---|
| [0001](0001-uv-workspace.md) | Adopt a uv workspace monorepo | Accepted |
| [0002](0002-layered-architecture.md) | Three-layer architecture (use-cases → engine → shared) | Accepted |
| [0003](0003-public-api-policy.md) | Public API = top-level `__init__.py` re-exports only | Accepted |
| [0004](0004-testing-strategy.md) | TDD for new public API; coverage thresholds | Accepted |
| [0005](0005-engine-shared-errors.md) | Engine dependency on utils for error types | Accepted |
| [0006](0006-distributed-adapter-registry.md) | Distributed adapter registry & runtime boundaries | Accepted |
| [0007](0007-adapter-manifest-v2.md) | Adapter identity & manifest v2 | Accepted |
| [0008](0008-plugin-discovery.md) | Plugin discovery via env vars + entry points | Accepted |
| [0009](0009-monorepo-structural-unification.md) | Monorepo structural unification (utils rename, single AGENTS.md, use-case template) | Accepted |
