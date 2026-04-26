# Community

`llm-patch` is community-first. The CLI is the GTM lever; the engine is
a clean library you can extend without forking; and the docs are the
glue.

## Where to go

| I want to… | Channel |
|---|---|
| Ask a question | [GitHub Discussions](https://github.com/seismael/llm-patch/discussions) |
| Report a bug | [Issues — bug template](https://github.com/seismael/llm-patch/issues/new?template=bug_report.yml) |
| Propose a feature | [Issues — feature template](https://github.com/seismael/llm-patch/issues/new?template=feature_request.yml) |
| Contribute a plugin (source / generator / registry) | [EXTENDING.md](EXTENDING.md) + [issue template](https://github.com/seismael/llm-patch/issues/new?template=new_source_plugin.yml) |
| Improve docs | [docs issue template](https://github.com/seismael/llm-patch/issues/new?template=documentation.yml) |
| Real-time chat | (Discord coming — placeholder) |

## Plugin gallery

`llm-patch` discovers plugins through Python entry points or the
`LLM_PATCH_PLUGIN_<KIND>` environment variables ([ADR-0008](adr/0008-plugin-discovery.md),
[EXTENDING.md](EXTENDING.md)). The community is encouraged to publish
plugins as separate PyPI packages.

| Plugin | Kind | Author | Status |
|---|---|---|---|
| _your plugin here_ | — | — | open a PR adding a row |

## Roadmap

We track work in GitHub Issues with labels:

- `area:cli` — CLI / DX work.
- `area:engine` — engine ABCs / pipelines.
- `area:registry` — adapter market & hubs.
- `good-first-issue` — friendly entry points.
- `help-wanted` — actively seeking contributors.

Architectural decisions land as ADRs in [docs/adr/](adr/). Read those
before proposing structural changes; refer to
[CONTRIBUTING.md](../CONTRIBUTING.md) for the contribution checklist.

## Code of Conduct

See [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md). TL;DR: be kind, be
specific, be patient.
