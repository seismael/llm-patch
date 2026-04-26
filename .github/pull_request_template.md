<!--
Thanks for contributing! Please confirm the items below before requesting review.
See SPEC.md and the per-project AGENTS.md for the full rules.
-->

## What & Why

<!-- One-paragraph summary of the change and the motivation. -->

## Affected Project(s)

- [ ] `llm-patch` (engine)
- [ ] `llm-patch-shared`
- [ ] `llm-patch-wiki-agent`
- [ ] tooling / docs / CI only

## Checklist

- [ ] Read the relevant `AGENTS.md` (root + per-project).
- [ ] Tests added/updated; **engine baseline preserved** (or change justified below).
- [ ] `make check` passes locally (`lint + typecheck + check-layering + test`).
- [ ] Public API change? → updated `__init__.py` `__all__`, `CHANGELOG.md`, and bumped version per SemVer.
- [ ] Architectural / dependency-direction change? → ADR added under `docs/adr/`.
- [ ] CHANGELOG.md `[Unreleased]` updated for every affected project.
- [ ] CLI surface change? → `llm-patch --help` still snappy; primary verbs unchanged or migration noted.
- [ ] New plugin? → followed [docs/EXTENDING.md](../docs/EXTENDING.md); env-var + entry-point both verified.

## Baseline Justification (if test count changed)

<!-- Engine baseline tracked in projects/llm-patch/CHANGELOG.md. -->
<!-- Required if engine test count differs from the previous release's baseline. -->
