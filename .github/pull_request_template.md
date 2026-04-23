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
- [ ] Tests added/updated; **engine baseline `216 passed, 3 skipped` preserved** (or change justified below).
- [ ] `make check` passes locally (`lint + typecheck + check-layering + test`).
- [ ] Public API change? → updated `__init__.py` `__all__`, `CHANGELOG.md`, and bumped version per SemVer.
- [ ] Architectural / dependency-direction change? → ADR added under `docs/adr/`.
- [ ] CHANGELOG.md `[Unreleased]` updated for every affected project.

## Baseline Justification (if test count changed)

<!-- Required if engine test count != 216 passed / 3 skipped. -->
