# 0004 — TDD for New Public API; Coverage Thresholds

- **Status**: Accepted
- **Date**: 2026-04-21
- **Tags**: testing, quality-gates

## Context

The engine has 216 passing unit/integration tests today. We want to
preserve that as a hard baseline and raise the bar on net-new code
without imposing churn on legacy code that already works.

## Decision

1. **Test-first for new public API**: any new symbol that will be
   re-exported from a project's `__init__.py` must land with at least
   one unit test that imports it from the top-level package and
   exercises one happy-path behavior.
2. **Coverage thresholds (branch coverage)**:
   - Engine `llm_patch.core.*`: **≥ 85%**.
   - Per-project overall: **≥ 75%**.
   - Changed lines in a PR: **≥ 80%**.
3. **Test layout** in every project:
   - `tests/unit/` — default suite, fast, no network, filesystem only
     under `tmp_path`.
   - `tests/integration/` — marked `@pytest.mark.integration`. Run with
     `pytest -m integration`.
4. **Baselines are release blockers**:
   - Engine: `216 passed, 3 skipped`.
   - shared-utils (now `utils`, see [ADR-0009](0009-monorepo-structural-unification.md)): smoke tests pass.
   - wiki-agent: smoke tests pass.

## Consequences

### Positive

- New public API is provably exercised before merge.
- Coverage thresholds catch silent regressions in the engine core.
- Clear unit/integration split keeps the default suite fast.

### Negative / Trade-offs

- Coverage tooling adds ~1–2s to local test runs.
- Tests for ABCs require concrete fakes; this is a one-time cost per
  ABC and pays off in confidence.

## Alternatives Considered

### Alternative A — Mandatory TDD everywhere (including bug fixes)

Stronger but higher friction. We require regression tests for bug fixes
but don't mandate strict red-green-refactor for non-public changes.

### Alternative B — No coverage threshold, just a number to watch

Drift is inevitable without a hard floor. Rejected.

## References

- [SPEC.md §5 Testing — TDD Policy](../../SPEC.md#5-testing--tdd-policy)
