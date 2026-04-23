---
applyTo: "**/tests/**/*.py"
description: "Testing conventions for the llm-patch monorepo: AAA, one behavior per test, no network, mock at the boundary, TDD for new public API."
---

# Tests — Conventions

These rules apply to every test file under any `tests/` directory.

## Structure

- **Arrange / Act / Assert** — separate the three phases visibly with
  blank lines.
- **One behavior per test.** A test name should read like a sentence:
  `test_<subject>_<expected_behavior>_when_<condition>`.
- Place fast tests under `tests/unit/`, slow / I/O-dependent tests
  under `tests/integration/` and mark them `@pytest.mark.integration`.

## Boundaries

- **No network** in unit tests. Use `httpx.MockTransport` or stubs.
- **No real model loads** in unit tests. Mock `IModelProvider` /
  `IAdapterLoader`.
- File I/O only via the `tmp_path` fixture. Never write under the repo.

## Mocking

- Mock at the **boundary** of the unit under test, not in the middle.
  If you find yourself mocking an internal collaborator, the test is
  probably exercising the wrong unit.
- Prefer hand-rolled fakes for ABCs (`class FakeRepo(IAdapterRepository)`)
  over `MagicMock` — they document the contract and break loudly when
  the contract changes.

## Public API Tests

- Any new public symbol must have at least one test that imports it
  from the **top-level package** (`from llm_patch import Foo`,
  `from llm_patch_wiki_agent import Bar`). This locks the public path
  per [ADR-0003](../../docs/adr/0003-public-api-policy.md).

## Baselines

- The engine baseline is `216 passed, 3 skipped`. A change that alters
  the count must justify it in the PR description.
