"""Tests for the end-to-end demo scenario.

Validates that the full pipeline genuinely works:
- Wiki content is detected and processed
- Adapters accumulate as wiki grows
- Content updates cause adapter regeneration
- Model answers improve with more knowledge
- Live watcher detects new files
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

# Repo root contains the examples/ directory of demo scripts. Since examples/
# is intentionally not a Python package (no __init__.py), load the demo module
# by file path via importlib so the integration test stays decoupled from any
# packaging assumptions about examples/.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEMO_PATH = _REPO_ROOT / "examples" / "e2e" / "demo_e2e_scenario.py"


def _load_run_scenario() -> Any:
    """Load run_scenario from examples/e2e/demo_e2e_scenario.py by path."""
    module_name = "_llm_patch_demo_e2e_scenario"
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached.run_scenario
    spec = importlib.util.spec_from_file_location(module_name, _DEMO_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load demo module from {_DEMO_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.run_scenario


@pytest.fixture()
def scenario_wiki(tmp_path: Path) -> Path:
    """Provide a clean temporary wiki directory."""
    wiki = tmp_path / "demo_wiki"
    wiki.mkdir()
    return wiki


class TestEndToEndScenario:
    """Full scenario validation."""

    def _run(self, wiki_dir: Path) -> dict[str, Any]:
        run_scenario = _load_run_scenario()
        return run_scenario(wiki_dir)

    def test_scenario_completes(self, scenario_wiki: Path) -> None:
        results = self._run(scenario_wiki)
        assert "steps" in results
        assert len(results["steps"]) == 5

    def test_step1_baseline_no_adapters(self, scenario_wiki: Path) -> None:
        results = self._run(scenario_wiki)
        step1 = results["steps"][0]
        assert step1["name"] == "baseline"
        assert step1["adapters"] == 0
        for ans in step1["answers"]:
            assert "no adapters loaded" in ans.lower()

    def test_step2_transformer_adds_knowledge(self, scenario_wiki: Path) -> None:
        results = self._run(scenario_wiki)
        step2 = results["steps"][1]
        assert step2["name"] == "transformer_paper"
        assert step2["adapters"] > 0
        assert "enhanced model" in step2["answers"][0].lower()

    def test_step3_lora_adds_more(self, scenario_wiki: Path) -> None:
        results = self._run(scenario_wiki)
        step3 = results["steps"][2]
        assert step3["name"] == "lora_paper_added"
        assert step3["adapters"] > results["steps"][1]["adapters"]
        assert "enhanced model" in step3["answers"][1].lower()

    def test_step4_update_changes_hash(self, scenario_wiki: Path) -> None:
        results = self._run(scenario_wiki)
        step4 = results["steps"][3]
        assert step4["name"] == "transformer_updated"
        assert step4["hash_changed"] is True
        assert "enhanced model" in step4["answers"][0].lower()

    def test_step5_final_adapters(self, scenario_wiki: Path) -> None:
        results = self._run(scenario_wiki)
        assert results["total_adapters"] > 0
        assert "sources/gpt3-paper" in results["all_adapter_ids"]

    def test_knowledge_progression(self, scenario_wiki: Path) -> None:
        """Verify adapters accumulate monotonically across steps."""
        results = self._run(scenario_wiki)
        counts = [s["adapters"] for s in results["steps"]]
        assert counts[0] == 0
        assert counts[1] > 0
        assert counts[2] > counts[1]
        assert counts[3] >= counts[2]
        assert counts[4] > counts[3]

    def test_answers_improve_over_steps(self, scenario_wiki: Path) -> None:
        """Step 1 answers are generic; later steps have enhanced knowledge."""
        results = self._run(scenario_wiki)
        for ans in results["steps"][0]["answers"]:
            assert "base model" in ans.lower()
        for step_idx in (2, 3, 4):
            enhanced = sum(
                1
                for ans in results["steps"][step_idx]["answers"]
                if "enhanced model" in ans.lower()
            )
            assert enhanced >= 2, (
                f"Step {step_idx + 1}: expected >= 2 enhanced answers, got {enhanced}"
            )
