"""``llm-patch --help`` must not import heavy ML deps."""

from __future__ import annotations

import subprocess
import sys


def test_help_does_not_import_torch_or_peft() -> None:
    """Cold-start ``--help`` should stay snappy: no torch/peft/transformers."""
    code = (
        "import sys\n"
        "from click.testing import CliRunner\n"
        "from llm_patch.cli import cli\n"
        "result = CliRunner().invoke(cli, ['--help'])\n"
        "assert result.exit_code == 0, result.output\n"
        "heavy = {m for m in ('torch', 'peft', 'transformers') if m in sys.modules}\n"
        "if heavy:\n"
        "    raise SystemExit('heavy import on --help: ' + ','.join(sorted(heavy)))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)


def test_advanced_groups_hidden_by_default() -> None:
    """``--help`` should hide advanced groups unless LLM_PATCH_ADVANCED=1."""
    from click.testing import CliRunner

    from llm_patch.cli import cli

    result = CliRunner(env={"LLM_PATCH_ADVANCED": ""}).invoke(cli, ["--help"])
    assert result.exit_code == 0
    # Primary verbs visible.
    for verb in ("init", "doctor", "compile", "chat", "push", "pull", "version"):
        assert verb in result.output, f"missing primary verb: {verb}"


def test_primary_verbs_are_registered() -> None:
    from llm_patch.cli import cli

    names = set(cli.commands)
    assert {"init", "doctor", "compile", "watch", "chat", "push", "pull", "hub", "serve", "version"} <= names
    # Hidden groups are still registered (reachable by name).
    assert {"adapter", "model", "source", "wiki"} <= names
