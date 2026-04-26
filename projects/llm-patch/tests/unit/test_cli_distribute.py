"""Tests for ``llm-patch push|pull|hub`` commands.

Use a fake :class:`IAdapterRegistryClient` injected via the canonical
``LLM_PATCH_PLUGIN_REGISTRY`` environment variable. The deprecated
``LLM_PATCH_REGISTRY`` alias is still recognized (with a warning); see
the ``TestRegistryEnvAlias`` class below.
"""

from __future__ import annotations

import json
import sys
from typing import Any

import pytest
from click.testing import CliRunner

from llm_patch import AdapterManifest, AdapterRef
from llm_patch.cli import cli
from llm_patch.core.interfaces import IAdapterRegistryClient


# ── Fake registry installed via env var ─────────────────────────────


class _FakeRegistry(IAdapterRegistryClient):
    def search(self, query: str, *, limit: int = 10) -> list[AdapterManifest]:
        return [
            AdapterManifest(
                adapter_id="acme__x__v1",
                rank=8,
                target_modules=["q"],
                storage_uri="/tmp",
                namespace="acme/x",
                version="1.0.0",
                description="hit",
            )
        ][:limit]

    def resolve(self, ref: AdapterRef) -> AdapterManifest:
        return AdapterManifest(
            adapter_id=ref.adapter_id,
            rank=8,
            target_modules=["q"],
            storage_uri="/tmp",
            namespace=f"{ref.namespace}/{ref.name}",
            version="1.0.0",
        )

    def pull(self, ref: AdapterRef) -> AdapterManifest:
        return self.resolve(ref)

    def push(self, adapter_id: str, ref: AdapterRef) -> AdapterManifest:
        return AdapterManifest(
            adapter_id=ref.adapter_id,
            rank=8,
            target_modules=["q"],
            storage_uri="/tmp",
            checksum_sha256="b" * 64,
        )


def _factory() -> _FakeRegistry:
    return _FakeRegistry()


# Register the factory on this module so 'tests.unit.test_cli_distribute:_factory'
# resolves at import time.
_THIS = sys.modules[__name__]
_THIS._factory = _factory  # type: ignore[attr-defined]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def with_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PATCH_REGISTRY", raising=False)
    monkeypatch.setenv("LLM_PATCH_PLUGIN_REGISTRY", f"{__name__}:_factory")


# ── Tests ────────────────────────────────────────────────────────────


class TestPushPull:
    def test_pull_emits_uri(self, runner: CliRunner, with_registry: None) -> None:
        result = runner.invoke(cli, ["pull", "hub://acme/x:v1"])
        assert result.exit_code == 0, result.output
        assert "hub://acme/x:v1" in result.output

    def test_pull_json_mode(self, runner: CliRunner, with_registry: None) -> None:
        result = runner.invoke(cli, ["pull", "hub://acme/x:v1", "--json"])
        assert result.exit_code == 0, result.output
        payload = json.loads(result.output.strip())
        assert payload["pulled"] == "hub://acme/x:v1"

    def test_pull_without_registry_fails_cleanly(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("LLM_PATCH_PLUGIN_REGISTRY", raising=False)
        monkeypatch.delenv("LLM_PATCH_REGISTRY", raising=False)
        result = runner.invoke(cli, ["pull", "hub://acme/x:v1"])
        assert result.exit_code != 0
        assert "registry" in result.output.lower()

    def test_push_rejects_unknown_scheme(
        self, runner: CliRunner, with_registry: None, tmp_path: Any
    ) -> None:
        adapter_dir = tmp_path / "ad"
        adapter_dir.mkdir()
        result = runner.invoke(
            cli, ["push", str(adapter_dir), "--target", "ftp://nope"]
        )
        assert result.exit_code != 0
        assert "scheme" in result.output.lower()

    def test_push_hf_scheme_is_deferred(
        self, runner: CliRunner, with_registry: None, tmp_path: Any
    ) -> None:
        adapter_dir = tmp_path / "ad"
        adapter_dir.mkdir()
        result = runner.invoke(
            cli, ["push", str(adapter_dir), "--target", "hf://owner/repo"]
        )
        assert result.exit_code != 0
        assert "not yet implemented" in result.output.lower()


class TestHubGroup:
    def test_search_text_output(self, runner: CliRunner, with_registry: None) -> None:
        result = runner.invoke(cli, ["hub", "search", "react"])
        assert result.exit_code == 0, result.output
        assert "acme__x__v1" in result.output

    def test_search_json_output(self, runner: CliRunner, with_registry: None) -> None:
        result = runner.invoke(cli, ["hub", "search", "react", "--json"])
        assert result.exit_code == 0
        payload = json.loads(result.output.strip())
        assert payload[0]["adapter_id"] == "acme__x__v1"

    def test_info_resolves(self, runner: CliRunner, with_registry: None) -> None:
        result = runner.invoke(cli, ["hub", "info", "hub://acme/x:v1"])
        assert result.exit_code == 0, result.output
        assert "v1" in result.output


class TestTopLevelVerbs:
    def test_compile_top_level_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["compile", "--help"])
        assert result.exit_code == 0
        assert "compile" in result.output.lower()

    def test_chat_top_level_help(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.output.lower()


class TestRegistryEnvAlias:
    """Phase-1 contract: env-var unification with backward-compat alias."""

    def test_legacy_env_var_resolves_with_deprecation_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import warnings as _w

        from llm_patch.cli.distribute import _resolve_registry_spec

        monkeypatch.delenv("LLM_PATCH_PLUGIN_REGISTRY", raising=False)
        monkeypatch.setenv("LLM_PATCH_REGISTRY", f"{__name__}:_factory")
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            spec = _resolve_registry_spec()
        assert spec == f"{__name__}:_factory"
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "LLM_PATCH_REGISTRY" in str(w.message)
            for w in caught
        )

    def test_canonical_env_var_wins_over_legacy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_patch.cli.distribute import _resolve_registry_spec

        monkeypatch.setenv("LLM_PATCH_PLUGIN_REGISTRY", "canonical:factory")
        monkeypatch.setenv("LLM_PATCH_REGISTRY", "legacy:factory")
        assert _resolve_registry_spec() == "canonical:factory"

    def test_neither_var_set_raises_registry_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from llm_patch.cli.distribute import _resolve_registry_spec
        from llm_patch_utils import RegistryUnavailableError

        monkeypatch.delenv("LLM_PATCH_PLUGIN_REGISTRY", raising=False)
        monkeypatch.delenv("LLM_PATCH_REGISTRY", raising=False)
        with pytest.raises(RegistryUnavailableError):
            _resolve_registry_spec()
